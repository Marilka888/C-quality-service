"""
PR-K real-package follow-up tests:

Driven by two real .docx packages (Cherevuyhho + Polyakov) processed via
docback → C-quality with qwen2.5:3b. Three regressions surfaced that the
synthetic smoke test missed:

  P0 — `evidence_floor=0.5` was too aggressive on real packages with
       BoW retrieval; perfectly grounded COVERED verdicts (conf=1.0)
       got demoted to MISSING_LOW_GROUNDING because retrieval ~0.44.
       Fix: split `low_confidence` (UI-dim flag) from `grounding_failed`
       (true grounding-gate flag); the aggregator only treats the latter
       as ungrounded. Default floor lowered from 0.50 to 0.30.

  P1 — `qwen2.5:3b` choked on prompt v5 metadata-field prefix and echoed
       the field labels back into JSON, breaking the parser. Fix:
       compact prompt for small models (≤4B), selected automatically by
       model name or via CQUALITY_PROMPT_VARIANT=compact|full env var.

  P2 — `_extract_constraints` picked up GOST/section/footnote numbers
       (19.301, 79, 18, 4.1, 5.1, …) as bogus generic constraints,
       polluting `numeric_conflict` rule and `uncoveredAspects`. Fix:
       drop unitless numeric matches whose context contains GOST / ГОСТ
       / [N] / пункт / статья / таблица etc.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.aggregate_coverage import (
    SUBCODE_COVERED,
    SUBCODE_MISSING_LOW_GROUNDING,
    CoverageAggregator,
)
from app.application.use_cases.build_requirements import (
    _extract_constraints,
    _is_reference_number,
)
from app.core.config import (
    CoverageAggregatorConfig,
    CoverageConfig,
    CoverageDebugConfig,
)
from app.domain.c_quality_enums import (
    CoverageStatus,
    LLMLabel,
    RequirementType,
)
from app.domain.c_quality_models import (
    Constraint,
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
    RetrievedCandidate,
)
from app.infrastructure.llm.prompts import (
    build_judge_prompt,
    build_judge_prompt_compact,
    should_use_compact_prompt,
)


# ── P0: split low_confidence / grounding_failed semantics ──────────────


class TestP0LowConfidenceSplit:
    def setup_method(self):
        self.agg = CoverageAggregator()
        self.aggcfg = CoverageAggregatorConfig()
        self.dbgcfg = CoverageDebugConfig(enabled=True)

    def _req(self) -> RequirementUnit:
        return RequirementUnit(
            req_id="r1", source_document_id="doc-tz",
            text="Время восстановления не должно превышать 24 часа.",
            normalized_text="время восстановления не должно превышать 24 часа",
            requirement_type=RequirementType.RELIABILITY,
        )

    def _unit(self) -> CoverageUnit:
        return CoverageUnit(
            unit_id="u1", target_document_id="doc-pmi", target_doc_role="pmi",
            text="При отказе время восстановления не должно превышать времени, "
                 "требующегося на перезагрузку операционной системы.",
            normalized_text="при отказе время восстановления...",
        )

    def test_grounded_covered_with_below_floor_retrieval_stays_covered(self):
        """Real-package symptom (Polyakov 0.20::sent1):
        LLM said COVERED conf=1.0 with proper citation, retrieval=0.44 < floor=0.5.
        Old code marked low_confidence=True → aggregator interpreted as
        ungrounded → MISSING_LOW_GROUNDING. After P0: status stays
        COVERED, low_confidence flag preserved on the row for UI dimming."""
        j = PairJudgment(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            llm_label=LLMLabel.COVERED, rule_adjusted_label=LLMLabel.COVERED,
            llm_confidence=1.0,
            low_confidence=True,         # below evidence_floor → UI-dim flag
            grounding_failed=False,      # but properly grounded
        )
        cand = RetrievedCandidate(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            retrieval_score=0.44,
        )
        res = self.agg.aggregate(
            requirement=self._req(),
            judgments=[j],
            candidates_by_unit_id={"u1": cand},
            units_by_id={"u1": self._unit()},
            target_document_id="doc-pmi", target_doc_role="pmi",
            aggregator_cfg=self.aggcfg, debug_cfg=self.dbgcfg,
        )
        assert res.status == CoverageStatus.COVERED, (
            f"grounded COVERED with below-floor retrieval must stay COVERED; "
            f"got {res.status} subcode={res.status_subcode}"
        )
        assert res.status_subcode == SUBCODE_COVERED
        assert res.low_confidence is True   # row-level dim flag preserved

    def test_ungrounded_covered_demoted(self):
        """Sanity guard: when LLM hallucinated a citation that's NOT in
        evidence (`grounding_failed=True`), aggregator must still demote
        COVERED → MISSING_LOW_GROUNDING. P0 split must not weaken the
        original BUG-3 grounding gate."""
        j = PairJudgment(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            llm_label=LLMLabel.COVERED, rule_adjusted_label=LLMLabel.COVERED,
            llm_confidence=0.85,
            low_confidence=True,
            grounding_failed=True,       # actually ungrounded
        )
        cand = RetrievedCandidate(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            retrieval_score=0.55,
        )
        res = self.agg.aggregate(
            requirement=self._req(),
            judgments=[j],
            candidates_by_unit_id={"u1": cand},
            units_by_id={"u1": self._unit()},
            target_document_id="doc-pmi", target_doc_role="pmi",
            aggregator_cfg=self.aggcfg, debug_cfg=self.dbgcfg,
        )
        assert res.status == CoverageStatus.MISSING
        assert res.status_subcode == SUBCODE_MISSING_LOW_GROUNDING

    def test_default_evidence_floor_lowered_to_0_30(self):
        """The default floor is now 0.30. Anything below this is so weak
        that even valid LLM judgments can't be trusted."""
        cfg = CoverageConfig()
        assert cfg.retrieval.evidence_floor == pytest.approx(0.30)


# ── P1: compact prompt for small models ────────────────────────────────


class TestP1CompactPromptSelection:
    @pytest.mark.parametrize("model, expected", [
        # Small models → compact
        ("qwen2.5:3b", True),
        ("qwen2.5:1.5b", True),
        ("llama3.2:3b", True),
        ("llama3.2:1b", True),
        ("phi3:3.8b", True),
        ("tinyllama", True),
        ("gemma:2b", True),
        # Larger models → full prompt
        ("qwen2.5:7b", False),
        ("llama3:8b", False),
        ("mixtral:8x7b", False),
        ("qwen2.5:14b", False),
        ("qwen2.5:72b", False),
        # Edge cases
        (None, False),
        ("", False),
    ])
    def test_compact_selection_by_model_name(self, model, expected):
        assert should_use_compact_prompt(model) is expected

    def test_env_var_override_compact(self, monkeypatch):
        """CQUALITY_PROMPT_VARIANT=compact forces compact even on large
        models (researcher override for prompt-engineering experiments)."""
        monkeypatch.setenv("CQUALITY_PROMPT_VARIANT", "compact")
        assert should_use_compact_prompt("qwen2.5:72b") is True

    def test_env_var_override_full(self, monkeypatch):
        """CQUALITY_PROMPT_VARIANT=full forces full even on tiny models."""
        monkeypatch.setenv("CQUALITY_PROMPT_VARIANT", "full")
        assert should_use_compact_prompt("qwen2.5:1.5b") is False

    def test_compact_prompt_is_significantly_shorter(self):
        """Compact prompt must drop the labelled-metadata block that
        confused 3B in real-package run. As a rough check, compact user
        prompt has fewer than half the lines of the full one for a
        typical request."""
        req = RequirementUnit(
            req_id="r", source_document_id="d", text="Текст требования.",
            normalized_text="текст требования",
            requirement_type=RequirementType.SECURITY,
        )
        unit = CoverageUnit(
            unit_id="u", target_document_id="d", target_doc_role="pmi",
            text="Текст фрагмента.", normalized_text="текст фрагмента",
        )
        _, full_user = build_judge_prompt(req, unit)
        _, compact_user = build_judge_prompt_compact(req, unit)
        full_lines = full_user.count("\n")
        compact_lines = compact_user.count("\n")
        assert compact_lines < full_lines, (
            f"compact ({compact_lines}) should be shorter than full ({full_lines})"
        )

    def test_compact_prompt_omits_metadata_field_labels(self):
        """The compact prompt must NOT contain the labelled-metadata
        fields ("Применимость к этому документу", "Уровень покрытия")
        that small models echoed into their JSON output."""
        req = RequirementUnit(
            req_id="r", source_document_id="d",
            text="Документация должна быть в LMS.",
            normalized_text="документация должна быть в lms",
            requirement_type=RequirementType.DELIVERY_REQUIREMENT,
        )
        unit = CoverageUnit(
            unit_id="u", target_document_id="d", target_doc_role="pz",
            text="x", normalized_text="x",
        )
        _, compact_user = build_judge_prompt_compact(req, unit)
        # Forbidden labels — those that the 3B echoed back as JSON keys.
        forbidden = [
            "Применимость к этому документу",
            "Уровень покрытия:",
            "[prompt_version=",
            "Тип документа-источника фрагмента:",
        ]
        for label in forbidden:
            assert label not in compact_user, (
                f"compact prompt must not contain metadata label {label!r}"
            )


# ── P2: GOST / section / footnote numbers filter ───────────────────────


class TestP2ConstraintFilter:
    @pytest.mark.parametrize("text, expected_count", [
        # Real-package symptom (Polyakov 0.41::sent4): "ГОСТ 19.301-79 [18]
        # п.4.1.1 ... п.5.1, 5.2" — old code extracted seven bogus numeric
        # constraints. After P2: zero (no measurable units anywhere).
        (
            "Программа и методика испытаний (ГОСТ 19.301-79) [18] в котором "
            "указывают перечень функций (п. 4.1.1) и документации (п. 5.1, 5.2).",
            0,
        ),
        # Pure GOST reference.
        ("См. ГОСТ Р 34.601-90 для процесса разработки.", 0),
        # Pure footnote refs.
        ("Согласно [12] и [18, c.5], архитектура двухуровневая.", 0),
        # Pure section refs.
        ("В пункте 4.1.1 настоящего ТЗ описаны функции.", 0),
        # Article ref.
        ("Регулируется статьёй 153 УК РФ.", 0),
        # Mixed: GOST AND a real constraint (90 days).
        (
            "Согласно ГОСТ 19.301-79 [18] журнал хранится не менее 90 дней.",
            1,  # only the 90 days
        ),
        # Real constraints must NOT be filtered.
        ("Время отклика не должно превышать 2 секунд при 100 запросах.", 2),
        ("Хранить журнал 90 дней.", 1),
    ])
    def test_real_constraints_kept_refs_dropped(self, text, expected_count):
        cs = _extract_constraints(text)
        actual = len(cs)
        assert actual == expected_count, (
            f"text={text!r}: expected {expected_count} constraints, got {actual}: {cs}"
        )

    def test_polyakov_0_41_sent4_zero_bogus_constraints(self):
        """Verbatim Polyakov 0.41::sent4 fragment — old code extracted
        7 generic constraints (19.301, 79, 18, 4.1, 1, 5.1, 5.2) all
        of which were section/standard/footnote refs."""
        text = (
            "Программа и методика испытаний (ГОСТ 19.301-79) [18] в котором "
            "указывают: перечень функций программы, выделенных в программе "
            "для испытаний, и перечень требований, которым должны "
            "соответствовать эти функции (со ссылкой на пункт 4.1.1. "
            "настоящего технического задания); перечень необходимой "
            "документации и требования к ней (со ссылкой на пункты 5.1 и 5.2 "
            "настоящего технического задания); методы испытаний и обработки "
            "информации; технические средства и порядок проведения испытаний."
        )
        cs = _extract_constraints(text)
        assert cs == [], (
            f"Polyakov sent4 must extract zero constraints; got {cs}"
        )

    def test_is_reference_number_heuristic(self):
        """Direct test of the reference-context detector."""
        cases = [
            ("ГОСТ 19.301", 5, 11, True),       # "19.301" inside "ГОСТ ..."
            ("[18]", 1, 3, True),                # "18" inside footnote
            ("пункт 4.1.1", 6, 11, True),
            ("статья 153", 7, 10, True),
            ("90 дней", 0, 2, False),            # standalone number = real
            ("2 секунд", 0, 1, False),
        ]
        for text, ms, me, expected in cases:
            actual = _is_reference_number(text, ms, me)
            assert actual is expected, (
                f"_is_reference_number({text!r}, {ms}, {me}) = {actual} "
                f"(expected {expected})"
            )
