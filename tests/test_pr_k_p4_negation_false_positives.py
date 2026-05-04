"""
PR-K P4 regression tests: real-package false-CONFLICT cases from
Polyakov package (run-2 with P0/P1/P2 active).

After P0 lowered evidence_floor 0.5→0.30 and P2 extended _PROHIBITION_RE
to catch 'не должно' and other forms, the negation-contradiction rule
started firing on pairs with NO topical link. The numeric rule has had
a topical-link guard for a long time (shared kind / entity overlap /
LLM confidence); the negation rule didn't, so the same problem migrated
into the modality-mismatch path.

The four false-positive CONFLICT verdicts in the Polyakov-2 run that
this test file pins down:

  * 0.14::sent1 'время отклика 3 сек' vs Windows-config evidence
    (top-1 retrieval landed on a hardware-spec fragment because of
    BoW similarity coincidence; req has 'не должно', unit doesn't,
    and the rule fired CONFLICT despite zero topical link).

  * 0.17::sent3 'не должна аварийно завершать' vs admin-fragment in
    PZ — completely different topic; rule fires anyway.

  * 0.17::sent3 vs PMI 'должна продолжать корректно функционировать'
    — same-outcome that should have been caught by the existing
    pre-classified compatibility table, but the LLM's verdict at the
    time was confidently positive (PARTIAL conf=0.95) and we should
    have honoured it via a judge-strongly-positive guard.

  * 0.20::sent1 'время восстановления не должно превышать общее время
    на перезагрузку' vs PMI 'время восстановления не должно превышать
    времени, требующегося на перезагрузку' — semantically identical,
    same prohibition phrasing on both sides ("не должно превышать"),
    grounded COVERED conf=1.0. P4 same_upper_bound rule must catch.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.verify_pairs import (
    PairVerifier,
    _same_outcome_negation_compatible,
)
from app.domain.c_quality_enums import LLMLabel, Modality, RequirementType
from app.domain.c_quality_models import (
    Constraint,
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
)


def _req(text: str, req_type: RequirementType = RequirementType.RELIABILITY,
         constraints=None) -> RequirementUnit:
    from app.application.use_cases.build_requirements import _normalize_text
    return RequirementUnit(
        req_id="r1", source_document_id="doc-tz",
        text=text, normalized_text=_normalize_text(text),
        requirement_type=req_type,
        modality=Modality.MUST_NOT if "не должн" in text.lower() else Modality.MUST,
        constraints=constraints or [],
    )


def _unit(text: str, role: str = "pmi") -> CoverageUnit:
    from app.application.use_cases.build_requirements import _normalize_text
    return CoverageUnit(
        unit_id="u1", target_document_id=f"doc-{role}", target_doc_role=role,
        text=text, normalized_text=_normalize_text(text),
    )


def _judgment(label: LLMLabel, conf: float = 0.5) -> PairJudgment:
    return PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=label, rule_adjusted_label=label, llm_confidence=conf,
    )


class TestPolyakovFalseConflictTopicalGuard:
    """The four false-positive CONFLICTs from the Polyakov-2 run, plus
    run-4/5 post-fix F regressions (LLM-native CONFLICT with high
    confidence on off-topic evidence)."""

    def setup_method(self):
        self.verifier = PairVerifier()

    def test_0_14_sent1_perf_vs_windows_config_no_conflict(self):
        """0.14::sent1 — TZ "время отклика не должно превышать 3 секунд"
        accidentally paired with Windows hardware config (top-1
        BoW retrieval landed on the wrong fragment). No topical link
        whatsoever — must NOT fire CONFLICT through the negation rule."""
        req = _req(
            "Время отклика приложения при условии отсутствия сетевого "
            "взаимодействия не должно превышать 3 секунд.",
            req_type=RequirementType.PERFORMANCE,
            constraints=[Constraint(kind="response_time", operator="<=",
                                    value=3, unit="sec")],
        )
        unit = _unit(
            "Во время испытаний использовался персональный компьютер под "
            "управлением операционной системы Windows 10 Pro [10], "
            "оснащённый следующими техническими характеристиками: "
            "Процессор Intel Core i5-7500, 16 ГБ RAM, GTX 1060."
        )
        j = _judgment(LLMLabel.IRRELEVANT, conf=0.30)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label != LLMLabel.CONFLICT, (
            f"0.14::sent1 vs Windows config promoted to CONFLICT; "
            f"actions={out.verifier_actions}"
        )

    def test_0_17_sent3_avariyno_vs_admin_pz_no_conflict(self):
        """0.17::sent3 PZ — TZ "Система не должна аварийно завершать"
        vs PZ admin-fragment about role management. Different topic.
        Must NOT fire CONFLICT."""
        req = _req(
            "Система не должна аварийно завершать свою работу в случае "
            "возникновения ошибки.",
            req_type=RequirementType.RELIABILITY,
        )
        unit = _unit(
            "Администратор – имеет возможность выдавать роли другим "
            "пользователям, создавать новые коллекции и редактировать "
            "их иерархию, удалять уже принятые исследования или "
            "добавлять новые, без необходимости проходить модерацию.",
            role="pz",
        )
        j = _judgment(LLMLabel.IRRELEVANT, conf=0.20)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label != LLMLabel.CONFLICT, (
            f"avariyno vs admin paired with negation contradiction; "
            f"actions={out.verifier_actions}"
        )

    def test_0_17_sent3_llm_conflict_on_offtopic_pz_no_conflict(self):
        """PR-K post-fix F: 0.17::sent3 PZ — LLM hallucinated CONFLICT
        (conf=0.85) on off-topic PZ admin fragment. The negation rule
        must NOT confirm it using LLM confidence as a topical-link
        proxy. Real-world failure in Polyakov run-4 and run-5."""
        req = _req(
            "Система не должна аварийно завершать свою работу в случае "
            "возникновения ошибки.",
            req_type=RequirementType.RELIABILITY,
        )
        unit = _unit(
            "Администратор – имеет возможность выдавать роли другим "
            "пользователям, создавать новые коллекции и редактировать "
            "их иерархию, удалять уже принятые исследования или "
            "добавлять новые, без необходимости проходить модерацию.",
            role="pz",
        )
        # LLM was confidently wrong — this is the exact Polyakov failure mode.
        j = _judgment(LLMLabel.CONFLICT, conf=0.85)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label != LLMLabel.CONFLICT, (
            f"LLM-hallucinated CONFLICT on off-topic PZ admin survived "
            f"verifier (post-fix F); actions={out.verifier_actions}"
        )
        # Verify the suppression action was recorded (for audit trail).
        suppressed = any(
            "suppress_negation" in a
            for a in (out.verifier_actions or [])
        )
        assert suppressed, (
            f"Expected a suppress_negation_* action; got: {out.verifier_actions}"
        )

    def test_0_17_sent3_llm_conflict_on_offtopic_pmi_no_conflict(self):
        """PR-K post-fix F: 0.17::sent3 PMI variant — LLM hallucinated
        CONFLICT on an off-topic access-control testing unit (the
        semantically relevant unit was at rank 1 but the judged unit
        is rank 5). Same fix path — no text-level topical link."""
        req = _req(
            "Система не должна аварийно завершать свою работу в случае "
            "возникновения ошибки.",
            req_type=RequirementType.RELIABILITY,
        )
        unit = _unit(
            "Тест-кейс 12: Проверка контроля доступа. "
            "Попытка открыть защищённый раздел без авторизации. "
            "Ожидаемый результат: пользователь перенаправляется на страницу входа.",
        )
        j = _judgment(LLMLabel.CONFLICT, conf=0.80)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label != LLMLabel.CONFLICT, (
            f"LLM-hallucinated CONFLICT on off-topic PMI access-control "
            f"unit survived verifier (post-fix F); actions={out.verifier_actions}"
        )

    def test_0_17_sent3_avariyno_vs_prodolzhat_same_outcome(self):
        """0.17::sent3 PMI — TZ "не должна аварийно завершать" vs PMI
        "должна продолжать корректно функционировать". Same-outcome
        compatibility table (existing) MUST catch this and demote to
        PARTIAL, NOT raise CONFLICT."""
        req = _req(
            "Система не должна аварийно завершать свою работу в случае "
            "возникновения ошибки.",
            req_type=RequirementType.RELIABILITY,
        )
        unit = _unit(
            "В случае ошибки при обработке запроса система должна "
            "продолжать корректно функционировать."
        )
        j = _judgment(LLMLabel.PARTIAL, conf=0.85)
        out = self.verifier.verify(j, req, unit)
        # Either same-outcome catches it (PARTIAL via demote) OR the
        # judge-strongly-positive guard catches it (label preserved).
        # Both outcomes are NOT CONFLICT.
        assert out.rule_adjusted_label != LLMLabel.CONFLICT, (
            f"same-outcome 'avariyno'/'prodolzhat' produced CONFLICT; "
            f"actions={out.verifier_actions}"
        )

    def test_0_20_sent1_same_upper_bound_phrasing_no_conflict(self):
        """0.20::sent1 — TZ "Время восстановления... не должно превышать
        общее время... на перезагрузку" vs PMI "время восстановления
        не должно превышать времени, требующегося на перезагрузку
        операционной системы и запуск программы". Same prohibition
        phrasing on BOTH sides; numeric values not specified — these
        are equivalent semantic statements, NOT a contradiction."""
        req = _req(
            "Время восстановления после отказа работы системы не должно "
            "превышать общее время, необходимое на перезагрузку "
            "составляющих системы.",
            req_type=RequirementType.RELIABILITY,
        )
        unit = _unit(
            "При отказе, который произошел по вине каких-либо внешних "
            "факторов и не является непоправимым, время восстановления "
            "не должно превышать времени, требующегося на перезагрузку "
            "операционной системы и запуск программы."
        )
        j = _judgment(LLMLabel.COVERED, conf=1.0)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label != LLMLabel.CONFLICT, (
            f"same upper-bound phrasing on both sides produced CONFLICT; "
            f"actions={out.verifier_actions}"
        )


class TestSameUpperBoundCompatibilityRule:
    """Direct test of the new same_upper_bound check inside
    _same_outcome_negation_compatible."""

    @pytest.mark.parametrize("req_text, unit_text, expected", [
        # Same upper-bound phrasing on both sides — compatible.
        (
            "Время восстановления не должно превышать общее время на перезагрузку.",
            "Время восстановления не должно превышать времени, необходимого для устранения неисправностей.",
            True,
        ),
        (
            "Время отклика не должно превышать 2 секунд.",
            "Время отклика не должно превышать 5 секунд.",
            True,  # both are upper-bound prohibitions; numeric mismatch caught by numeric rule
        ),
        # ALL four genders.
        ("X не должен превышать Y.", "X не должен превышать Z.", True),
        ("X не должна превышать Y.", "X не должна превышать Z.", True),
        ("X не должно превышать Y.", "X не должно превышать Z.", True),
        ("X не должны превышать Y.", "X не должны превышать Z.", True),
        # Asymmetric: only one side has prohibition — falls back to old rules.
        (
            "Время отклика не должно превышать 3 секунд.",
            "Система использует Windows.",
            False,
        ),
        # Different verb after prohibition — must NOT trigger.
        (
            "Система не должна сохранять пароль.",
            "Система не должна обрабатывать запросы.",
            False,
        ),
    ])
    def test_same_upper_bound(self, req_text, unit_text, expected):
        actual = _same_outcome_negation_compatible(req_text, unit_text)
        assert actual is expected, (
            f"req={req_text!r} unit={unit_text!r}: "
            f"expected={expected}, got={actual}"
        )


class TestNegationGuardsDoNotBreakRealConflict:
    """Sanity: the new guards must NOT mask GENUINE prohibition
    contradictions. Real CONFLICT cases must still produce CONFLICT."""

    def setup_method(self):
        self.verifier = PairVerifier()

    def test_genuine_prohibition_vs_affirmation_still_conflict(self):
        """Both sides talk about THE SAME thing (passwords + storage),
        modality is opposite. Real CONFLICT, must survive new guards."""
        req = _req(
            "Программа не должна сохранять пароль пользователя в логах.",
            req_type=RequirementType.SECURITY,
        )
        unit = _unit(
            "Программа сохраняет пароль пользователя в системном логе "
            "для целей аудита.",
        )
        # LLM uncertain (PARTIAL low conf) — verifier should override.
        j = _judgment(LLMLabel.PARTIAL, conf=0.45)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.CONFLICT, (
            f"genuine password-storage CONFLICT was suppressed; "
            f"actions={out.verifier_actions}"
        )
        assert "conflict_confirmed_negation" in out.verifier_actions

    def test_genuine_conflict_with_strong_topic_link_high_judge_conf(self):
        """Edge case: judge says CONFLICT conf=0.85, with prohibition
        mismatch and clear topical link. Must be confirmed CONFLICT —
        the judge-strongly-positive guard targets COVERED/PARTIAL only."""
        req = _req(
            "Система не должна передавать персональные данные третьим лицам.",
            req_type=RequirementType.SECURITY,
        )
        unit = _unit(
            "Система передаёт персональные данные пользователей "
            "партнёрской аналитической платформе для обработки.",
        )
        j = _judgment(LLMLabel.CONFLICT, conf=0.85)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.CONFLICT
        assert "conflict_confirmed_negation" in out.verifier_actions
