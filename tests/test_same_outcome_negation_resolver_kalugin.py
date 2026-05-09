"""
P0 #7 — semantic same-outcome negation resolver.

Калугин-class symptom: Verifier flagged CONFLICT on pairs whose surface
phrasings differ in polarity but describe the SAME outcome:
  TZ:  «программа не должна аварийно завершаться»
  PMI: «программа должна обрабатывать ошибки и продолжать работу»

Old behaviour: hard-coded _SAME_OUTCOME_PAIRS table missed the variant,
the negation rule fired, PARTIAL was upgraded to CONFLICT. This test
file pins three contract changes:

  1. Pattern-based stem table covers «не должн[аоы] X» ↔ «должн[аоы]
     антоним(X)» across (аварийн|с ошибк|некорректн|нестабильн|со сбо|
     прерыван|потерять данные|блокировать) without surface-string
     hard-coding.

  2. Optional `same_outcome_sim_fn` on PairVerifier: when injected and
     returning ≥ 0.65 cosine similarity, the negation rule is suppressed
     before it can fire. Default unset — pure-pattern fallback for
     offline pipelines and unit tests.

  3. Hard rule: verifier may CONFIRM but never UPGRADE. Five+ regression
     pairs «не должен X / должен антоним(X)» starting from LLM PARTIAL
     are pinned to keep PARTIAL.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.verify_pairs import (
    PairVerifier,
    _same_outcome_negation_compatible,
)
from app.domain.c_quality_enums import LLMLabel, Modality
from app.domain.c_quality_models import (
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
)


def _req(text: str) -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="doc-tz",
        text=text,
        normalized_text=text.lower(),
        modality=Modality.MUST_NOT if "не должн" in text.lower() else Modality.UNKNOWN,
    )


def _unit(text: str) -> CoverageUnit:
    return CoverageUnit(
        unit_id="u1",
        target_document_id="doc-pmi",
        target_doc_role="pmi",
        text=text,
        normalized_text=text.lower(),
    )


def _judgment(label: LLMLabel, conf: float = 0.55) -> PairJudgment:
    return PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=label, rule_adjusted_label=label, llm_confidence=conf,
    )


# ── (1) Pattern-based same-outcome table ───────────────────────────────


_SAME_OUTCOME_PAIRS = [
    # crash / continue working — original Калугин symptom
    ("Программа не должна аварийно завершаться.",
     "Программа должна обрабатывать ошибки и продолжать работу."),
    # error termination / correct functioning
    ("Сервис не должен завершаться с ошибкой при пиковой нагрузке.",
     "Сервис должен корректно функционировать при пиковой нагрузке."),
    # incorrect operation / correct operation
    ("Модуль не должен работать некорректно при обновлении конфигурации.",
     "Модуль должен корректно работать при обновлении конфигурации."),
    # unstable / stable
    ("Подсистема не должна работать нестабильно под нагрузкой.",
     "Подсистема должна стабильно функционировать под нагрузкой."),
    # operate with errors / operate without errors
    ("Программа не должна работать со сбоями в течение суток.",
     "Программа должна работать без сбоев в течение суток."),
    # crash / keep running
    ("Сервер не должен падать при одновременных запросах.",
     "Сервер должен продолжать работать при одновременных запросах."),
    # data loss / data preservation
    ("Система не должна терять данные при отключении питания.",
     "Система должна сохранять данные при отключении питания."),
]


@pytest.mark.parametrize("req_text, unit_text", _SAME_OUTCOME_PAIRS)
def test_same_outcome_pattern_recognised_kalugin(req_text: str, unit_text: str) -> None:
    # The pattern table recognises the pair in either ordering
    # (req-prohibits + unit-affirms, and the reverse).
    assert _same_outcome_negation_compatible(req_text, unit_text), (
        f"pattern table failed to recognise same-outcome pair:\n  req: {req_text}\n  unit: {unit_text}"
    )
    assert _same_outcome_negation_compatible(unit_text, req_text), (
        f"pattern table failed in reverse ordering:\n  unit: {unit_text}\n  req: {req_text}"
    )


# ── (2) Embedding similarity gate ──────────────────────────────────────


def test_embedding_similarity_above_threshold_suppresses_negation_kalugin() -> None:
    # When the injected similarity backend returns ≥ 0.65, the verifier
    # treats the pair as same-outcome and suppresses the negation rule.
    # The phrasing is intentionally OFF the pattern table so we know
    # the suppression came from the embedding, not the regex fallback.
    sim_fn = lambda a, b: 0.81
    verifier = PairVerifier(same_outcome_sim_fn=sim_fn)

    req = _req("Программа не должна обрывать сессию пользователя при таймауте.")
    unit = _unit("Программа должна аккуратно завершать пользовательскую сессию по таймауту.")
    out = verifier.verify(_judgment(LLMLabel.PARTIAL), req, unit)

    assert out.rule_adjusted_label == LLMLabel.PARTIAL
    assert "suppress_negation_embedding_same_outcome" in out.verifier_actions


def test_embedding_similarity_below_threshold_falls_through_kalugin() -> None:
    # Below threshold: the embedder gate yields nothing; verifier falls
    # through to the existing pattern-table path. Off-pattern, off-topic
    # pairs simply land in the no-topic-link suppression branch (no
    # CONFLICT promotion either way).
    sim_fn = lambda a, b: 0.30
    verifier = PairVerifier(same_outcome_sim_fn=sim_fn)
    req = _req("Программа не должна обрывать сессию пользователя при таймауте.")
    unit = _unit("Программа должна аккуратно завершать пользовательскую сессию по таймауту.")
    out = verifier.verify(_judgment(LLMLabel.PARTIAL), req, unit)
    assert "suppress_negation_embedding_same_outcome" not in out.verifier_actions


def test_embedding_failure_does_not_break_verifier_kalugin() -> None:
    # If the injected fn raises, verifier must not crash — falls back
    # to the pattern table.
    def boom(a: str, b: str) -> float:
        raise RuntimeError("embedder offline")

    verifier = PairVerifier(same_outcome_sim_fn=boom)
    req = _req("Программа не должна аварийно завершаться.")
    unit = _unit("Программа должна обрабатывать ошибки и продолжать работу.")
    out = verifier.verify(_judgment(LLMLabel.CONFLICT, conf=0.7), req, unit)
    # The pattern table catches this pair (LLM-CONFLICT pre-check) and
    # upgrades to COVERED with the existing same-outcome path.
    assert out.rule_adjusted_label == LLMLabel.COVERED


# ── (3) Hard rule: confirm-only, never upgrade ─────────────────────────


@pytest.mark.parametrize("req_text, unit_text", _SAME_OUTCOME_PAIRS)
def test_same_outcome_pair_keeps_partial_no_upgrade_kalugin(req_text: str, unit_text: str) -> None:
    # Five+ regression pairs (parametrised over the table above): when
    # LLM returned PARTIAL on a same-outcome pair, the verifier must
    # NOT upgrade to CONFLICT. Either the pattern table suppresses
    # negation outright (preserve PARTIAL) or the no-upgrade rule
    # demotes a would-be CONFLICT back to PARTIAL.
    verifier = PairVerifier()
    req = _req(req_text)
    unit = _unit(unit_text)
    out = verifier.verify(_judgment(LLMLabel.PARTIAL), req, unit)
    assert out.rule_adjusted_label == LLMLabel.PARTIAL, (
        f"verifier upgraded PARTIAL → {out.rule_adjusted_label} on same-outcome pair:\n"
        f"  req:  {req_text}\n  unit: {unit_text}\n  actions: {out.verifier_actions}"
    )
    # Critical: never tagged as a confirmed CONFLICT.
    assert "conflict_confirmed_negation" not in out.verifier_actions
