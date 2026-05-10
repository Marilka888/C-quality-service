"""
Polyakov-regression Step 7 (2026-05-11): aspect-mismatch guards in
PairVerifier.

The LLM judge frequently raises PARTIAL/COVERED on pairs sharing
only generic vocabulary («система», «интерфейс», «ошибка») when the
actual semantic topics don't match. Real-package examples from the
May-11 Polyakov run:

  * 0.14 «время отклика 3 сек» (response_time) vs PMI evidence
    «Процессор Intel i5» / «время восстановления» → judge said
    PARTIAL conf 0.6, should be MISSING.
  * 0.18::sent2 «устойчивость к атакам типа Внедрение кода»
    (code_injection) vs PMI «разграничение доступа по ролям» →
    PARTIAL conf 0.7, should be MISSING.
  * 0.15::sent2 «макет в Figma» (figma_design) vs PMI «интерактивный
    интерфейс в браузере» (browser_ui) → PARTIAL conf 0.7, should
    be MISSING.
  * 0.17::sent4 «обработка данных с сервера» (data_from_server) vs
    «обработка неверных запросов» (invalid_request_handling) →
    COVERED conf 0.95, should be PARTIAL or MISSING.

Strategy:
  * Classify each side into a small set of fine-grained topics
    (heuristic regex). Multi-topic per text is allowed.
  * Curated mismatch table `_TOPIC_MISMATCH_PAIRS` lists the
    (req_topic, unit_topic) directions that genuinely don't match.
  * Demote PARTIAL/COVERED → IRRELEVANT when (a) at least one
    mismatch pair fires AND (b) req+unit share NO topics — pure
    off-topic, not multi-aspect coverage.
  * Last gate before the verifier's no-op exit, so all preserve-
    paths (entity-rich, tech-stack, shared-substantive) run first.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.verify_pairs import (
    PairVerifier,
    _classify_topics,
    _topic_mismatch_reason,
)
from app.domain.c_quality_enums import (
    CoverageUnitType,
    LLMLabel,
    RequirementType,
)
from app.domain.c_quality_models import (
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
)


# ── Topic classifier ───────────────────────────────────────────────


@pytest.mark.parametrize("text, expected_topic", [
    # Polyakov 0.14 — response_time.
    ("Время отклика приложения при условии отсутствия сетевого "
     "взаимодействия не должно превышать 3 секунд.", "response_time"),
    # Polyakov 0.20 — recovery_time.
    ("Время восстановления после отказа работы системы не должно "
     "превышать общее время на перезагрузку составляющих системы.",
     "recovery_time"),
    # Hardware-spec evidence.
    ("Процессор Intel(R) Core(TM) i5-7500 (4 ядра, 3.4 ГГц).",
     "hardware_specs"),
    ("Более 900 ГБ доступного дискового пространства.", "hardware_specs"),
    # Polyakov 0.18::sent2 — code_injection.
    ("Система должна быть устойчива к атакам типа «Внедрение кода».",
     "code_injection"),
    # Access-control evidence.
    ("Для проверки корректности работы разграничения доступа "
     "используются пользователи с различными ролями.",
     "access_control"),
    ("Система должна обеспечивать возможность настройки доступа к "
     "объектам на основе ролей пользователей.", "access_control"),
    # Polyakov 0.15::sent2 — figma_design.
    ("Макет интерфейса должен быть разработан в Figma.", "figma_design"),
    # Polyakov PMI 0.15::sent2 evidence — browser_ui.
    ("Программный интерфейс должен быть представлен в виде "
     "интерактивного пользовательского интерфейса, запускаемого в "
     "браузере.", "browser_ui"),
    # Polyakov 0.17::sent4 — data_from_server.
    ("Также система должна корректно обрабатывать данные, полученные "
     "с сервера.", "data_from_server"),
    # Polyakov PMI evidence — invalid_request_handling.
    ("Система должна корректно обрабатывать неверные запросы любого "
     "вида и выдавать информативные сообщения об ошибках.",
     "invalid_request_handling"),
])
def test_topic_classifier_recognises_polyakov_patterns(
    text: str, expected_topic: str,
) -> None:
    topics = _classify_topics(text)
    assert expected_topic in topics, (
        f"text {text!r} → topics {topics!r}; expected {expected_topic!r}"
    )


def test_topic_classifier_returns_empty_for_unrelated_text() -> None:
    """Generic text without any of the curated topics returns empty
    set — safe default (no mismatch can fire on unclassified text)."""
    topics = _classify_topics(
        "Программа должна предоставлять интерфейс для работы с проектами."
    )
    assert topics == set()


# ── Mismatch table ─────────────────────────────────────────────────


@pytest.mark.parametrize("req_topics, unit_topics, should_mismatch", [
    # Curated mismatch pairs all fire.
    ({"response_time"}, {"recovery_time"}, True),
    ({"response_time"}, {"hardware_specs"}, True),
    ({"code_injection"}, {"access_control"}, True),
    ({"figma_design"}, {"browser_ui"}, True),
    ({"data_from_server"}, {"invalid_request_handling"}, True),
    # Same topic on both sides — never a mismatch.
    ({"response_time"}, {"response_time"}, False),
    # Overlap (multi-topic) — never a mismatch even with one bad pair.
    ({"response_time", "recovery_time"}, {"recovery_time"}, False),
    # Reverse direction not in table — must not fire.
    ({"recovery_time"}, {"response_time"}, False),
    # Unrelated topics — no mismatch (not in table).
    ({"figma_design"}, {"hardware_specs"}, False),
    # Empty → no mismatch (safe default).
    (set(), {"response_time"}, False),
    ({"response_time"}, set(), False),
])
def test_topic_mismatch_table(req_topics, unit_topics, should_mismatch) -> None:
    reason = _topic_mismatch_reason(req_topics, unit_topics)
    if should_mismatch:
        assert reason is not None, (
            f"expected mismatch for req={req_topics} unit={unit_topics}"
        )
    else:
        assert reason is None, (
            f"expected NO mismatch for req={req_topics} unit={unit_topics}; "
            f"got reason={reason!r}"
        )


# ── End-to-end: PairVerifier demotes mismatched PARTIAL/COVERED ────


def _req(text: str, req_type: RequirementType = RequirementType.OTHER) -> RequirementUnit:
    return RequirementUnit(
        req_id="r1", source_document_id="tz",
        text=text, normalized_text=text.lower(),
        requirement_type=req_type,
    )


def _unit(text: str) -> CoverageUnit:
    return CoverageUnit(
        unit_id="u1", target_document_id="doc-pmi", target_doc_role="pmi",
        unit_type=CoverageUnitType.PARAGRAPH,
        text=text, normalized_text=text.lower(),
    )


def _partial_judgment(conf: float = 0.7) -> PairJudgment:
    return PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=LLMLabel.PARTIAL,
        rule_adjusted_label=LLMLabel.PARTIAL,
        llm_confidence=conf,
        cited_phrases=["test"],
        explanation="Test PARTIAL.",
    )


def _covered_judgment(conf: float = 0.95) -> PairJudgment:
    return PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=LLMLabel.COVERED,
        rule_adjusted_label=LLMLabel.COVERED,
        llm_confidence=conf,
        cited_phrases=["test"],
        explanation="Test COVERED.",
    )


def test_polyakov_0_14_response_time_vs_recovery_demoted() -> None:
    """Polyakov 0.14: req «время отклика 3 сек» vs unit «время
    восстановления при отказе» — different topics, must demote."""
    req = _req(
        "Время отклика приложения при условии отсутствия сетевого "
        "взаимодействия не должно превышать 3 секунд.",
        RequirementType.PERFORMANCE,
    )
    unit = _unit(
        "При отказе, который произошел по вине каких-либо внешних "
        "факторов, время восстановления не должно превышать времени, "
        "требующегося на перезагрузку операционной системы."
    )
    out = PairVerifier().verify(_partial_judgment(conf=0.7), req, unit)
    assert out.rule_adjusted_label == LLMLabel.IRRELEVANT, (
        f"response_time vs recovery_time must demote PARTIAL → IRRELEVANT; "
        f"got {out.rule_adjusted_label}; explanation={out.explanation!r}"
    )


def test_polyakov_0_14_response_time_vs_hardware_demoted() -> None:
    """Polyakov 0.14 alt evidence: hardware specs (Intel processor)
    vs response time req — different topics."""
    req = _req(
        "Время отклика приложения не должно превышать 3 секунд.",
        RequirementType.PERFORMANCE,
    )
    unit = _unit("Процессор Intel(R) Core(TM) i5-7500 (4 ядра, 3.4 ГГц).")
    out = PairVerifier().verify(_partial_judgment(conf=0.6), req, unit)
    assert out.rule_adjusted_label == LLMLabel.IRRELEVANT


def test_polyakov_0_18_sent2_injection_vs_rbac_demoted() -> None:
    """Polyakov 0.18::sent2: req «атаки типа Внедрение кода» vs
    evidence «настройка доступа на основе ролей» — security but
    different aspect."""
    req = _req(
        "Система должна быть устойчива к атакам типа «Внедрение кода».",
        RequirementType.SECURITY,
    )
    unit = _unit(
        "Система должна обеспечивать возможность настройки доступа "
        "к объектам на основе ролей пользователей и их принадлежности "
        "к проектам, включая доступ к метаданным, файлам и операциям "
        "изменения данных."
    )
    out = PairVerifier().verify(_partial_judgment(conf=0.7), req, unit)
    assert out.rule_adjusted_label == LLMLabel.IRRELEVANT


def test_polyakov_0_15_sent2_figma_vs_browser_demoted() -> None:
    """Polyakov 0.15::sent2: req «макет в Figma» vs evidence
    «интерактивный интерфейс в браузере» — different artefact."""
    req = _req(
        "Макет интерфейса должен быть разработан в Figma.",
        RequirementType.INTERFACE,
    )
    unit = _unit(
        "Программный интерфейс должен быть представлен в виде "
        "интерактивного пользовательского интерфейса, запускаемого "
        "в браузере."
    )
    out = PairVerifier().verify(_partial_judgment(conf=0.7), req, unit)
    assert out.rule_adjusted_label == LLMLabel.IRRELEVANT


def test_polyakov_0_17_sent4_server_data_vs_invalid_request_demoted() -> None:
    """Polyakov 0.17::sent4: req «данные с сервера» vs evidence
    «обработка неверных запросов» — different I/O direction. Must
    demote even from COVERED conf 0.95."""
    req = _req(
        "Также система должна корректно обрабатывать данные, "
        "полученные с сервера.",
        RequirementType.RELIABILITY,
    )
    unit = _unit(
        "Система должна корректно обрабатывать неверные запросы "
        "любого вида и выдавать информативные сообщения об ошибках."
    )
    out = PairVerifier().verify(_covered_judgment(conf=0.95), req, unit)
    assert out.rule_adjusted_label == LLMLabel.IRRELEVANT, (
        f"data_from_server vs invalid_request must demote even COVERED; "
        f"got {out.rule_adjusted_label}; explanation={out.explanation!r}"
    )


# ── Sanity: same-topic and overlap don't trigger ────────────────────


def test_same_topic_response_time_unaffected() -> None:
    """Sanity: when both sides are response_time, no demote — that's
    the happy case."""
    req = _req(
        "Время отклика приложения не должно превышать 3 секунд.",
        RequirementType.PERFORMANCE,
    )
    unit = _unit(
        "Тест измерения времени отклика приложения проводится при "
        "локальном запуске."
    )
    out = PairVerifier().verify(_partial_judgment(conf=0.7), req, unit)
    assert out.rule_adjusted_label == LLMLabel.PARTIAL


def test_unclassified_pair_unaffected() -> None:
    """Sanity: pair where neither side hits any topic — no demote."""
    req = _req("Программа должна предоставлять интерфейс для проектов.")
    unit = _unit("Реализован интерфейс для работы с проектами.")
    out = PairVerifier().verify(_partial_judgment(conf=0.7), req, unit)
    assert out.rule_adjusted_label == LLMLabel.PARTIAL


def test_irrelevant_judgment_not_processed() -> None:
    """Sanity: IRRELEVANT verdicts skip the mismatch guard entirely
    (only PARTIAL/COVERED trigger it). Verifies the early-exit."""
    req = _req(
        "Время отклика приложения не должно превышать 3 секунд.",
        RequirementType.PERFORMANCE,
    )
    unit = _unit("Процессор Intel(R) Core(TM) i5-7500.")
    j = PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=LLMLabel.IRRELEVANT,
        rule_adjusted_label=LLMLabel.IRRELEVANT,
        llm_confidence=0.5,
    )
    out = PairVerifier().verify(j, req, unit)
    # IRRELEVANT stays IRRELEVANT (no demote needed, no upgrade either).
    assert out.rule_adjusted_label == LLMLabel.IRRELEVANT
