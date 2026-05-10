"""
Polyakov-regression Step 6+9 (2026-05-11): the C-quality aggregator
let COVERED rows ship with non-empty uncoveredAspects — internally
inconsistent ("fully covered" + "here's what's missing"). All five
PMI COVEREDs in the May-11 Polyakov report had this pattern:

  * 0.11::sent1 «Регистрация, авторизация и аутентификация» COVERED
    with uncovered=[«регистрация»,«аутентификация»,«авторизация»,
    «регISTRATION»] (the «регISTRATION» is latin-mix LLM noise).
  * 0.15::sent1 «понятный интерфейс…» COVERED with uncovered=
    [«понятный интерфейс»,«спокойные тона»,«заранее разработанный
    макет»,«фирменный дизайн ВШЭ»].
  * 0.17::sent2/sent4 reliability rows COVERED with uncovered tag-
    noise («specific_object_match», «verb_match», …) leaked from
    the verifier.
  * 0.20::sent1 «время восстановления» COVERED with substring chains
    («время восстановления» / «общее время на перезагрузку
    составляющих системы» / «перезагрузка составляющих системы»).

Two coordinated fixes:

Step 9 — `_normalize_uncovered_aspects`:
  * Drop empty / pure-noise tags via `_ASPECT_NOISE_TAG_RE` (catches
    verifier-internal labels and latin-mix artefacts);
  * Strip leading/trailing punctuation;
  * Drop strict substrings (when phrase A is a substring of phrase B,
    keep only B — strictly more informative);
  * Preserve insertion order on the kept set.

Step 6 — post-aggregation guard:
  * If `chosen_status == COVERED` and normalized aspects non-empty,
    downgrade to PARTIAL with subcode
    `PARTIAL_DOWNGRADED_FROM_COVERED` and an explicit
    `aggregation_reason` extension noting the downgrade.
"""
from __future__ import annotations

from app.application.use_cases.aggregate_coverage import (
    CoverageAggregator,
    SUBCODE_COVERED,
    SUBCODE_PARTIAL,
    SUBCODE_PARTIAL_DOWNGRADED_FROM_COVERED,
    _normalize_uncovered_aspects,
)
from app.domain.c_quality_enums import (
    CoverageStatus,
    CoverageUnitType,
    LLMLabel,
    RequirementType,
)
from app.domain.c_quality_models import (
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
    RetrievedCandidate,
)


# ── Step 9: aspect normalization ────────────────────────────────────


def test_normalize_drops_empty_and_short() -> None:
    out = _normalize_uncovered_aspects(["", "  ", "ab", "real aspect"])
    assert out == ["real aspect"]


def test_normalize_drops_verifier_noise_tags() -> None:
    """Polyakov 0.17::sent2/sent4 leaked verifier-internal tag labels
    into uncovered_aspects. They must be dropped — the reviewer sees
    only honest domain phrases."""
    raws = [
        "specific_object_match",
        "verb_object_coverage",
        "sufficient_lexical_density",
        "verb_match",
        "object_phrase_match",
        # And one real domain phrase — must survive.
        "корректная обработка данных с сервера",
    ]
    out = _normalize_uncovered_aspects(raws)
    assert out == ["корректная обработка данных с сервера"]


def test_normalize_drops_latin_mix_noise() -> None:
    """Polyakov 0.11::sent1 had «регISTRATION» — LLM started typing
    the cyrillic word in english half-way through. Must be filtered."""
    raws = ["регистрация", "авторизация", "регISTRATION"]
    out = _normalize_uncovered_aspects(raws)
    # «регISTRATION» dropped; the two real cyrillic terms preserved.
    assert "регистрация" in out
    assert "авторизация" in out
    assert "регISTRATION" not in out


def test_normalize_drops_strict_substring_chains() -> None:
    """Polyakov 0.20::sent1 / 0.10::sent1 had progressive substrings.
    Keep only the most-informative (longest) variant."""
    raws = [
        "загрузка",
        "загрузка файлов",
        "загрузка файлов в и из системы",
    ]
    out = _normalize_uncovered_aspects(raws)
    assert out == ["загрузка файлов в и из системы"]


def test_normalize_preserves_distinct_aspects() -> None:
    """Sanity: distinct aspects (no substring relation) all survive,
    in original insertion order."""
    raws = [
        "регистрация пользователей",
        "авторизация по паролю",
        "аутентификация по токену",
    ]
    out = _normalize_uncovered_aspects(raws)
    assert out == [
        "регистрация пользователей",
        "авторизация по паролю",
        "аутентификация по токену",
    ]


def test_normalize_dedups_case_insensitive() -> None:
    """Same phrase in different case = one aspect."""
    out = _normalize_uncovered_aspects(["Понятный интерфейс", "понятный интерфейс"])
    assert out == ["Понятный интерфейс"]


def test_normalize_strips_trailing_punctuation() -> None:
    out = _normalize_uncovered_aspects(["загрузка файлов.", "  поиск,  "])
    assert "загрузка файлов" in out
    assert "поиск" in out


# ── Step 6: COVERED downgrade ───────────────────────────────────────


def _req() -> RequirementUnit:
    return RequirementUnit(
        req_id="r1",
        source_document_id="tz",
        text="Регистрация, авторизация и аутентификация.",
        normalized_text="регистрация, авторизация и аутентификация.",
        requirement_type=RequirementType.SECURITY,
    )


def _unit() -> CoverageUnit:
    return CoverageUnit(
        unit_id="u1",
        target_document_id="doc-pmi",
        target_doc_role="pmi",
        unit_type=CoverageUnitType.PARAGRAPH,
        text="Система должна обеспечивать возможность регистрации и авторизации пользователей.",
        normalized_text=(
            "система должна обеспечивать возможность регистрации и авторизации пользователей."
        ),
    )


def _covered_judgment_with_uncovered(missing: list[str]) -> PairJudgment:
    """LLM judge said COVERED but emitted uncovered_aspects — the
    pathology Step 6 is designed to catch."""
    return PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=LLMLabel.COVERED,
        rule_adjusted_label=LLMLabel.COVERED,
        llm_confidence=0.95,
        cited_phrases=["регистрации и авторизации"],
        missing_aspects=missing,
        explanation="Фрагмент полностью покрывает требование.",
    )


def test_polyakov_0_11_sent1_covered_with_uncovered_downgrades() -> None:
    """Exact Polyakov 0.11::sent1 reproduction: judge says COVERED,
    uncovered_aspects = [«регистрация», «аутентификация»,
    «авторизация», «регISTRATION»]. Step 6 must downgrade to PARTIAL
    with explicit subcode."""
    req = _req()
    unit = _unit()
    judgment = _covered_judgment_with_uncovered(
        ["регистрация", "аутентификация", "авторизация", "регISTRATION"]
    )
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=[judgment],
        candidates_by_unit_id={"u1": RetrievedCandidate(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            retrieval_score=0.55,
        )},
        units_by_id={"u1": unit},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.status == CoverageStatus.PARTIAL, (
        f"COVERED with non-empty uncovered_aspects must downgrade to "
        f"PARTIAL; got {result.status}"
    )
    assert result.status_subcode == SUBCODE_PARTIAL_DOWNGRADED_FROM_COVERED
    # Latin-mix noise should also be filtered out by Step 9.
    assert "регISTRATION" not in result.uncovered_aspects
    # Reason should mention the downgrade.
    assert "Downgrade" in (result.aggregation_reason or "") or \
           "downgrade" in (result.aggregation_reason or "").lower()


def test_polyakov_0_15_sent1_ui_covered_downgrades() -> None:
    """Reproduce 0.15::sent1: judge says COVERED on a UI requirement,
    but uncovered = «понятный интерфейс», «спокойные тона»,
    «заранее разработанный макет». Must downgrade."""
    req = RequirementUnit(
        req_id="r1", source_document_id="tz",
        text="Приложение должно обладать понятным интерфейсом, реализованном "
             "в спокойных тонах и фирменном дизайном ВШЭ на основе заранее "
             "разработанного макета.",
        normalized_text="приложение должно обладать понятным интерфейсом",
        requirement_type=RequirementType.INTERFACE,
    )
    unit = CoverageUnit(
        unit_id="u1", target_document_id="doc-pmi", target_doc_role="pmi",
        unit_type=CoverageUnitType.PARAGRAPH,
        text="UI должен быть выполнен в фирменном стиле ВШЭ.",
        normalized_text="ui должен быть выполнен в фирменном стиле вшэ.",
    )
    judgment = PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=LLMLabel.COVERED,
        rule_adjusted_label=LLMLabel.COVERED,
        llm_confidence=0.9,
        cited_phrases=["фирменном стиле ВШЭ"],
        missing_aspects=[
            "понятный интерфейс",
            "спокойные тона",
            "заранее разработанный макет",
        ],
        explanation="Полностью покрывает аспекты ВШЭ.",
    )
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=[judgment],
        candidates_by_unit_id={"u1": RetrievedCandidate(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            retrieval_score=0.58,
        )},
        units_by_id={"u1": unit},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.status == CoverageStatus.PARTIAL
    assert result.status_subcode == SUBCODE_PARTIAL_DOWNGRADED_FROM_COVERED
    # All three real aspects survive normalization (no substring chain).
    assert len(result.uncovered_aspects) == 3


def test_covered_with_only_noise_aspects_stays_covered() -> None:
    """If the judge's uncovered_aspects contains ONLY verifier-noise
    tags (`specific_object_match`, `verb_match`, …), they get filtered
    by Step 9 → effective list is empty → no downgrade. The row
    correctly stays COVERED."""
    req = _req()
    unit = _unit()
    judgment = _covered_judgment_with_uncovered(
        ["specific_object_match", "verb_match", "sufficient_lexical_density"]
    )
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=[judgment],
        candidates_by_unit_id={"u1": RetrievedCandidate(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            retrieval_score=0.55,
        )},
        units_by_id={"u1": unit},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.status == CoverageStatus.COVERED, (
        f"COVERED with only noise-tag aspects must stay COVERED after "
        f"Step 9 normalization; got {result.status}, "
        f"aspects={result.uncovered_aspects}"
    )
    assert result.uncovered_aspects == []
    assert result.status_subcode == SUBCODE_COVERED


def test_clean_covered_unaffected() -> None:
    """Sanity: COVERED with truly empty uncovered_aspects stays
    COVERED. Step 6 only fires when the inconsistency exists."""
    req = _req()
    unit = _unit()
    judgment = _covered_judgment_with_uncovered([])
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=[judgment],
        candidates_by_unit_id={"u1": RetrievedCandidate(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            retrieval_score=0.55,
        )},
        units_by_id={"u1": unit},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.status == CoverageStatus.COVERED
    assert result.uncovered_aspects == []


def test_partial_with_aspects_unaffected_by_step6() -> None:
    """Sanity: PARTIAL with uncovered_aspects (the happy case Step 6
    is named after) stays PARTIAL — no double-downgrade."""
    req = _req()
    unit = _unit()
    judgment = PairJudgment(
        req_id="r1", unit_id="u1", target_document_id="doc-pmi",
        llm_label=LLMLabel.PARTIAL,
        rule_adjusted_label=LLMLabel.PARTIAL,
        llm_confidence=0.7,
        cited_phrases=["регистрации"],
        missing_aspects=["аутентификация"],
        explanation="Authentication is missing.",
    )
    result = CoverageAggregator().aggregate(
        requirement=req,
        judgments=[judgment],
        candidates_by_unit_id={"u1": RetrievedCandidate(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            retrieval_score=0.55,
        )},
        units_by_id={"u1": unit},
        target_document_id="doc-pmi",
        target_doc_role="PMI",
    )
    assert result.status == CoverageStatus.PARTIAL
    assert result.status_subcode == SUBCODE_PARTIAL  # not the downgrade subcode
    assert result.uncovered_aspects == ["аутентификация"]
