"""
Entity extractor tests — both the spaCy path and the regex fallback.

The fallback path is what runs in CI by default (CQUALITY_DISABLE_SPACY=1
in the test env), so its contract MUST be preserved: title-case nouns
and acronyms still extracted, lowercase domain phrases still missed
(by design — that's what spaCy is for).

The spaCy path is gated behind a `pytest.importorskip` so the suite
stays green on machines without ru_core_news_md installed.
"""
from __future__ import annotations

import os

import pytest


# ── regex fallback path (always runs) ────────────────────────────────────


class TestRegexFallback:
    """Behaviour when spaCy is unavailable / disabled. This is the
    legacy behaviour and MUST be preserved for backwards compatibility
    with existing test expectations in test_coverage_pipeline.py."""

    def setup_method(self):
        # Force fallback regardless of whether spaCy is installed.
        os.environ["CQUALITY_DISABLE_SPACY"] = "1"
        # Reset cached state so the env var takes effect this test.
        import app.application.use_cases.build_requirements as br
        br._SPACY_NLP = None

    def teardown_method(self):
        os.environ.pop("CQUALITY_DISABLE_SPACY", None)
        import app.application.use_cases.build_requirements as br
        br._SPACY_NLP = None

    def test_extracts_title_case_terms(self):
        from app.application.use_cases.build_requirements import _extract_entities
        result = _extract_entities("Программа должна использовать TypeScript и Angular.")
        # Title-case Russian word + English Title-case + acronym-like all-caps
        assert any("TypeScript" in e or "typescript" in e.lower() for e in result)
        assert any("Angular" in e or "angular" in e.lower() for e in result)

    def test_extracts_acronyms(self):
        from app.application.use_cases.build_requirements import _extract_entities
        result = _extract_entities("API использует JSON для передачи данных.")
        result_lower = [e.lower() for e in result]
        assert "api" in result_lower
        assert "json" in result_lower

    def test_misses_mid_sentence_lowercase_phrases(self):
        """Documents the regex fallback's known limitation: lowercase
        phrases mid-sentence (no title-case lead word) are not extracted.
        spaCy path fixes this. We use a sentence where 'входные данные' /
        'регулярные выражения' sit mid-sentence after a verb — regex
        cannot anchor, so it misses them."""
        from app.application.use_cases.build_requirements import _extract_entities
        result = _extract_entities(
            "Система проверяет входные данные и применяет регулярные выражения."
        )
        result_lower = [e.lower() for e in result]
        # These multi-word lowercase phrases are missed by the regex —
        # they would surface only via spaCy noun-chunks.
        assert "входные данные" not in result_lower
        assert "регулярные выражения" not in result_lower

    def test_empty_text_returns_empty(self):
        from app.application.use_cases.build_requirements import _extract_entities
        assert _extract_entities("") == []
        assert _extract_entities("   ") == []


# ── spaCy path (skipped if model unavailable) ────────────────────────────


@pytest.fixture
def spacy_extractor():
    """Returns the live extractor with spaCy loaded, or skips the test."""
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("ru_core_news_md")
    except OSError:
        pytest.skip(
            "ru_core_news_md not installed; run "
            "`python -m spacy download ru_core_news_md` to enable spaCy tests"
        )
    os.environ.pop("CQUALITY_DISABLE_SPACY", None)
    import app.application.use_cases.build_requirements as br
    br._SPACY_NLP = None  # force re-probe
    yield br._extract_entities
    br._SPACY_NLP = None


class TestSpacyPath:
    """When spaCy + ru_core_news_md are present, lowercase domain
    phrases must be extracted as multi-word noun chunks. This is the
    path that fixes the Polyakov Rule 5 false demotions."""

    def test_lowercase_domain_phrase_extracted(self, spacy_extractor):
        """Audit (Polyakov 0.41::sent4): 'перечень функций программы',
        'методы испытаний', 'технические средства' must all surface."""
        result = spacy_extractor(
            "Перечень функций программы, методы испытаний, технические средства."
        )
        result_lemma = [e.lower() for e in result]
        # spaCy noun_chunks return surface forms; lemmas are normalised
        # internally for dedup. We assert the surface form is present.
        assert any("функций" in e or "функция" in e for e in result_lemma), result
        assert any("испытани" in e for e in result_lemma), result

    def test_acronyms_still_extracted(self, spacy_extractor):
        """Acronym sweep must run even when spaCy is active."""
        result = spacy_extractor("Все запросы передаются через REST API в формате JSON.")
        result_lower = [e.lower() for e in result]
        assert "rest" in result_lower or "api" in result_lower
        assert "json" in result_lower

    def test_polyakov_0_41_paraphrase_overlap(self, spacy_extractor):
        """End-to-end: req and unit that paraphrase each other must
        produce ≥1 shared entity. Pre-spaCy this returned 0 (Rule 5
        demoted PARTIAL → IRRELEVANT)."""
        req_entities = spacy_extractor(
            "Перечень функций программы, методы испытаний, "
            "технические средства, порядок проведения испытаний."
        )
        unit_entities = spacy_extractor(
            "Порядок проведения испытаний. Испытание проверки выполнения "
            "требований к программной документации."
        )

        # Lemmatise both for fair comparison (regardless of surface form).
        from app.core.lemmatize import lemma

        def _normalize(entities):
            return frozenset(
                frozenset(lemma(w) for w in e.lower().split() if w)
                for e in entities
            )

        shared = _normalize(req_entities) & _normalize(unit_entities)
        # Without lemmatisation we expect ≥ 1 shared lemma (испытани*).
        assert len(shared) >= 1, (
            f"spaCy must surface shared entities for paraphrase pairs; "
            f"req={req_entities}, unit={unit_entities}"
        )
