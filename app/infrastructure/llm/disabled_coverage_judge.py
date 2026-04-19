"""
Fallback judge used when LLM is disabled.

Decision priority (each path checked in order, first match wins):

  COVERED — one of these must be true:
    1. constraint kind match (same non-generic kind on both sides)
    2. normalized text exact match (after punct-strip)
    3. text containment  — req text ⊆ unit text or vice-versa (≥5 words)
    4. near-exact lexical match  — lex ≥ 0.80 with non-trivial object content
    5. verb match + object covers (Jaccard ≥ 0.6 or exact) + sec_plausible + lex ≥ 0.15
    6. verb match + object covers + lex ≥ 0.20
    X. artifact match — both texts share a deliverable artifact category (Jaccard ≥ 0.6)

  PARTIAL — topical relevance without confirmed coverage:
    7. verb match + partial object overlap [0.25, 0.6)
    8. high lex+entity BUT object match is weak  (demoted from old "Strong dual evidence")
    9. lex ≥ 0.20
   10. lex ≥ 0.12 AND entity_overlap ≥ 0.15
   XA. partial artifact category overlap (artifact_jac > 0)
   XV. PMI verification-step unit shortlisted against a TZ req (no sec_plausible required)
   10. shared content token count ≥ 2 (Jaccard too low due to length mismatch)

  IRRELEVANT — everything else (including verb match with disjoint object)

Key invariant: "Strong dual evidence" (lex ≥ 0.40 AND entity_ov ≥ 0.25) alone
is NEVER sufficient for COVERED when object match is weak.  It is demoted to
PARTIAL.  The exact/near-exact text-match paths handle cases where verb
extraction doesn't fire but the texts are essentially identical.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.core.text import tokenize_content
from app.domain.c_quality_enums import LLMLabel
from app.domain.c_quality_models import Constraint, CoverageUnit, PairJudgment, RequirementUnit
from app.infrastructure.llm.coverage_judge import CoverageJudge

# ---------------------------------------------------------------------------
# Action verb whitelist (requirement-statement verbs in TZ / PMI section 3/6)
# ---------------------------------------------------------------------------

_ACTION_VERB_RE = re.compile(
    r"\b(предоставлять|предоставить|обеспечивать|обеспечить|реализовывать|реализовать|"
    r"поддерживать|выполнять|выполнить|хранить|сохранять|осуществлять|осуществить|"
    r"контролировать|формировать|отображать|разграничивать|предусматривать|"
    r"ограничивать|регистрировать|фиксировать|вести|генерировать|экспортировать|"
    r"импортировать|передавать|принимать|проверять)\b",
    re.I | re.UNICODE,
)

# Object phrase ends at first occurrence of any of these characters
_OBJECT_BOUNDARY_RE = re.compile(r"[:;,.]")

# Punctuation strip regex for exact/containment text comparison
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Words that appear in requirement templates but don't distinguish specific requirements.
# Filtered from object tokens before computing Jaccard overlap.
# Note: "система", "должен/а/ны/но" are already in core STOP_WORDS.
_OBJECT_BOILERPLATE: frozenset = frozenset({
    "возможность", "возможности", "возможностью", "возможностей",
    "функциональность", "функциональности",
    "интерфейс", "интерфейса", "интерфейсу", "интерфейсе", "интерфейсы",
    "приложение", "приложения", "приложении", "приложению",
    "программный", "программного", "программному", "программном",
    "программа", "программы", "программе", "программой",
    "продукт", "продукта", "продукту", "продуктом", "продуктов",
    "разрабатываемый", "разрабатываемого", "разрабатываемому",
    "системы", "системе", "системой", "систему",   # "система" already in STOP_WORDS
    "об", "о",                                     # prepositions absent from STOP_WORDS
})

_SECTION_NUM_RE = re.compile(r"^\s*(\d+)(?:[.\s]|$)")

# Thresholds
_NEAR_EXACT_LEX_THRESHOLD = 0.80   # lex for near-exact text match path
_MIN_CONTAINMENT_WORDS = 5         # minimum req words for text-containment check

# ---------------------------------------------------------------------------
# Document artifact canonical map — morphological variants → canonical category
# ---------------------------------------------------------------------------

_ARTIFACT_CANONICAL: dict = {
    "программа и методика испытаний": "pmi_doc",
    "программы и методики испытаний": "pmi_doc",
    "программ и методик испытаний": "pmi_doc",
    "программу и методику испытаний": "pmi_doc",
    "программой и методикой испытаний": "pmi_doc",
    "руководство оператора": "operator_manual",
    "руководства оператора": "operator_manual",
    "руководству оператора": "operator_manual",
    "руководством оператора": "operator_manual",
    "руководстве оператора": "operator_manual",
    "руководство пользователя": "user_manual",
    "руководства пользователя": "user_manual",
    "руководству пользователя": "user_manual",
    "руководством пользователя": "user_manual",
    "руководстве пользователя": "user_manual",
    "текст программы": "program_text",
    "тексте программы": "program_text",
    "текста программы": "program_text",
    "текстом программы": "program_text",
    "пояснительная записка": "explanatory_note",
    "пояснительной записки": "explanatory_note",
    "пояснительную записку": "explanatory_note",
    "пояснительной запиской": "explanatory_note",
    "техническое задание": "tech_spec",
    "технического задания": "tech_spec",
    "техническому заданию": "tech_spec",
    "техническим заданием": "tech_spec",
    "программная документация": "prog_docs",
    "программной документации": "prog_docs",
    "программную документацию": "prog_docs",
    # GOST 19 document family
    "описание программы": "program_description",
    "описания программы": "program_description",
    "описание алгоритма": "algorithm_description",
    "описания алгоритма": "algorithm_description",
    "спецификация требований": "requirements_spec",
    "спецификации требований": "requirements_spec",
    "спецификацию требований": "requirements_spec",
    "ведомость держателей подлинников": "document_registry",
    "перечень эксплуатационных документов": "operational_docs_list",
    "перечня эксплуатационных документов": "operational_docs_list",
    "формуляр": "formular",
    "формуляра": "formular",
    "ведомость эксплуатационных документов": "operational_docs_list",
}

# Regex for PMI verification-step units: "Пункт N) проверяется ...", etc.
_VERIFICATION_UNIT_RE = re.compile(
    r"""
    пункт\w*\s+\S+\s*[).:\s]*проверяется
    | проверяется\s+(через|с\s+помощью|с\s+использованием|методом|путём|путем)
    | для\s+проверки\s+(пункта|требования|данного|указанного)
    | метод\w*\s+(верификации|проверки)\w*
    | тестирование\w*\s+(предусматривает|включает|проводится)
    """,
    re.I | re.VERBOSE | re.UNICODE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _entity_overlap(entities_a: List[str], entities_b: List[str]) -> float:
    if not entities_a or not entities_b:
        return 0.0
    a = {e.lower() for e in entities_a}
    b = {e.lower() for e in entities_b}
    return len(a & b) / len(a | b)


def _has_meaningful_constraint_kind_match(
    req_constraints: List[Constraint],
    unit_constraints: List[Constraint],
) -> bool:
    """True when both sides share at least one non-generic constraint kind."""
    req_kinds = {c.kind for c in req_constraints if c.kind and c.kind != "generic"}
    unit_kinds = {c.kind for c in unit_constraints if c.kind and c.kind != "generic"}
    return bool(req_kinds & unit_kinds)


def _strip_punct(text: str) -> str:
    """Remove punctuation and collapse whitespace — used for exact/containment matching."""
    return " ".join(_PUNCT_RE.sub(" ", text).split())


def _text_match_signals(
    req: RequirementUnit, unit: CoverageUnit
) -> Tuple[bool, bool, str]:
    """
    Compute exact / containment text-match signals.

    Returns:
        text_exact      — stripped normalized texts are identical
        text_containment — one stripped text is a word-boundary-aligned substring
                           of the other (req must have ≥ _MIN_CONTAINMENT_WORDS)
        match_label     — short label for logging / matched_aspects
    """
    req_s = _strip_punct(req.normalized_text)
    unit_s = _strip_punct(unit.normalized_text)

    if req_s == unit_s:
        return True, False, "normalized_text_exact_match"

    # Word-boundary-padded containment: " req " in " unit " or vice-versa
    req_words = req_s.split()
    unit_words = unit_s.split()
    containment = False
    if len(req_words) >= _MIN_CONTAINMENT_WORDS:
        containment = f" {req_s} " in f" {unit_s} "
    if not containment and len(unit_words) >= _MIN_CONTAINMENT_WORDS:
        containment = f" {unit_s} " in f" {req_s} "

    if containment:
        return False, True, "text_containment"
    return False, False, ""


def _extract_action_object(text: str) -> Tuple[Optional[str], str, frozenset]:
    """
    Return (verb, object_phrase, object_tokens).

    Finds the first action verb, extracts the phrase following it up to the
    first sentence boundary.  Object tokens have stop words and requirement
    boilerplate removed.  Returns (None, "", frozenset()) if no verb found.
    """
    m = _ACTION_VERB_RE.search(text)
    if not m:
        return None, "", frozenset()

    verb = m.group(1).lower()
    after_verb = text[m.end():].strip()
    boundary = _OBJECT_BOUNDARY_RE.search(after_verb)
    object_phrase = after_verb[:boundary.start()].strip() if boundary else after_verb.strip()

    raw_tokens = tokenize_content(object_phrase.lower())
    object_tokens = frozenset(raw_tokens - _OBJECT_BOILERPLATE)
    return verb, object_phrase.lower(), object_tokens


def _top_shared_tokens(a: set, b: set, n: int = 5) -> List[str]:
    """Return up to n shared tokens, sorted for determinism."""
    return sorted(a & b)[:n]


def _extract_artifact_categories(text: str) -> frozenset:
    """Return canonical artifact category names found in text (any morphological form)."""
    text_lower = text.lower()
    return frozenset(canon for phrase, canon in _ARTIFACT_CANONICAL.items() if phrase in text_lower)


def _artifact_jaccard(req_arts: frozenset, unit_arts: frozenset) -> float:
    if not req_arts or not unit_arts:
        return 0.0
    return len(req_arts & unit_arts) / len(req_arts | unit_arts)


def _is_verification_unit(text: str) -> bool:
    """True when text describes a PMI verification step rather than a testable requirement."""
    return bool(_VERIFICATION_UNIT_RE.search(text))


def _section_role_plausible(req: RequirementUnit, unit: CoverageUnit) -> bool:
    """TZ section 4.x paired with PMI section 3.x or 6.x is structurally plausible."""
    req_sec = (req.source_section_id or "").strip()
    unit_sec = (unit.section_id or "").strip()
    if not req_sec or not unit_sec:
        return False

    def _leading(s: str) -> Optional[int]:
        m2 = _SECTION_NUM_RE.match(s)
        return int(m2.group(1)) if m2 else None

    return _leading(req_sec) == 4 and _leading(unit_sec) in (3, 6)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class DisabledCoverageJudge(CoverageJudge):
    """
    Deterministic conservative pseudo-judge for when LLM is unavailable.

    COVERED requires one of: constraint-kind match, exact/near-exact text match,
    or verb + object phrase match.  "Strong dual evidence" (high lex + entity)
    alone is NOT sufficient — it is capped at PARTIAL to prevent neighboring
    template-like requirements from cross-classifying as COVERED.
    """

    def judge(self, req: RequirementUnit, unit: CoverageUnit) -> PairJudgment:
        req_tokens = tokenize_content(req.normalized_text)
        unit_tokens = tokenize_content(unit.normalized_text)
        lex = _jaccard(req_tokens, unit_tokens)

        entity_ov = _entity_overlap(req.entities, unit.entities)
        ck_match = _has_meaningful_constraint_kind_match(req.constraints, unit.constraints)

        # Text-level exact / containment signals
        text_exact, text_containment, text_match_label = _text_match_signals(req, unit)

        # Object-aware phrase extraction
        req_verb, req_obj_phrase, req_obj_tokens = _extract_action_object(req.normalized_text)
        unit_verb, unit_obj_phrase, unit_obj_tokens = _extract_action_object(unit.normalized_text)

        verb_match = bool(req_verb and unit_verb and req_verb == unit_verb)
        sec_plausible = _section_role_plausible(req, unit)

        object_overlap = _jaccard(req_obj_tokens, unit_obj_tokens) if verb_match else 0.0
        object_exact_match = bool(
            verb_match and req_obj_phrase and req_obj_phrase == unit_obj_phrase
        )
        object_covers = object_exact_match or object_overlap >= 0.6

        # Near-exact lexical match: very high token overlap with non-trivial content.
        # Catches cases where verb extraction doesn't fire but texts are near-identical.
        non_trivial_obj = len(req_obj_tokens | unit_obj_tokens) >= 2
        near_exact_lex = lex >= _NEAR_EXACT_LEX_THRESHOLD and non_trivial_obj

        # Artifact-aware signals: deliverable document categories
        req_artifacts = _extract_artifact_categories(req.normalized_text)
        unit_artifacts = _extract_artifact_categories(unit.normalized_text)
        artifact_jac = _artifact_jaccard(req_artifacts, unit_artifacts)

        # Verification-unit detection: PMI step that describes how to verify, not what to test
        is_verify = _is_verification_unit(unit.normalized_text)

        # Shared content tokens — used for explainability in all PARTIAL paths
        shared_tokens = req_tokens & unit_tokens
        shared_token_count = len(shared_tokens)
        top_shared = _top_shared_tokens(req_tokens, unit_tokens)

        # ── build signal string ────────────────────────────────────────────
        signals: List[str] = [f"lex={lex:.2f}"]
        matched_aspects: List[str] = []

        if ck_match:
            ck_kinds = sorted(
                {c.kind for c in req.constraints if c.kind and c.kind != "generic"}
                & {c.kind for c in unit.constraints if c.kind and c.kind != "generic"}
            )
            signals.append(f"constraint_kind={'+'.join(ck_kinds)}")
            matched_aspects.extend(ck_kinds)
        if entity_ov > 0:
            signals.append(f"entity_overlap={entity_ov:.2f}")
        if text_match_label:
            signals.append(text_match_label)
        if near_exact_lex and not text_exact and not text_containment:
            signals.append(f"near_exact_lex={lex:.2f}")
        if verb_match:
            signals.append(f"verb={req_verb}")
            if object_exact_match:
                signals.append("obj=exact")
            elif object_overlap > 0:
                signals.append(f"obj_overlap={object_overlap:.2f}")
            else:
                signals.append("obj_overlap=0")
        if sec_plausible:
            signals.append("sec_prior=TZ4→PMI3/6")
        if artifact_jac > 0:
            signals.append(f"artifact_jac={artifact_jac:.2f}")
        if is_verify:
            signals.append("is_verify")

        signal_str = ", ".join(signals)
        missing_aspects: List[str] = []

        # ══════════════════════════════════════════════════════════════════
        # COVERED paths
        # ══════════════════════════════════════════════════════════════════

        # P1: shared named constraint kind (verifier validates exact values)
        if ck_match:
            label = LLMLabel.COVERED
            explanation = (
                f"[disabled-judge] Constraint-kind match ({signal_str}) "
                f"→ tentative COVERED; rule verifier will validate numeric values"
            )
            matched_aspects.append("constraint_kind_match")

        # P2: exact normalized text (same content, trivially verified)
        elif text_exact:
            label = LLMLabel.COVERED
            matched_aspects.append("normalized_text_exact_match")
            explanation = (
                f"[disabled-judge] Exact normalized text match ({signal_str}) → COVERED"
            )

        # P3: req text verbatim inside unit text (PMI restates TZ requirement)
        elif text_containment:
            label = LLMLabel.COVERED
            matched_aspects.append("text_containment")
            explanation = (
                f"[disabled-judge] TZ requirement text contained verbatim in PMI unit "
                f"({signal_str}) → COVERED"
            )

        # P4: near-exact lexical match (≥0.80 lex, non-trivial object content)
        elif near_exact_lex:
            label = LLMLabel.COVERED
            matched_aspects.append("near_exact_lex")
            explanation = (
                f"[disabled-judge] Near-exact lexical match ({signal_str}) → COVERED"
            )

        # P5: verb + object match + structurally plausible section pairing
        elif verb_match and object_covers and sec_plausible and lex >= 0.15:
            label = LLMLabel.COVERED
            obj_desc = f'"{req_obj_phrase}"' if req_obj_phrase else "(no object)"
            matched_aspects += [f"action_verb:{req_verb}", f"object_overlap:{req_obj_phrase}"]
            explanation = (
                f"[disabled-judge] Same action verb and object phrase match {obj_desc} "
                f"+ TZ4→PMI3/6 section pairing ({signal_str}) → COVERED"
            )

        # P6: verb + object match + moderate lexical support (no section info)
        elif verb_match and object_covers and lex >= 0.20:
            label = LLMLabel.COVERED
            obj_desc = f'"{req_obj_phrase}"' if req_obj_phrase else "(no object)"
            matched_aspects += [f"action_verb:{req_verb}", f"object_overlap:{req_obj_phrase}"]
            explanation = (
                f"[disabled-judge] Same action verb and object phrase match {obj_desc} "
                f"({signal_str}) → COVERED"
            )

        # PX_ART: both texts share the same deliverable artifact category
        elif artifact_jac >= 0.6:
            label = LLMLabel.COVERED
            matched_aspects.append(f"artifact_match:{'+'.join(sorted(req_artifacts & unit_artifacts))}")
            explanation = (
                f"[disabled-judge] Matching document artifact category "
                f"({signal_str}) → COVERED"
            )

        # ══════════════════════════════════════════════════════════════════
        # PARTIAL paths
        # ══════════════════════════════════════════════════════════════════

        # P7: verb matches, partial object overlap
        elif verb_match and object_overlap >= 0.25:
            label = LLMLabel.PARTIAL
            matched_aspects += [
                f"action_verb_match:{req_verb}",
                f"partial_object_overlap:{object_overlap:.2f}",
            ]
            if top_shared:
                matched_aspects.append(f"shared_tokens:{','.join(top_shared)}")
            missing_aspects = [
                "exact_object_match",
                f"need_object_overlap>=0.6_got_{object_overlap:.2f}",
            ]
            explanation = (
                f"[disabled-judge] Same action verb ({req_verb!r}) with partial object overlap "
                f'(req_obj={req_obj_phrase!r:.40} vs unit_obj={unit_obj_phrase!r:.40}, '
                f"{signal_str}) → PARTIAL; object coverage insufficient for COVERED"
            )

        # P8: high lex + entity but object match is weak — DEMOTED from old "Strong dual evidence"
        elif lex >= 0.40 and entity_ov >= 0.25:
            label = LLMLabel.PARTIAL
            shared_entities = sorted(
                {e.lower() for e in req.entities} & {e.lower() for e in unit.entities}
            )[:5]
            matched_aspects += [
                f"lex:{lex:.2f}",
                f"entity_overlap:{entity_ov:.2f}",
            ]
            if shared_entities:
                matched_aspects.append(f"shared_entities:{','.join(shared_entities)}")
            if top_shared:
                matched_aspects.append(f"shared_tokens:{','.join(top_shared)}")
            missing_aspects = [
                "object_phrase_match",
                "verb_object_coverage",
            ]
            explanation = (
                f"[disabled-judge] High lex+entity overlap but object phrase weak "
                f"({signal_str}); shared_entities=[{','.join(shared_entities)}] → PARTIAL"
            )

        # P9: moderate lexical overlap
        elif lex >= 0.20 or (lex >= 0.12 and entity_ov >= 0.15):
            label = LLMLabel.PARTIAL
            matched_aspects += [f"lex:{lex:.2f}"]
            if top_shared:
                matched_aspects.append(f"shared_tokens:{','.join(top_shared)}")
            if entity_ov > 0:
                matched_aspects.append(f"entity_overlap:{entity_ov:.2f}")
            missing_aspects = [
                "specific_object_match",
                "constraint_match",
                "exact_text_overlap",
            ]
            explanation = (
                f"[disabled-judge] Topical overlap: shared_tokens=[{','.join(top_shared)}] "
                f"({signal_str}) → PARTIAL; no specific object/constraint match"
            )

        # PX_ART_PARTIAL: partial artifact category overlap
        elif artifact_jac > 0:
            label = LLMLabel.PARTIAL
            shared_arts = sorted(req_artifacts & unit_artifacts)
            matched_aspects += [
                f"artifact_family:{'+'.join(shared_arts)}",
                "document_family_overlap",
            ]
            missing_aspects = [
                "exact_artifact_identity",
                "full_artifact_coverage",
            ]
            explanation = (
                f"[disabled-judge] Partial document artifact family overlap "
                f"(families=[{','.join(shared_arts)}], {signal_str}) → PARTIAL; "
                f"exact artifact not confirmed"
            )

        # PX_VERIFY: PMI verification-step unit shortlisted against a TZ requirement.
        # sec_plausible is NOT required — many PMI fragments lack section_id.
        # Retrieval shortlisting already provides topical relevance gate.
        elif is_verify:
            label = LLMLabel.PARTIAL
            matched_aspects += [
                "verification_unit",
                "testing_context_present",
            ]
            if top_shared:
                matched_aspects.append(f"shared_tokens:{','.join(top_shared)}")
            missing_aspects = [
                "exact_requirement_text",
                "direct_functional_match",
            ]
            explanation = (
                f"[disabled-judge] PMI verification-step unit shortlisted for "
                f"TZ requirement (shared_tokens=[{','.join(top_shared)}], {signal_str}) "
                f"→ PARTIAL; does not directly state requirement"
            )

        # P10: at least 3 shared content tokens AND lex ≥ 0.12 — catches low-lex topical overlap
        # where Jaccard is low due to length disparity but vocabulary is clearly related.
        # Threshold kept conservative (≥3 + lex≥0.12) to avoid promoting pairs that share
        # only document-wide boilerplate tokens (e.g. "программирования", "языка").
        elif shared_token_count >= 3 and lex >= 0.12:
            label = LLMLabel.PARTIAL
            matched_aspects += [
                f"shared_content_words:{','.join(top_shared)}",
                f"shared_token_count:{shared_token_count}",
            ]
            missing_aspects = [
                "sufficient_lexical_density",
                "object_phrase_match",
                "verb_match",
            ]
            explanation = (
                f"[disabled-judge] Weak but non-zero vocabulary overlap: "
                f"shared_tokens=[{','.join(top_shared)}] ({signal_str}) → PARTIAL; "
                f"Jaccard too low for stronger claim"
            )

        # ══════════════════════════════════════════════════════════════════
        # IRRELEVANT
        # ══════════════════════════════════════════════════════════════════

        else:
            label = LLMLabel.IRRELEVANT
            if verb_match:
                explanation = (
                    f"[disabled-judge] Same action verb ({req_verb!r}) but disjoint objects "
                    f'(req={req_obj_phrase!r:.40} vs unit={unit_obj_phrase!r:.40}) → IRRELEVANT'
                )
            else:
                if lex > 0 and shared_token_count > 0:
                    explanation = (
                        f"[disabled-judge] Insufficient overlap "
                        f"(shared_tokens=[{','.join(top_shared)}], {signal_str}) → IRRELEVANT"
                    )
                else:
                    explanation = (
                        f"[disabled-judge] No shared content tokens or structural signals "
                        f"({signal_str}) → IRRELEVANT"
                    )

        return PairJudgment(
            req_id=req.req_id,
            unit_id=unit.unit_id,
            target_document_id=unit.target_document_id,
            llm_label=label,
            llm_confidence=round(max(lex, 0.6 if ck_match else 0.0), 3),
            rule_adjusted_label=label,
            matched_aspects=matched_aspects,
            missing_aspects=missing_aspects if label != LLMLabel.COVERED else [],
            explanation=explanation,
        )
