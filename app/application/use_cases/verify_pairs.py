"""
Stage 6: rule-based verifier applied after LLM judgment.

Rules (in priority order):
1. Numeric constraint conflict → override to CONFLICT
2. Negation/modality contradiction → override to CONFLICT
3. Req has constraints but unit has none → downgrade COVERED → PARTIAL
4. Entity mismatch (no shared entities) on "COVERED" → downgrade to PARTIAL

The verifier never upgrades a label; it can only keep or downgrade.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from app.core.logging import get_logger
from app.domain.c_quality_enums import LLMLabel, Modality
from app.domain.c_quality_models import (
    Constraint,
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
)

logger = get_logger(__name__)

_VALUE_TOLERANCE = 1e-6

# Strict prohibition markers only — excludes quantifiers like "не более / не менее".
#
# PR-K follow-up (sweep TZ#2): the old regex missed neuter "не должно",
# verb form "запрещается", and gender variants of "запрещён/недопустим",
# producing a false-positive CONFLICT in the smoke run on
# "Время отклика не должно превышать 2 секунд" vs the same PMI fragment
# (which uses masc "не должен"). All gender/number forms now recognised.
# Common Russian content words that pass the «≥5 chars» length gate but
# are too generic to count as a topical anchor on their own. Used by
# Rule 5 (PARTIAL preservation): we only consider a shared token a
# topical anchor if it survives this stoplist.
_CONTENT_TOKEN_STOPWORDS = frozenset({
    "система", "систему", "системы", "системе", "систем",
    "должен", "должна", "должно", "должны",
    "может", "могут",
    "также", "только", "также", "может", "может",
    "соответствии", "соответствует", "соответствия", "соответствующи",
    "следующий", "следующее", "следующие", "следующих",
    "необходимо", "обеспечив", "обеспечивает",
    "является", "являются",
    "целью", "целях",
    "данный", "данных", "данные",
    "пользователь", "пользователя", "пользователем", "пользователю", "пользователей",
    "программ", "программа", "программой", "программе", "программу", "программы",
    "приложен", "приложение", "приложения", "приложению", "приложением",
    "функций", "функции", "функция", "функцию", "функциях",
    "интерфейс", "интерфейса",
})


_PROHIBITION_RE = re.compile(
    r"\b(?:"
    # Russian "не должен / не должна / не должно / не должны"
    # (all four gender/number forms — "не должно" was the missing one).
    r"не\s+долж(?:ен|на|но|ны)|"
    # "запрещено / запрещена / запрещены" + verb forms "запрещается / запрещаются".
    r"запрещен(?:о|а|ы)?|запрещается|запрещаются|"
    # "недопустимо / недопустима / недопустимы".
    r"недопустим(?:о|а|ы)?|"
    # "не допускается / не допускаются / не разрешается / не разрешаются".
    r"не\s+допускается|не\s+допускаются|"
    r"не\s+разрешается|не\s+разрешаются|"
    # "без возможности (изменения|отката|…)" — preposition-class prohibition
    # ("без возможности" + dependent noun in genitive).
    r"без\s+возможност[ьи]|"
    # English variants.
    r"not\s+allowed|forbidden|prohibited"
    r")\b",
    re.I,
)


# ── Same-aspect / negation-compatibility table (PR-G refactor) ──────────
#
# CONFLICT may be raised only when both sides talk about the SAME
# aspect (same metric, same component, same operation). When the
# requirement and the unit talk about different aspects but happen to
# share lexical signals (e.g. both mention "не должно превышать" + a
# number), CONFLICT is a false positive.
#
# `_SAME_OUTCOME_PAIRS` lists known semantic-equivalence pairs: a
# prohibition phrase on one side and a positive affirmation on the
# other that describes the same outcome. The audit-time symptom was
# "система не должна аварийно завершаться" vs "система должна
# продолжать корректно функционировать" — both express continued
# operation under errors, so CONFLICT was wrong. The fix recognises
# the pair and suppresses the negation contradiction rule.

# P0 #7 (Калугин): the original list above held two narrow string-level
# pairs ("не должна аварийно завершаться" ↔ "должна продолжать
# корректно функционировать"). Any new semantic equivalence the LLM
# produced — «не должна завершаться с ошибкой» vs «должна обрабатывать
# ошибки» — slipped past and the negation rule fired CONFLICT. The
# replacement is a stem-level pattern table: each entry is
# (banned-bad-outcome-stem, equivalent-good-outcome-stem); both sides
# are matched in either order. Stems are deliberately broad (аварийн |
# с ошибк | некорректн | нестабильн | со сбо | прерыван | падат |
# крашит) so that surface-form variation does not produce false-CONFLICT
# rows.
_SAME_OUTCOME_PAIRS: list[tuple[re.Pattern, re.Pattern]] = [
    # ── Continued operation under failure ────────────────────────────
    # Banned: «не должн[аоы] аварийно завершать[ся] | падать | крашить
    # | прерывать работу | завершаться с ошибкой | работать с ошибками
    # | работать некорректно | работать нестабильно | работать со сбоями».
    (
        re.compile(
            r"не\s+долж(?:ен|на|но|ны)\s+(?:"
            r"аварийн\w+\s+заверш"
            r"|падат\w*"
            r"|крашит\w*"
            r"|прерыват\w*\s+работ"
            r"|заверш\w+\s+(?:с\s+ошибк|со\s+сбо)"
            r"|работат\w+\s+(?:с\s+ошибк|со\s+сбо|некорректн|нестабильн)"
            r"|выполнят\w+\s+(?:некорректн|нестабильн|со\s+сбо)"
            r")",
            re.I,
        ),
        re.compile(
            r"(?:долж\w+|обязан\w*)\s+(?:"
            r"продолжат\w+\s+(?:корректн\w*\s+)?(?:работ|функционир)"
            r"|(?:корректн\w*|стабильн\w*)\s+(?:работат|функционир)"
            r"|работат\w+\s+(?:корректн|стабильн|без\s+(?:ошиб|сбо))"
            r"|функционир\w+\s+(?:корректн|стабильн|без\s+(?:ошиб|сбо))"
            r"|обрабатыват\w+\s+ошибк"
            r"|обеспечив\w+\s+(?:непрерывн\w*|стабильн\w*|корректн\w*)\s+работ"
            r")"
            r"|продолж\w+\s+(?:корректн\w+\s+)?(?:функционир|работ)"
            r"|(?:корректн\w*|стабильн\w*)\s+(?:работа|функционирован)",
            re.I,
        ),
    ),
    # ── Data preservation ────────────────────────────────────────────
    # Banned: «не должн[аоы] терять / потерять / удалять безвозвратно
    # данные | информацию».
    (
        re.compile(
            r"не\s+долж(?:ен|на|но|ны)\s+(?:"
            r"терят\w*"
            r"|потерят\w*"
            r"|утрачив\w*"
            r"|удалят\w*\s+безвозврат\w*"
            r")\s+(?:дан|информаци)",
            re.I,
        ),
        re.compile(
            r"(?:долж\w+|обязан\w*)\s+(?:"
            r"сохранят\w*"
            r"|сберег\w*"
            r"|резервн\w*\s+копир"
            r"|обеспечив\w+\s+сохран"
            r")\s+(?:дан|информаци)|"
            r"долж\w+\s+(?:сохран|сберег|резервн)\w*\s+дан",
            re.I,
        ),
    ),
    # ── Access denial vs explicit availability ───────────────────────
    # Banned: «не должн[оы] блокировать (доступ|ввод|пользователю)»;
    # equivalent positive: «должн[оы] предоставлять / обеспечивать
    # доступ / возможность».
    (
        re.compile(
            r"не\s+долж(?:ен|на|но|ны)\s+(?:блокироват\w*|препятствоват\w*|"
            r"огранич\w+\s+доступ\w*)",
            re.I,
        ),
        re.compile(
            r"(?:долж\w+|обязан\w*)\s+(?:предоставлят\w+|обеспечив\w+|"
            r"гарантироват\w+)\s+(?:доступ|возможност)",
            re.I,
        ),
    ),
]


def _same_outcome_negation_compatible(
    req_text: str,
    unit_text: str,
    *,
    check_upper_bound: bool = True,
) -> bool:
    """True when one side prohibits a bad outcome while the other
    affirms the equivalent positive outcome — these are semantically
    compatible, never CONFLICT.

    Args:
        req_text: requirement text.
        unit_text: coverage unit text.
        check_upper_bound: when True (default), also recognises the
            structural pattern where BOTH sides use the same upper-bound
            prohibition phrasing ("не должно превышать X / Y") — a
            same-modality / same-direction constraint that is NOT a
            negation contradiction. This guard is appropriate for Rule 2
            (negation-contradiction check), where it prevents the rule
            from firing on structurally identical but topically related
            texts.

            Set to False when calling from the LLM-native CONFLICT
            pre-check (before the numeric rule runs). In that context
            the structural `same_upper_bound` pattern is insufficient:
            "не должно превышать" on DIFFERENT metrics (e.g., "время
            отклика" vs "время восстановления") would incorrectly
            suppress a correct LLM CONFLICT verdict. The numeric rule
            and the aggregator's unverified-CONFLICT gate are better
            positioned to handle such cases.
    """
    rt = (req_text or "").lower()
    ut = (unit_text or "").lower()
    for prohib_re, pos_re in _SAME_OUTCOME_PAIRS:
        # Either ordering: requirement-prohibits + unit-affirms,
        # or vice-versa.
        if prohib_re.search(rt) and pos_re.search(ut):
            return True
        if prohib_re.search(ut) and pos_re.search(rt):
            return True

    if not check_upper_bound:
        return False

    # PR-K P4: when BOTH sides use the same upper-bound prohibition
    # phrasing ("не должно превышать X" on TZ side, "не должно превышать
    # Y" on the unit side), this is a same-modality / same-direction
    # constraint — the verifier's mismatch detector wrongly fires
    # because of regex-level differences between gender forms. They
    # are compatible (both upper-bound; numeric values are checked
    # separately by the numeric rule). Real-package symptom (Polyakov
    # 0.20::sent1).
    #
    # NOTE: this branch is only reached from Rule 2 (negation-
    # contradiction check), not from the LLM-CONFLICT pre-check —
    # see the check_upper_bound parameter.
    same_upper_bound = re.compile(
        r"не\s+долж(?:ен|на|но|ны)\s+превышат",
        re.I | re.UNICODE,
    )
    if same_upper_bound.search(rt) and same_upper_bound.search(ut):
        return True
    return False


def _types_can_conflict(
    req: "RequirementUnit",
    unit: "CoverageUnit",
    judgment_text_overrides: tuple[str, str] | None = None,
) -> bool:
    """Whether the requirement and the coverage unit are about a
    sufficiently-aligned aspect to allow CONFLICT.

    Decision rules:
      * Out-of-scope requirement types (DELIVERY, PROCESS, ECONOMIC)
        never produce CONFLICT — coverage isn't the question for them.
      * Documentation / environment requirements only conflict when both
        sides talk about the same documentation aspect; numeric mismatches
        on hardware specs ("16 ГБ" in PMI bench vs "32 ГБ" in TZ
        recommendation) are not real conflicts.
      * Otherwise default to True; the per-rule logic in PairVerifier
        will refine.
    """
    from app.domain.c_quality_enums import RequirementType
    rt = req.requirement_type
    if rt in {
        RequirementType.DELIVERY_REQUIREMENT,
        RequirementType.PROCESS_REQUIREMENT,
        RequirementType.ECONOMIC_OR_NEED,
    }:
        return False
    return True


_TIME_UNITS = {"days", "hours", "min", "sec", "ms"}
_SIZE_UNITS = {"kb", "mb", "gb", "tb"}
_RATE_UNITS = {"rps", "rpm"}

# Factors that convert a given unit into the canonical unit of its class.
# Canonicals: seconds for time, bytes for size, requests-per-second for rate.
# A value in unit U can be compared to a value in unit V (same class) by
# multiplying each by its factor, then comparing numerically.
_UNIT_TO_CANONICAL: dict = {
    # time → seconds
    "ms":   0.001,
    "sec":  1.0,
    "min":  60.0,
    "hours": 3600.0,
    "days": 86400.0,
    # size → bytes
    "kb": 1024.0,
    "mb": 1024.0 ** 2,
    "gb": 1024.0 ** 3,
    "tb": 1024.0 ** 4,
    # rate → rps
    "rps": 1.0,
    "rpm": 1.0 / 60.0,
}


def _canonical_value(value: float, unit: Optional[str]) -> Optional[float]:
    """
    Convert `value` from `unit` to the canonical unit of its class.

    Returns None if the unit has no canonical mapping (e.g. "%", None) —
    callers should then fall back to strict equality by unit string.
    """
    if not unit:
        return None
    factor = _UNIT_TO_CANONICAL.get(unit)
    if factor is None:
        return None
    return value * factor


def _same_unit_class(u1: Optional[str], u2: Optional[str]) -> bool:
    """Are two units in the same equivalence class?"""
    for cls in (_TIME_UNITS, _SIZE_UNITS, _RATE_UNITS):
        if u1 in cls and u2 in cls:
            return True
    return u1 == u2


def _values_conflict(rc: Constraint, uc: Constraint) -> Optional[str]:
    """
    Return a conflict description if the two constraints are incompatible,
    None otherwise.

    Guards (in order):
    1. Must be in the same unit class (e.g. both time, both size).
    2. Unitless constraints (unit=None on both sides) are skipped — they are
       likely document numbers, years, section references, or IP fragments
       extracted by the constraint parser; comparing them cross-document is
       pure noise.
    3. If both constraints carry an explicit, specific kind and the kinds differ,
       they belong to different aspects and must not be compared
       (e.g. retention_period vs response_time both measured in seconds).
    4. Same value → no conflict.
    """
    if not _same_unit_class(rc.unit, uc.unit):
        return None  # different measurement class

    # Skip unitless pairs — too noisy (years, counts, section numbers, …)
    if rc.unit is None and uc.unit is None:
        return None

    # Different named aspects in the same unit class → not comparable
    if rc.kind and uc.kind and rc.kind != uc.kind:
        return None

    # Compare in a common unit when possible: "90 дней" vs "3 месяца" (if a
    # months→days converter existed) or "2 сек" vs "2000 мс" must NOT fire
    # a conflict. Fall back to strict value equality when either unit lies
    # outside the known canonical table.
    rc_canon = _canonical_value(rc.value, rc.unit)
    uc_canon = _canonical_value(uc.value, uc.unit)
    if rc_canon is not None and uc_canon is not None:
        # Tolerance: 0.1% of the larger value — handles rounding in
        # conversions like 60*60*24 days without false conflicts.
        tol = max(abs(rc_canon), abs(uc_canon)) * 1e-3
        if abs(rc_canon - uc_canon) <= max(tol, _VALUE_TOLERANCE):
            return None
        return (
            f"req[{rc.kind}]: {rc.operator}{rc.value} {rc.unit or ''} "
            f"(~{rc_canon:g} canon) | "
            f"unit[{uc.kind}]: {uc.operator}{uc.value} {uc.unit or ''} "
            f"(~{uc_canon:g} canon)"
        )

    # No canonical conversion available — require exact match on raw value.
    if abs(rc.value - uc.value) < _VALUE_TOLERANCE:
        return None

    return (
        f"req[{rc.kind}]: {rc.operator}{rc.value} {rc.unit or ''} | "
        f"unit[{uc.kind}]: {uc.operator}{uc.value} {uc.unit or ''}"
    )


def _find_numeric_conflict(
    req_constraints: List[Constraint],
    unit_constraints: List[Constraint],
) -> List[str]:
    conflicts: List[str] = []
    for rc in req_constraints:
        for uc in unit_constraints:
            desc = _values_conflict(rc, uc)
            if desc:
                conflicts.append(desc)
    return conflicts


def _negation_contradiction(req: RequirementUnit, unit: CoverageUnit) -> bool:
    """
    True only when a requirement with explicit MUST_NOT modality is covered by
    a unit that makes a positive assertion (or vice-versa).

    Quantifier-class prohibitions («не должно превышать», «не более»,
    «не позднее», «не менее») are NOT treated as prohibitive for the
    purposes of this rule — they declare a numeric bound, not a banned
    behaviour. Real-package symptom (Polyakov 0.20::sent1):
    «время восстановления не должно превышать общее время…» (TZ) was
    matched against a positive-phrasing unit «Система должна корректно
    обрабатывать неверные запросы…» (PMI). The negation rule fired
    because TZ had «не должн» and PMI did not, then the negation-rule
    suppression for «same upper bound on both sides» couldn't trigger
    (PMI doesn't have «не должно превышать»), and the result was a
    false-positive CONFLICT.

    Numeric-bound conflicts on the same metric («3 сек» vs «10 сек»)
    are handled by Rule 1 (numeric conflict) — that rule has the
    appropriate aspect/topic guards. Stripping quantifier-prohibitions
    from this rule is therefore safe.
    """

    def _is_quantifier_only_prohibition(text: str) -> bool:
        """True when the only prohibition in `text` is a numeric-bound
        quantifier («не должно превышать», «не более X»), with no
        action-banning prohibition («не должна аварийно завершать»,
        «запрещено», «недопустимо», …)."""
        if not text:
            return False
        for m in _PROHIBITION_RE.finditer(text):
            tail = text[m.end():m.end() + 80].lower()
            head = m.group(0).lower()
            # Quantifier-class tails: «не должн... превышать» / «не должн...
            # быть менее» / «не более N». «не позднее N». «не менее N».
            # If the prohibition doesn't continue with one of these, it's
            # an action-banning prohibition — keep it.
            quantifier_continuation = bool(re.match(
                r"\s*(?:превышат|быть\s+(?:менее|больше|более|меньше|не\s+менее)|"
                r"(?:более|менее|позднее|раньше)\s+\b)",
                tail, re.I,
            ))
            head_quantifier = bool(re.match(
                r"не\s+(?:более|менее|позднее|раньше)\b", head, re.I,
            ))
            if not (quantifier_continuation or head_quantifier):
                return False
        return True

    req_prohibited_raw = (
        req.modality == Modality.MUST_NOT
        or bool(_PROHIBITION_RE.search(req.normalized_text))
    )
    unit_prohibited_raw = (
        unit.modality == Modality.MUST_NOT if hasattr(unit, "modality")
        else bool(_PROHIBITION_RE.search(unit.normalized_text))
    )

    # Strip quantifier-only prohibitions. We treat the side as «not
    # prohibited for negation-rule purposes» — the side has a numeric
    # bound, which is Rule 1's responsibility, not Rule 2's.
    req_prohibited = req_prohibited_raw and not _is_quantifier_only_prohibition(
        req.normalized_text
    )
    unit_prohibited = unit_prohibited_raw and not _is_quantifier_only_prohibition(
        unit.normalized_text if hasattr(unit, "normalized_text") else ""
    )

    return req_prohibited != unit_prohibited


def _same_upper_bound_same_aspect(req: RequirementUnit, unit: CoverageUnit) -> bool:
    if not _same_outcome_negation_compatible(req.text, unit.text, check_upper_bound=True):
        return False

    from app.core.text import tokenize_content

    req_tokens = tokenize_content(req.normalized_text)
    unit_tokens = tokenize_content(unit.normalized_text)
    shared_tokens = req_tokens & unit_tokens
    shared_constraint_kinds = (
        {c.kind for c in req.constraints if c.kind and c.kind != "generic"}
        & {c.kind for c in unit.constraints if c.kind and c.kind != "generic"}
    )
    return bool(shared_constraint_kinds) or len(shared_tokens) >= 4 or _entity_overlap(req, unit) >= 0.20


# Methodology vocabulary used by the PMI-copy-without-methodology rule.
# A PMI fragment that just duplicates the requirement text from ТЗ
# without any of these words is not really describing how to verify —
# it's a copy. See verify_pairs PMI-copy rule for usage.
_PMI_METHODOLOGY_RE = re.compile(
    r"\b(?:"
    r"проверяется|проверка|проверки|проверить|проверяют|"
    r"тест(?:а|у|ом|ы|ов|ам|ах|овый|ируется)?|"
    r"тест[\s\-]?кейс|тестирование|тестирования|"
    r"методика|методики|методику|"
    r"испытани[ея]|испытаний|испытание|испытания|"
    r"критери[йия]|критериев|"
    r"выполняется\s+проверка|производится\s+проверка|"
    r"ожидаемый\s+результат|"
    r"условие\s+приемки|приёмки|приемочн|"
    r"процедура\s+(?:проверки|испытани)|"
    r"шаги\s+(?:проверки|тестирования)|"
    r"тест[\-\s]?план"
    r")\b",
    re.IGNORECASE,
)


def _pmi_has_methodology(text: str) -> bool:
    """True when `text` (a PMI fragment) contains methodology / test /
    verification vocabulary. Used by the PMI-copy-without-methodology
    rule in verify_pairs to distinguish 'PMI describes how to test'
    from 'PMI just quotes the requirement verbatim'."""
    if not text:
        return False
    return bool(_PMI_METHODOLOGY_RE.search(text))


def _entity_overlap(req: RequirementUnit, unit: CoverageUnit) -> float:
    if not req.entities or not unit.entities:
        return 0.0
    # Compare entities by lemmatised word-bag so "Модуль аутентификации"
    # and "модулем аутентификации" (same concept, different surface form)
    # count as a match.
    from app.core.lemmatize import lemma

    def _key(entity: str) -> frozenset:
        return frozenset(lemma(w) for w in entity.lower().split() if w)

    req_set = {_key(e) for e in req.entities}
    unit_set = {_key(e) for e in unit.entities}
    # Remove trivial empty keys that can appear for degenerate inputs
    req_set.discard(frozenset())
    unit_set.discard(frozenset())
    if not req_set or not unit_set:
        return 0.0
    return len(req_set & unit_set) / len(req_set | unit_set)


# Polyakov-regression (2026-05-10): tech-stack token families. When
# both the requirement and the evidence mention identifiers from the
# same family, the topic alignment is genuine — entity-overlap demoter
# must not fire. Each family is a frozenset of lowercased substring
# probes; the helper checks `probe in text.lower()` (no word-boundary
# strictness because we want «.ts» to match «...файлы .ts...» and
# «typescript» to match «typescriptом»).
_TECH_STACK_FAMILIES: dict[str, frozenset[str]] = {
    "client_lang": frozenset({
        "typescript", "javascript", "angular", "react", "vue",
        ".ts", ".tsx", ".js", ".jsx",
    }),
    "markup": frozenset({
        "html", "css", "scss", "sass", "less",
        ".html", ".css", ".scss", ".sass",
    }),
    "transport": frozenset({
        "rest api", "rest-api",
        " rest ", " rest.", " rest,", " rest;",
        " json ", " json.", " json,", " json;", " json:",
        "jsonом", "https",
        "http/", "graphql", "websocket",
    }),
    "spa_arch": frozenset({"spa", "single-page", "single page application"}),
    "container": frozenset({
        "docker", "docker-compose", "docker compose", "kubernetes",
        "k8s", " containerd",
    }),
    "vcs": frozenset({
        # Word-bounded variants — substring probes can't use \b, so we
        # enumerate the punctuation/space contexts that surround «git»
        # in normal Russian/English prose.
        " git ", " git.", " git,", " git;", " git:", " git\n",
        "git-репозитор", "git репозитор", "github", "gitlab",
    }),
    "design_tool": frozenset({"figma", "sketch", "adobe xd"}),
    "platform_dspace": frozenset({"dspace", "d-space"}),
    "wsh": frozenset({"вшэ", "вышка", "higher school of economics"}),
}


def _tech_stack_co_occurrence(req_text: str, unit_text: str) -> Optional[str]:
    """Return the family name (e.g. "client_lang") when the requirement
    and the evidence both mention any identifier from the same tech-
    stack family. Otherwise None.

    Catches Polyakov 0.27::sent1 — TZ «TypeScript с использованием
    библиотеки Angular» vs PZ «.ts, .html, .css» — both clearly point
    to the same client-side stack but neither lex-jaccard nor entity
    overlap can bridge the syntactic gap (extension form vs full name,
    English-vs-Russian transliteration).
    """
    if not req_text or not unit_text:
        return None
    rq = req_text.lower()
    un = unit_text.lower()
    for family, probes in _TECH_STACK_FAMILIES.items():
        req_hits = any(p in rq for p in probes)
        if not req_hits:
            continue
        unit_hits = any(p in un for p in probes)
        if unit_hits:
            return family
    return None


# Polyakov-regression Step 7 (2026-05-11): aspect-mismatch guards.
#
# The LLM judge frequently raises PARTIAL on pairs that share generic
# vocabulary («система», «интерфейс», «ошибка») but belong to
# different semantic topics. Real-package examples from the May-11
# Polyakov run:
#
#   * 0.14 «время отклика 3 сек» (response_time) vs PMI evidence
#     about hardware («Процессор Intel i5») / recovery time («время
#     восстановления при отказе») → judge said PARTIAL conf 0.6,
#     should be MISSING.
#   * 0.18::sent2 «устойчивость к атакам типа Внедрение кода»
#     (code_injection) vs PMI evidence about access roles
#     («разграничение доступа») → PARTIAL conf 0.7, should be
#     MISSING.
#   * 0.15::sent2 «макет в Figma» (figma_design) vs PMI evidence
#     about browser UI («интерактивный интерфейс в браузере») →
#     PARTIAL conf 0.7, should be MISSING.
#   * 0.17::sent4 «обработка данных с сервера»
#     (data_from_server) vs PMI evidence about invalid request
#     handling → COVERED conf 0.95 — different aspects.
#
# Strategy: classify each side into a small set of fine-grained
# topics (heuristic regex). When the requirement has topic A and the
# evidence has topic B such that (A,B) is on the mismatch list AND
# the requirement does NOT also have topic B → demote PARTIAL/COVERED
# to IRRELEVANT. The verifier's earlier preserve-paths (entity-rich
# / tech-stack / shared-substantive) all run before this — Step 7 is
# the LAST pre-aggregation guard, catching pairs that everything else
# accepted but that semantically don't match.

# Topic classification — ordered list of (topic_name, regex). First
# match wins; a requirement / unit can carry multiple topics so we
# collect ALL hits, not just the first.
_TOPIC_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("response_time", re.compile(
        r"врем\w*\s+(?:отклик|ответ)|отклик\w*\s+(?:приложен|сервис|систем)|"
        r"\bresponse\s*time\b|не\s+должно\s+превышать\s+\d+\s*сек",
        re.IGNORECASE | re.UNICODE,
    )),
    ("recovery_time", re.compile(
        r"врем\w*\s+восстановлен|восстановлен\w*\s+(?:после|при)\s+отказ|"
        r"перезагрузк\w+\s+(?:операц|систем|программ|компонент|составл)",
        re.IGNORECASE | re.UNICODE,
    )),
    ("hardware_specs", re.compile(
        r"процессор|intel|amd|ryzen|i[3579]-\d+|\d+\s*(?:гб|gb|мб|mb)\b|"
        r"оперативн\w*\s+памят|дисков\w*\s+пространств|ssd|hdd|"
        r"монитор\s*\(|разрешени\w*\s+\d+x\d+",
        re.IGNORECASE | re.UNICODE,
    )),
    ("code_injection", re.compile(
        r"внедрен\w*\s+код|инъекц\w+|sql\s*injection|xss|cross.site|"
        r"атак\w+\s+типа\s+«?(?:внедрен|инъекц)|"
        r"уязвим\w*\s+(?:к|против)\s+(?:внедр|инъекц)",
        re.IGNORECASE | re.UNICODE,
    )),
    ("access_control", re.compile(
        r"разграничен\w*\s+доступ|ролев\w*\s+модел|"
        r"роли\s+пользоват|права\s+доступ|"
        # «настройки доступа к объектам на основе ролей пользователей» —
        # don't require the «доступ» and «на основе ролей» to be
        # adjacent; allow up to a few intervening words.
        r"настройк\w*\s+доступ[а-я\s]*на\s+основ\w*\s+рол|"
        # Generic «на основе ролей» phrase.
        r"на\s+основ\w*\s+рол(?:ей|и|ям|ями|ях)\s+пользоват|"
        # «пользователи с различными ролями» — RBAC-test phrasing.
        r"пользовател\w+\s+с\s+различн\w+\s+рол|"
        r"\brbac\b|access\s*control",
        re.IGNORECASE | re.UNICODE,
    )),
    ("figma_design", re.compile(
        r"\bfigma\b|макет\w*\s+(?:в|разработ\w+\s+в|интерфейс\w*\s+в)\s+figma|"
        r"прототип\w*\s+в\s+figma",
        re.IGNORECASE | re.UNICODE,
    )),
    ("browser_ui", re.compile(
        r"в\s+браузер|интерактивн\w*\s+(?:пользоват|интерфейс)|"
        r"chrome|edge|firefox|safari|yandex|opera|"
        r"запуска\w*\s+в\s+браузер",
        re.IGNORECASE | re.UNICODE,
    )),
    ("data_from_server", re.compile(
        # Allow optional `,` and whitespace between «данные» and the
        # qualifier («полученные» / «с сервера» / «от сервера») —
        # «данные, полученные с сервера» is the canonical form.
        r"данн\w+[,\s]+(?:полученн|с\s+сервер|от\s+сервер)|"
        r"ответ\w*\s+(?:сервер|серверн\w+\s+част)|"
        r"данн\w+[,\s]+полученн\w+\s+с\s+сервер",
        re.IGNORECASE | re.UNICODE,
    )),
    ("invalid_request_handling", re.compile(
        r"неверн\w*\s+запрос|некорректн\w*\s+(?:ввод|запрос|данн)|"
        r"обработк\w+\s+(?:неверн|некорректн|ошибочн)\w*\s+(?:запрос|ввод)|"
        r"информативн\w*\s+сообщен\w*\s+об\s+ошибк",
        re.IGNORECASE | re.UNICODE,
    )),
)


def _classify_topics(text: str) -> set[str]:
    """Return the set of topics the text matches. Empty set when none
    of the topic patterns fire — safe default (no mismatch can fire
    against an unclassified text)."""
    if not text:
        return set()
    out: set[str] = set()
    for topic, pat in _TOPIC_PATTERNS:
        if pat.search(text):
            out.add(topic)
    return out


# Pairs (req_topic, unit_topic) where mixing is a topical mismatch:
# req IS about the first topic, evidence IS about the second, and the
# two are not interchangeable. Order matters — only the listed
# direction triggers the demotion.
_TOPIC_MISMATCH_PAIRS: frozenset[tuple[str, str]] = frozenset({
    # Performance vs reliability/hardware confusion (Polyakov 0.14).
    ("response_time", "recovery_time"),
    ("response_time", "hardware_specs"),
    # Security: injection ≠ RBAC (Polyakov 0.18::sent2).
    ("code_injection", "access_control"),
    # Design tooling ≠ delivered UI (Polyakov 0.15::sent2).
    ("figma_design", "browser_ui"),
    # Different I/O directions / failure modes (Polyakov 0.17::sent4).
    ("data_from_server", "invalid_request_handling"),
})


def _topic_mismatch_reason(
    req_topics: set[str], unit_topics: set[str],
) -> Optional[str]:
    """Return a human-readable mismatch reason when the (req, unit)
    pair hits one of `_TOPIC_MISMATCH_PAIRS` AND the unit does NOT
    also carry the requirement's topic. Otherwise None.

    The "AND not overlap" guard means: an evidence paragraph that
    discusses BOTH the right topic and the wrong topic stays — it's
    multi-aspect coverage, not a mismatch. Only pure off-topic
    evidence triggers the demote.
    """
    if not req_topics or not unit_topics:
        return None
    # Overlap → topics agree on at least one — never a mismatch.
    if req_topics & unit_topics:
        return None
    for r in req_topics:
        for u in unit_topics:
            if (r, u) in _TOPIC_MISMATCH_PAIRS:
                return f"req topic {r!r} ≠ evidence topic {u!r}"
    return None


def _append_action(judgment: PairJudgment, action: str) -> None:
    """Append a verifier action tag onto the judgment.

    Conventions (so the aggregator can recognise rule outcomes):
      * `conflict_*`   — verifier explicitly confirmed a CONFLICT
                         (numeric, negation, …). Aggregator promotes
                         CONFLICT to a verified verdict.
      * `demote_*`     — a positive label was demoted (CONFLICT → PARTIAL,
                         COVERED → PARTIAL, …).
      * `suppress_*`   — a rule was about to fire but a guard suppressed
                         it (false-positive guard).
      * `no_op_*`      — verifier ran but no rule applied; label preserved.
    """
    actions = list(getattr(judgment, "verifier_actions", []) or [])
    actions.append(action)
    judgment.verifier_actions = actions


class PairVerifier:
    """Applies rule-based adjustments to a PairJudgment.

    P0 #7 (Калугин) — two contract changes that close the false-CONFLICT
    class around semantic same-outcome phrasing:

      1. Optional `same_outcome_sim_fn` (and `same_outcome_sim_threshold`,
         default 0.65). When provided, the similarity between the
         requirement and the unit is computed BEFORE the polarity rule;
         pairs at or above the threshold are treated as same-outcome and
         the negation contradiction is suppressed. Falls through to the
         pattern table (`_SAME_OUTCOME_PAIRS`) when no embedder is
         injected so unit tests and offline pipelines stay deterministic.
      2. Hard rule: the negation rule may CONFIRM an LLM-CONFLICT verdict
         (CONFLICT remains CONFLICT, with provenance) but may NEVER
         upgrade a PARTIAL/COVERED/IRRELEVANT label to CONFLICT. Any
         residual prohibition mismatch becomes a warning on the existing
         label, not a label change.
    """

    def __init__(
        self,
        same_outcome_sim_fn: Optional[Callable[[str, str], float]] = None,
        same_outcome_sim_threshold: float = 0.65,
    ) -> None:
        self._same_outcome_sim_fn = same_outcome_sim_fn
        self._same_outcome_sim_threshold = same_outcome_sim_threshold

    def _embedding_says_same_outcome(self, req_text: str, unit_text: str) -> bool:
        fn = self._same_outcome_sim_fn
        if fn is None:
            return False
        try:
            score = float(fn(req_text or "", unit_text or ""))
        except Exception:  # embedder must never break the verifier
            logger.warning(
                "same_outcome_sim_fn raised; falling back to pattern table",
                exc_info=True,
            )
            return False
        return score >= self._same_outcome_sim_threshold

    def verify(
        self,
        judgment: PairJudgment,
        req: RequirementUnit,
        unit: CoverageUnit,
    ) -> PairJudgment:
        # PR-G refactor: type-aware suppression of CONFLICT for
        # requirement classes where coverage isn't the question.
        # DELIVERY / PROCESS / ECONOMIC requirements never produce a
        # CONFLICT row — the aggregator's applicability filter handles
        # them downstream. Runs before the IRRELEVANT shortcut so that
        # the no_op_type_excluded action is always recorded for these
        # type classes.
        if not _types_can_conflict(req, unit):
            if judgment.llm_label == LLMLabel.CONFLICT:
                # Demote to PARTIAL so any matched aspects are still
                # surfaced; aggregator will mark the row OUT_OF_SCOPE.
                judgment.rule_adjusted_label = LLMLabel.PARTIAL
                judgment.explanation += (
                    " [rule] Type is delivery/process/economic — coverage "
                    "CONFLICT is not meaningful for this requirement class."
                )
                _append_action(judgment, "demote_conflict_type_excluded")
                return judgment
            judgment.rule_adjusted_label = judgment.llm_label
            _append_action(judgment, "no_op_type_excluded")
            return judgment

        # PR-K P0 fix: when the LLM said IRRELEVANT, we still let the
        # deterministic numeric-conflict check run below — small local
        # models (qwen2.5:3b in smoke tests) sometimes label "журнал 90
        # дней" vs "журнал 30 суток" as IRRELEVANT, missing the obvious
        # numeric mismatch. The numeric-rule has its own strict
        # topical-link guard (shared constraint kind / entity overlap /
        # ≥2 shared tokens / LLM confidence) so we don't false-positive
        # on truly unrelated pairs that happen to share a number.
        # Negation / COVERED-demotion rules still skip on IRRELEVANT
        # below, because their guards are weaker than the numeric one.
        is_irrelevant = judgment.llm_label == LLMLabel.IRRELEVANT

        # PR-G refactor: same-outcome negation compatibility — phrases
        # like "не должна аварийно завершаться" vs "должна продолжать
        # корректно функционировать" are semantically equivalent, not
        # contradictory. Suppress the LLM's CONFLICT verdict here BEFORE
        # the rule-based negation rule below considers them.
        if (
            judgment.llm_label == LLMLabel.CONFLICT
            and (
                self._embedding_says_same_outcome(req.text, unit.text)
                or _same_outcome_negation_compatible(req.text, unit.text, check_upper_bound=False)
                or _same_upper_bound_same_aspect(req, unit)
            )
        ):
            # Audit (Polyakov 0.17::sent3): "Система не должна аварийно
            # завершать свою работу" vs "система должна продолжать корректно
            # функционировать" describe the SAME outcome — the requirement is
            # satisfied. Old behaviour demoted the LLM CONFLICT to PARTIAL,
            # so the row stayed in the "warnings" pile. The semantics are
            # equivalent → upgrade to COVERED so the row is reported as such.
            # We keep `verifier_actions` provenance ("upgrade_..._same_outcome")
            # so the trace and aggregator can see the override.
            judgment.rule_adjusted_label = LLMLabel.COVERED
            judgment.explanation += (
                " [rule] Same-outcome negation compatibility detected — "
                "prohibition of bad outcome and affirmation of good outcome "
                "are equivalent; upgraded CONFLICT → COVERED."
            )
            _append_action(judgment, "upgrade_conflict_same_outcome_covered")
            return judgment

        # ── Audit (Annenkov package): PMI-copy-without-methodology ──────
        #
        # Symptom: when ПМИ just copies the requirement text from ТЗ
        # verbatim (no test procedure, no expected result, no acceptance
        # criterion), small-model judges (qwen-3b) ставят COVERED on
        # exact-text match. Larger models (Llama-70b) correctly say
        # MISSING because PMI is supposed to describe HOW to verify,
        # not duplicate the requirement.
        #
        # Deterministic fix: if the LLM said COVERED on a PMI target
        # AND the evidence text is essentially a copy of the requirement
        # (high lex_jac AND no methodology vocabulary), demote to PARTIAL.
        # This makes small / large models agree on the right outcome.
        #
        # Guard `lex_jac >= 0.80` is intentionally strict: real PMI text
        # paraphrases (lex_jac 0.30-0.60) — only near-verbatim duplication
        # of the TZ requirement triggers the rule. Calibrated against
        # tests/test_coverage_pipeline.py::TestNearVerbatimRequirementMatch
        # where "Обеспечивать разграничение прав..." vs "Система должна
        # обеспечивать разграничение прав... по ролям" has lex_jac=0.56
        # (legitimate COVERED, must not be demoted). Annenkov-style
        # verbatim copies sit at lex_jac≈1.00 and trigger correctly.
        if (
            judgment.llm_label == LLMLabel.COVERED
            and (unit.target_doc_role or "").strip().lower() == "pmi"
            and not _pmi_has_methodology(unit.text)
        ):
            from app.core.text import tokenize_content

            req_toks = tokenize_content(req.normalized_text)
            unit_toks = tokenize_content(unit.normalized_text)
            lex_jac = (
                len(req_toks & unit_toks) / len(req_toks | unit_toks)
                if req_toks and unit_toks
                else 0.0
            )
            if lex_jac >= 0.80:
                judgment.rule_adjusted_label = LLMLabel.PARTIAL
                missing = list(judgment.missing_aspects) + [
                    "методика проверки требования не описана в фрагменте",
                ]
                judgment.missing_aspects = missing
                judgment.explanation += (
                    f" [rule] PMI fragment is a near-verbatim copy of the "
                    f"requirement (lex_jac={lex_jac:.2f}) without methodology "
                    f"vocabulary (тест/проверяется/проверка/критерий/etc.); "
                    f"demoted COVERED → PARTIAL."
                )
                _append_action(judgment, "demote_covered_pmi_no_methodology")
                return judgment

        conflict_details: List[str] = list(judgment.conflict_aspects)

        # Rule 1: numeric constraint conflict.
        #
        # Guard: require some topical-relatedness signal before we claim a
        # conflict. A req and a unit that happen to share `3 сек` as a
        # number, with zero entity overlap and almost no lexical overlap,
        # are simply two unrelated sentences — not a contradiction. Raising
        # CONFLICT on them was the biggest single source of false positives
        # in manual review (e.g. pkg_0005 #7, pkg_0008 #7).
        numeric_conflicts = _find_numeric_conflict(req.constraints, unit.constraints)
        if numeric_conflicts:
            from app.core.text import tokenize_content

            ent_overlap = _entity_overlap(req, unit)
            req_tokens = tokenize_content(req.normalized_text)
            unit_tokens = tokenize_content(unit.normalized_text)
            shared_tokens = req_tokens & unit_tokens
            # If both sides have a constraint with the same declared kind
            # (e.g. both `retention_period`), we already know they are
            # talking about the same aspect — that IS the topic link.
            shared_constraint_kinds = (
                {c.kind for c in req.constraints if c.kind and c.kind != "generic"}
                & {c.kind for c in unit.constraints if c.kind and c.kind != "generic"}
            )
            # Topical signal: same constraint kind OR some entity overlap
            # OR enough shared content tokens OR LLM confidence.
            has_topic_link = (
                bool(shared_constraint_kinds)
                or ent_overlap >= 0.15
                or len(shared_tokens) >= 2
                or (judgment.llm_confidence or 0) >= 0.4
            )

            # v3-review regression: the BGE judge confidently says COVERED on
            # pairs like "ОС Windows 7/10/11" vs "Windows 10" (10 is in the
            # allowed set). The numeric_conflicts rule then blindly fires
            # CONFLICT because 7 ≠ 10. The judge already aggregated the
            # semantic picture — if it is confidently COVERED, trust it and
            # suppress the rule override. 4/4 false CONFLICTs in manual
            # review came from this path.
            #
            # PR-K: bump the threshold from 0.70 to 0.80 so that the
            # DisabledCoverageJudge's structural ck_match path (conf=0.70,
            # which only confirms same constraint kind, NOT same value)
            # cannot suppress a real numeric conflict between 30 and 90.
            # Real LLMs aggregating semantic picture should comfortably
            # report ≥0.80 on a confident COVERED.
            judge_strongly_says_covered = (
                judgment.llm_label == LLMLabel.COVERED
                and (judgment.llm_confidence or 0) >= 0.80
            )
            if judge_strongly_says_covered:
                judgment.explanation += (
                    f" [rule] Numeric value mismatch suppressed — judge is "
                    f"confidently COVERED (conf={judgment.llm_confidence:.2f}): "
                    f"{'; '.join(numeric_conflicts)}"
                )
                logger.debug(
                    "Rule: numeric mismatch SUPPRESSED (judge confident COVERED) "
                    "for req=%s unit=%s",
                    req.req_id[:8], unit.unit_id[:8],
                )
                _append_action(judgment, "suppress_numeric_judge_strong_covered")
                # fall through to subsequent rules — skip the CONFLICT branch
                has_topic_link = False
            if has_topic_link:
                conflict_details.extend(numeric_conflicts)
                judgment.rule_adjusted_label = LLMLabel.CONFLICT
                judgment.conflict_aspects = conflict_details
                judgment.explanation += (
                    f" [rule] Numeric conflict (ent_ov={ent_overlap:.2f}, "
                    f"shared_tokens={len(shared_tokens)}): {'; '.join(numeric_conflicts)}"
                )
                logger.debug(
                    "Rule: numeric conflict for req=%s unit=%s (ent=%.2f, shared=%d)",
                    req.req_id[:8], unit.unit_id[:8], ent_overlap, len(shared_tokens),
                )
                _append_action(judgment, "conflict_confirmed_numeric")
                # PR-K P0 fix: when verifier deterministically promotes
                # IRRELEVANT/PARTIAL/COVERED → CONFLICT, the original
                # llm_confidence reflects the LLM's confidence in a
                # DIFFERENT label and is no longer meaningful for the
                # aggregator's CONFLICT-confidence gate. Bump to 0.95 so
                # the rule-confirmed CONFLICT survives the gate. The
                # `verifier_actions` tag preserves provenance — the
                # aggregator and the trace can still see this came from
                # a deterministic rule, not from a confident LLM verdict.
                if is_irrelevant or judgment.llm_confidence < 0.85:
                    judgment.llm_confidence = 0.95
                return judgment
            else:
                judgment.explanation += (
                    f" [rule] Numeric value mismatch suppressed — no topical link "
                    f"(ent_ov={ent_overlap:.2f}, shared_tokens={len(shared_tokens)}, "
                    f"conf={judgment.llm_confidence:.2f}): {'; '.join(numeric_conflicts)}"
                )
                logger.debug(
                    "Rule: numeric mismatch SUPPRESSED for req=%s unit=%s (no topic)",
                    req.req_id[:8], unit.unit_id[:8],
                )
                _append_action(judgment, "suppress_numeric_no_topic")

        # PR-K P0 fix: at this point the numeric rule has had its chance.
        # If the LLM said IRRELEVANT and no numeric conflict promoted us to
        # CONFLICT, return IRRELEVANT now. The remaining rules (negation,
        # COVERED→PARTIAL demotions) have weaker topical guards and would
        # produce false-positive CONFLICTs on truly unrelated pairs whose
        # vocabularies happen to share a prohibition word.
        if is_irrelevant:
            judgment.rule_adjusted_label = LLMLabel.IRRELEVANT
            _append_action(judgment, "no_op_irrelevant")
            return judgment

        # Rule 2: negation/modality contradiction
        # Guard: only fire when llm_confidence >= 0.25.
        # At lower confidence the pair shares only superficial vocabulary — req and unit
        # are likely about *different* topics that happen to share a few tokens, so a
        # modality mismatch (one prohibits, the other permits) is not a real conflict.
        if _negation_contradiction(req, unit) and judgment.llm_confidence >= 0.25:
            # P0 #7: embedding-driven same-outcome check runs first.
            # When the optional similarity backend says the two texts
            # describe the same outcome (cos ≥ 0.65), suppress the
            # negation rule entirely — the prohibition phrasing is
            # equivalent to the positive affirmation.
            if self._embedding_says_same_outcome(req.text, unit.text):
                judgment.rule_adjusted_label = judgment.llm_label
                judgment.explanation += (
                    " [rule] Negation contradiction suppressed — embedding "
                    "similarity ≥ threshold indicates same-outcome phrasing."
                )
                _append_action(judgment, "suppress_negation_embedding_same_outcome")
                return judgment
            # PR-G refactor: same-outcome compatibility table catches the
            # pre-classified semantic equivalences ("don't crash" ≡ "keep
            # working"). When recognised, the modality mismatch is fake —
            # demote to PARTIAL instead of raising CONFLICT.
            if _same_outcome_negation_compatible(req.text, unit.text):
                judgment.rule_adjusted_label = LLMLabel.PARTIAL
                judgment.explanation += (
                    " [rule] Negation contradiction suppressed — same-outcome "
                    "phrasing detected (prohibition + positive affirmation "
                    "describe the same outcome)."
                )
                _append_action(judgment, "suppress_negation_same_outcome")
                return judgment

            # PR-K P4: topical-link guard — same logic the numeric rule has,
            # because the extended _PROHIBITION_RE (P2 covered "не должно"
            # neuter form etc.) now matches more pairs and false-fires on
            # unrelated ones. Real-package symptom (Polyakov 0.14::sent1):
            # TZ "время отклика не должно превышать 3 секунд" was paired
            # with PMI evidence "Windows 10 Pro [10] / Intel i5-7500 / 16 GB"
            # because the selector took only top-1 retrieval and that one
            # had highest BoW score by accident. Negation-rule fired since
            # req has prohibition + unit doesn't.
            from app.core.text import tokenize_content

            ent_overlap_neg = _entity_overlap(req, unit)
            req_tokens_neg = tokenize_content(req.normalized_text)
            unit_tokens_neg = tokenize_content(unit.normalized_text)
            shared_tokens_neg = req_tokens_neg & unit_tokens_neg
            # Negation contradiction is a softer signal than numeric
            # mismatch, so the topical-link bar is HIGHER (≥3 shared
            # content tokens vs 2 for numeric, ent_overlap ≥0.20 vs 0.15).
            #
            # PR-K post-fix (F): when the negation rule is CONFIRMING an
            # LLM-native CONFLICT (not upgrading PARTIAL/COVERED), the
            # LLM's confidence is NOT a reliable topical-link proxy —
            # the LLM may have been confidently wrong on off-topic
            # evidence. Real-package symptom: Polyakov run-4/5
            # req 0.17::sent3 vs PZ admin fragment (and vs PMI access-
            # control unit): LLM said CONFLICT conf=0.85 on completely
            # unrelated evidence; the confidence proxy caused the negation
            # rule to confirm → two false CONFLICT rows in every run.
            # For UPGRADES (PARTIAL/COVERED → CONFLICT), LLM confidence
            # IS informative — the LLM saw something relevant and was
            # sure of it; the proxy is appropriate there.
            # Post-fix F: when confirming an LLM-native CONFLICT we strip the
            # LLM confidence proxy (it was circular — high conf on a wrong
            # verdict was treated as topical evidence). For UPGRADES
            # (PARTIAL/COVERED → CONFLICT) the proxy is still allowed because
            # the LLM had genuine reason to be confident about relatedness.
            #
            # Threshold ≥ 2 (was ≥ 3) for shared content tokens: the
            # ≥ 3 figure was calibrated with pymorphy3 lemmatisation where
            # "сохранять"/"сохраняет", "логах"/"логе" collapse to the same
            # lemma. Without pymorphy3 (lowercase-only fallback) genuinely
            # related pairs share 2 exact-form tokens, so ≥ 2 is the
            # correct minimum that keeps off-topic pairs (0 shared) filtered
            # while preserving genuine contradictions.
            if judgment.llm_label == LLMLabel.CONFLICT:
                has_topic_link_neg = (
                    ent_overlap_neg >= 0.20
                    or len(shared_tokens_neg) >= 2
                )
            else:
                has_topic_link_neg = (
                    ent_overlap_neg >= 0.20
                    or len(shared_tokens_neg) >= 2
                    or (judgment.llm_confidence or 0) >= 0.50
                )
            if not has_topic_link_neg:
                judgment.explanation += (
                    f" [rule] Negation contradiction suppressed — no topical "
                    f"link (ent_ov={ent_overlap_neg:.2f}, "
                    f"shared_tokens={len(shared_tokens_neg)}, "
                    f"conf={judgment.llm_confidence:.2f})."
                )
                logger.debug(
                    "Rule: negation contradiction SUPPRESSED for req=%s unit=%s "
                    "(no topic, ent=%.2f, shared=%d)",
                    req.req_id[:8], unit.unit_id[:8],
                    ent_overlap_neg, len(shared_tokens_neg),
                )
                _append_action(judgment, "suppress_negation_no_topic")
                # Post-fix F: an LLM-native CONFLICT that was suppressed
                # because of no topical link is a hallucination — demote to
                # PARTIAL so it does not end up as a contradiction row.
                # For non-CONFLICT labels fall through; they are already
                # non-contradictory and subsequent rules may still adjust.
                if judgment.llm_label == LLMLabel.CONFLICT:
                    # No topical link means the pair is off-topic. An LLM that
                    # hallucinated CONFLICT on unrelated text provides zero
                    # coverage evidence — the pair is effectively IRRELEVANT.
                    # Setting PARTIAL here was the source of false-PARTIAL rows
                    # in Polyakov runs where negation-suppression fired on
                    # admin-fragment evidence completely unrelated to the req.
                    judgment.rule_adjusted_label = LLMLabel.IRRELEVANT
                    judgment.explanation += (
                        " [rule] LLM-CONFLICT demoted to IRRELEVANT — "
                        "negation suppressed with no topical link; "
                        "pair treated as off-topic."
                    )
                    _append_action(judgment, "demote_conflict_negation_no_topic")
                    return judgment
                # fall through for non-CONFLICT labels
            else:
                # PR-K P4: when LLM is confidently positive (PARTIAL/COVERED
                # with conf ≥0.70), the verifier should NOT override its
                # verdict via negation-rule. Mirrors the
                # `judge_strongly_says_covered` guard on the numeric rule.
                # Real-package symptom (Polyakov 0.20::sent1): LLM said
                # COVERED conf≈1.0 on "не должно превышать общее время"
                # vs "не должно превышать времени" — exact same prohibition
                # phrasing on both sides — but the negation rule fired
                # anyway and demoted to false-CONFLICT.
                judge_strongly_positive = (
                    judgment.llm_label in (LLMLabel.COVERED, LLMLabel.PARTIAL)
                    and (judgment.llm_confidence or 0) >= 0.70
                )
                if judge_strongly_positive:
                    judgment.explanation += (
                        f" [rule] Negation contradiction suppressed — judge "
                        f"is confidently {judgment.llm_label.value} "
                        f"(conf={judgment.llm_confidence:.2f})."
                    )
                    logger.debug(
                        "Rule: negation contradiction SUPPRESSED for req=%s "
                        "unit=%s (judge confident %s, conf=%.2f)",
                        req.req_id[:8], unit.unit_id[:8],
                        judgment.llm_label.value, judgment.llm_confidence,
                    )
                    _append_action(judgment, "suppress_negation_judge_positive")
                    # fall through — leave rule_adjusted_label as LLM said
                elif judgment.llm_label == LLMLabel.CONFLICT:
                    # P0 #7 hard rule: the negation rule may CONFIRM an
                    # LLM-CONFLICT verdict but must NEVER upgrade
                    # PARTIAL/COVERED → CONFLICT. The pattern set is
                    # narrow enough that confirming an LLM that already
                    # decided CONFLICT is high-precision; promoting from
                    # PARTIAL on the same signal was the dominant source
                    # of false-CONFLICTs in Калугин-class packages.
                    judgment.rule_adjusted_label = LLMLabel.CONFLICT
                    msg = "[rule] Negation contradiction between requirement and coverage unit"
                    judgment.conflict_aspects = conflict_details + [msg]
                    judgment.explanation += f" {msg}"
                    _append_action(judgment, "conflict_confirmed_negation")
                    # PR-K P0: same confidence-bump as the numeric path.
                    if judgment.llm_confidence < 0.85:
                        judgment.llm_confidence = 0.95
                    return judgment
                else:
                    # P0 #7: prohibition mismatch on a non-CONFLICT LLM
                    # label. We surface it as a warning on the existing
                    # label (so the trace explains why the pair looked
                    # off) but we do not change the verdict — the
                    # canonical false-CONFLICT mode in Калугин-class
                    # packages was «не должен X» on TZ paired with «должен
                    # антоним(X)» on PMI/PZ which are semantically
                    # equivalent; LLM rightly returned PARTIAL/COVERED
                    # and the verifier had no business overriding that.
                    judgment.rule_adjusted_label = judgment.llm_label
                    judgment.explanation += (
                        " [rule] Prohibition mismatch detected but LLM verdict "
                        "is not CONFLICT — verifier may only confirm CONFLICT, "
                        "never upgrade. Label preserved."
                    )
                    _append_action(judgment, "no_op_negation_no_upgrade")
                    return judgment

        # Rule 3: COVERED but req has constraints and unit has none → PARTIAL
        if (
            judgment.llm_label == LLMLabel.COVERED
            and req.constraints
            and not unit.constraints
        ):
            judgment.rule_adjusted_label = LLMLabel.PARTIAL
            judgment.missing_aspects = list(judgment.missing_aspects) + [
                f"Numeric constraint not verified: {c.value} {c.unit or ''}"
                for c in req.constraints
            ]
            judgment.explanation += " [rule] Required numeric constraints absent in coverage unit → PARTIAL"
            _append_action(judgment, "demote_covered_constraints_missing")
            return judgment

        # Rule 4: COVERED but very low entity overlap when both have many entities → PARTIAL
        # Only applies when both sides have 3+ entities (single-entity docs produce noisy overlap)
        if (
            judgment.llm_label == LLMLabel.COVERED
            and len(req.entities) >= 3
            and len(unit.entities) >= 3
        ):
            overlap = _entity_overlap(req, unit)
            if overlap < 0.1:
                judgment.rule_adjusted_label = LLMLabel.PARTIAL
                judgment.explanation += f" [rule] Low entity overlap ({overlap:.2f}) with many entities → PARTIAL"
                _append_action(judgment, "demote_covered_low_entity_overlap")
                return judgment

        # Rule 4-bis (Step 8): stricter COVERED bar for PMI.
        #
        # ПМИ — это методика проверки. Чтобы строка из ПМИ покрывала
        # требование, она ОБЯЗАНА явно ссылаться на требование —
        # либо повторяя его ключевые сущности, либо цитируя его
        # формулировку (≥30% лексическое пересечение). PZ, напротив,
        # может покрывать требование «обоснованием решения» с гораздо
        # слабым vocabulary overlap (нарративные параграфы используют
        # implementation-лексику, а ТЗ — specification-лексику).
        #
        # Rule 4 (generic, выше) срабатывает только при ≥3 сущностях с
        # обеих сторон и порогe 0.10. Этот гораздо более строгий гейт
        # применяется ТОЛЬКО к ПМИ и ловит случаи, когда LLM ставит
        # COVERED на ПМИ-фрагменте, который структурно не привязан к
        # требованию: ни сущностного, ни лексического сцепления.
        # Симптом из Polyakov May-11: ПМИ test-step «Проверить вход
        # в систему под учётной записью admin» получает COVERED на
        # требовании «Журнал событий должен храниться 90 дней» —
        # entity_overlap=0, lex_jac=0.07.
        if (
            judgment.rule_adjusted_label == LLMLabel.COVERED
            and (unit.target_doc_role or "").strip().lower() == "pmi"
        ):
            from app.core.text import tokenize_content

            req_toks_p = tokenize_content(req.normalized_text)
            unit_toks_p = tokenize_content(unit.normalized_text)
            lex_jac_pmi = (
                len(req_toks_p & unit_toks_p) / len(req_toks_p | unit_toks_p)
                if req_toks_p and unit_toks_p else 0.0
            )
            ent_ov_pmi = _entity_overlap(req, unit)
            # Both signals weak → no verifiable link to the requirement.
            # Thresholds calibrated against the Annenkov-style verbatim
            # rule above (lex_jac ≥ 0.80 there): genuine PMI test steps
            # paraphrasing a requirement sit at lex_jac 0.30–0.60 and
            # ent_overlap ≥ 0.30; below both bars at once is structurally
            # unrelated.
            if lex_jac_pmi < 0.20 and ent_ov_pmi < 0.25:
                judgment.rule_adjusted_label = LLMLabel.PARTIAL
                judgment.missing_aspects = list(judgment.missing_aspects) + [
                    "ПМИ-фрагмент не ссылается на требование (нет общих сущностей и нет лексического пересечения)",
                ]
                judgment.explanation += (
                    f" [rule] PMI COVERED demoted — weak verbatim link "
                    f"(lex_jac={lex_jac_pmi:.2f}, ent_ov={ent_ov_pmi:.2f}); "
                    f"ПМИ должна явно цитировать или перечислять сущности требования."
                )
                _append_action(judgment, "demote_covered_pmi_weak_link")
                # Intentional fall-through (no `return`): the Step-7
                # aspect-mismatch demoter at the end of verify() still
                # needs a chance to escalate PARTIAL → IRRELEVANT when
                # the topics are genuinely off — its guard looks at
                # `llm_label` (still COVERED here) so it will fire.
                # Returning early here masked off-topic pairs as PARTIAL
                # instead of the correct IRRELEVANT.

        # Rule 5: LLM-PARTIAL with near-zero entity overlap → IRRELEVANT.
        #
        # When the LLM assigns PARTIAL but the requirement and the unit share
        # essentially no named entities, the partial-coverage verdict is almost
        # certainly driven by incidental vocabulary overlap rather than genuine
        # topical alignment. The canonical real-package failure mode is
        # competitive-analysis PZ text: the requirement asks for specific
        # technical features (TypeScript, Angular, Figma, REST API) while the
        # retrieved unit is a competitor-description or analysis section that
        # happens to share generic domain vocabulary — zero entity overlap, yet
        # the LLM calls it PARTIAL.
        #
        # Guards:
        #   * Requires ≥ 2 extracted entities on EACH side. Fewer entities
        #     likely means extraction failed — we must not penalise the pair
        #     on sparse entity data.
        #   * Confidence guard (< 0.85): extremely confident LLM verdicts are
        #     trusted over the entity signal; in practice genuine topically-
        #     related PARTIALs have conf ≥ 0.85 when entity lists are rich.
        #
        # Note: only fires on LLM-native PARTIAL labels. Verifier-demoted
        # PARTIAL (same-outcome CONFLICT, Rules 3/4) returns early and never
        # reaches this point.
        # Polyakov-regression (2026-05-10, R1+ tightening): demoter
        # gate now requires `len(req.entities) >= 3` AND
        # `len(unit.entities) == 0`. Per user spec: "повысить порог
        # entity-демотера до «есть ≥ 3 entities в требовании, и 0 в
        # evidence»". The previous `unit.entities >= 2` threshold still
        # produced false-MISSING on Polyakov 0.17::sent1 / 0.18::sent1
        # (req+evidence both had 2-3 entities, entity_overlap=0 by
        # extractor misalignment, lex_jac low — judge PARTIAL conf 0.7
        # got demoted). Genuinely off-topic evidence (the canonical
        # competitive-analysis blob) typically has ZERO extracted
        # entities relative to the technical req — that's the only
        # case where entity_overlap=0 is diagnostic, not a sign of
        # extractor noise. Conservative narrowing — only fires now
        # when evidence is genuinely empty of named things.
        if (
            judgment.llm_label == LLMLabel.PARTIAL
            and len(req.entities) >= 3
            and len(unit.entities) == 0
            and (judgment.llm_confidence or 0) < 0.85
        ):
            overlap = _entity_overlap(req, unit)
            if overlap < 0.05:
                # Audit (Polyakov 0.41::sent4): the entity extractor missed
                # the "перечень функций / методы испытаний / технические средства"
                # nominal phrases in the req side, so entity_overlap = 0 even
                # though the texts paraphrase the same content. Adding a lexical
                # jaccard floor catches this: real off-topic pairs (competitive
                # analysis vs functional req) score lex_jac < 0.10; honest
                # paraphrases land at 0.15-0.30. Demote only when BOTH signals
                # are low.
                from app.core.text import tokenize_content

                req_toks = tokenize_content(req.normalized_text)
                unit_toks = tokenize_content(unit.normalized_text)
                lex_jac = (
                    len(req_toks & unit_toks) / len(req_toks | unit_toks)
                    if req_toks and unit_toks
                    else 0.0
                )
                if lex_jac >= 0.20:
                    judgment.rule_adjusted_label = judgment.llm_label
                    judgment.explanation += (
                        f" [rule] Near-zero entity overlap ({overlap:.2f}) on "
                        f"LLM-PARTIAL verdict — preserved (lex_jac={lex_jac:.2f} "
                        f"≥ 0.20 indicates lexical paraphrase, not off-topic)."
                    )
                    _append_action(judgment, "preserve_partial_low_entity_high_lex")
                    return judgment

                # Soft anchor: if req and unit share at least one
                # substantive content token (≥5 chars, not a stopword)
                # AND the LLM was at least mildly confident (≥0.6),
                # preserve PARTIAL. This catches genuine same-domain
                # partial coverage where the entity extractor missed
                # the head noun on one side. Real-package symptom
                # (Polyakov 0.11::sent11): TZ «Комплексная система
                # фильтрации поиска по репозиторию. (Авторы, темы,
                # ключевые слова, дата…)» vs PMI «Система должна
                # обеспечивать поиск по публикациям и проектам.» —
                # both share the noun «поиск», LLM said PARTIAL conf
                # 0.7. Aggressive demotion forced MISSING; the user
                # correctly reads this as PARTIAL (search is covered,
                # specific filters aren't).
                substantive_shared = {
                    t for t in (req_toks & unit_toks)
                    if len(t) >= 5 and t not in _CONTENT_TOKEN_STOPWORDS
                }
                if substantive_shared and (judgment.llm_confidence or 0) >= 0.60:
                    judgment.rule_adjusted_label = judgment.llm_label
                    judgment.explanation += (
                        f" [rule] Near-zero entity overlap ({overlap:.2f}) on "
                        f"LLM-PARTIAL verdict — preserved (shared substantive "
                        f"token(s): {sorted(substantive_shared)[:3]}; "
                        f"conf={judgment.llm_confidence:.2f})."
                    )
                    _append_action(judgment, "preserve_partial_shared_anchor")
                    return judgment

                # Polyakov-regression (2026-05-10): tech-stack
                # co-occurrence preservation. When the requirement and
                # the evidence both mention identifiers from the same
                # technology stack family (TypeScript/Angular/REST/
                # JSON/Docker/Git/SPA/DSpace/Figma/HTML/CSS/SCSS/
                # extensions like .ts/.html/.css), the topic alignment
                # is genuine even when the entity extractor missed the
                # phrasing (extension-form vs full-name, English-vs-
                # Russian transliteration, etc.). Real-package failure
                # mode (Polyakov 0.27::sent1): TZ «Исходные коды
                # программы должны быть написаны на TypeScript с
                # использованием библиотеки Angular» vs PZ «.ts, .html,
                # .css, .scss» — both clearly about the same client-
                # side stack, but neither lex-jaccard nor entity overlap
                # picks it up because of the syntactic gap. Preserve
                # PARTIAL when there is any match in the same family.
                tech_match = _tech_stack_co_occurrence(
                    req.normalized_text, unit.normalized_text,
                )
                if tech_match and (judgment.llm_confidence or 0) >= 0.60:
                    judgment.rule_adjusted_label = judgment.llm_label
                    judgment.explanation += (
                        f" [rule] Near-zero entity overlap ({overlap:.2f}) on "
                        f"LLM-PARTIAL verdict — preserved (tech-stack "
                        f"co-occurrence: {tech_match}; "
                        f"conf={judgment.llm_confidence:.2f})."
                    )
                    _append_action(judgment, "preserve_partial_tech_stack")
                    return judgment

                judgment.rule_adjusted_label = LLMLabel.IRRELEVANT
                judgment.explanation += (
                    f" [rule] Near-zero entity overlap ({overlap:.2f}) and low "
                    f"lex_jac ({lex_jac:.2f}) on LLM-PARTIAL verdict — evidence "
                    f"likely off-topic; demoted to IRRELEVANT."
                )
                logger.debug(
                    "Rule 5: PARTIAL demoted to IRRELEVANT for req=%s unit=%s "
                    "(entity_overlap=%.2f, lex_jac=%.2f, conf=%.2f)",
                    req.req_id[:8], unit.unit_id[:8],
                    overlap, lex_jac, (judgment.llm_confidence or 0),
                )
                _append_action(judgment, "demote_partial_zero_entity_overlap")
                return judgment

        # Polyakov-regression Step 7 (2026-05-11): aspect-mismatch
        # guard. LAST gate before the no-op exit. The LLM judge
        # frequently raises PARTIAL/COVERED on pairs sharing only
        # generic vocabulary («система», «интерфейс», «ошибка») when
        # the actual semantic topics don't match. Real-package examples:
        #   * 0.14 «время отклика 3 сек» vs hardware/recovery time;
        #   * 0.18::sent2 «внедрение кода» vs RBAC/auth;
        #   * 0.15::sent2 «макет в Figma» vs «интерфейс в браузере»;
        #   * 0.17::sent4 «данные с сервера» vs invalid-request handling.
        # Demote to IRRELEVANT when topics are on the curated mismatch
        # list AND req+unit have NO overlapping topic.
        if judgment.llm_label in {LLMLabel.PARTIAL, LLMLabel.COVERED}:
            req_topics = _classify_topics(req.text or req.normalized_text or "")
            unit_topics = _classify_topics(unit.text or unit.normalized_text or "")
            mismatch_reason = _topic_mismatch_reason(req_topics, unit_topics)
            if mismatch_reason:
                judgment.rule_adjusted_label = LLMLabel.IRRELEVANT
                judgment.explanation += (
                    f" [rule] Topic mismatch demote: {mismatch_reason}. "
                    f"Req topics={sorted(req_topics)!r}, "
                    f"unit topics={sorted(unit_topics)!r}; no overlap → "
                    f"evidence is off-topic for the requirement."
                )
                _append_action(
                    judgment,
                    f"demote_topic_mismatch_{mismatch_reason.split(' ')[2].strip(chr(39))}",
                )
                logger.debug(
                    "Rule 7: %s → IRRELEVANT for req=%s unit=%s (%s)",
                    judgment.llm_label.name, req.req_id[:8], unit.unit_id[:8],
                    mismatch_reason,
                )
                return judgment

        # No adjustment needed — carry the LLM label forward.
        # PR-K: when an LLM-flagged CONFLICT survives all guards without
        # the verifier finding a concrete numeric/negation/aspect
        # contradiction, tag it explicitly so the aggregator can decide
        # whether to trust an unverified CONFLICT (current default: no —
        # downgraded to PARTIAL by the aggregator's verifier-confirmation
        # gate).
        judgment.rule_adjusted_label = judgment.llm_label
        if judgment.llm_label == LLMLabel.CONFLICT:
            _append_action(judgment, "no_op_llm_conflict_unverified")
        else:
            _append_action(judgment, "no_op_kept_label")
        return judgment
