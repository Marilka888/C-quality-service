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

    def test_0_20_sent1_llm_conflict_same_upper_bound_demoted(self):
        req = _req(
            "Время восстановления после отказа работы системы не должно превышать "
            "общее время, необходимое на перезагрузку составляющих системы.",
            req_type=RequirementType.RELIABILITY,
        )
        unit = _unit(
            "При отказе, который произошел по вине каких-либо внешних факторов "
            "и не является непоправимым, время восстановления не должно превышать "
            "времени, требующегося на перезагрузку операционной системы и запуск "
            "программы."
        )
        j = _judgment(LLMLabel.CONFLICT, conf=0.95)
        out = self.verifier.verify(j, req, unit)

        assert out.rule_adjusted_label == LLMLabel.COVERED, (
            f"LLM-native CONFLICT on same recovery-time upper bound must be "
            f"upgraded to COVERED (semantically equivalent prohibitions); "
            f"got={out.rule_adjusted_label}, actions={out.verifier_actions}"
        )
        assert "upgrade_conflict_same_outcome_covered" in (out.verifier_actions or [])


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

    @pytest.mark.parametrize("req_text, unit_text", [
        # DIFFERENT metrics: "время отклика" ≠ "время восстановления".
        # Both have "не должно превышать" but they are NOT the same constraint.
        (
            "Время отклика приложения не должно превышать 3 секунд.",
            "Время восстановления после отказа не должно превышать 5 минут.",
        ),
        (
            "Время отклика не должно превышать 2 секунд.",
            "Время восстановления не должно превышать общее время на перезагрузку.",
        ),
    ])
    def test_same_upper_bound_disabled_for_pre_check(self, req_text, unit_text):
        """check_upper_bound=False must suppress the structural same_upper_bound
        pattern so it never fires in the LLM-CONFLICT pre-check. Different
        metrics with the same phrasing must NOT be treated as compatible."""
        assert _same_outcome_negation_compatible(
            req_text, unit_text, check_upper_bound=False
        ) is False, (
            f"same_upper_bound must be disabled with check_upper_bound=False; "
            f"req={req_text!r} unit={unit_text!r}"
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

    def test_different_metrics_same_phrasing_no_demote_conflict_same_outcome(self):
        """Regression for Bug Fix 1: LLM says CONFLICT on два разных метрики
        ("время отклика" vs "время восстановления") that both use
        "не должно превышать". The same-outcome pre-check must NOT fire
        (upgrade_conflict_same_outcome_covered); the verifier should
        proceed to numeric/topical checks and NOT mask off-topic CONFLICT
        as COVERED via the same-outcome gate."""
        req = _req(
            "Время отклика приложения не должно превышать 3 секунд.",
            req_type=RequirementType.PERFORMANCE,
            constraints=[Constraint(kind="response_time", operator="<=",
                                    value=3, unit="sec")],
        )
        unit = _unit(
            "Время восстановления после отказа не должно превышать 5 минут."
        )
        j = _judgment(LLMLabel.CONFLICT, conf=0.70)
        out = self.verifier.verify(j, req, unit)
        assert "upgrade_conflict_same_outcome_covered" not in (out.verifier_actions or []), (
            f"same_upper_bound pre-check must not fire for different metrics; "
            f"actions={out.verifier_actions}"
        )

    def test_negation_no_topic_conflict_demoted_to_irrelevant_not_partial(self):
        """Fix A regression: when negation is suppressed due to no topical
        link AND the LLM said CONFLICT, the verifier must set
        rule_adjusted_label = IRRELEVANT, not PARTIAL.

        The old behaviour (→ PARTIAL) caused off-topic evidence to register
        as partial coverage in the aggregator, inflating PARTIAL counts in
        real packages (Polyakov 0.17::sent3 PZ / PMI).
        """
        req = _req(
            "Система не должна аварийно завершать свою работу в случае "
            "возникновения ошибки.",
            req_type=RequirementType.RELIABILITY,
        )
        # Completely unrelated admin-roles fragment — no shared entities,
        # no shared content tokens; negation fires but has no topical link.
        unit = _unit(
            "Администратор – имеет возможность выдавать роли другим "
            "пользователям, создавать новые коллекции и редактировать "
            "их иерархию, удалять уже принятые исследования или "
            "добавлять новые, без необходимости проходить модерацию.",
            role="pz",
        )
        j = _judgment(LLMLabel.CONFLICT, conf=0.85)
        out = self.verifier.verify(j, req, unit)

        assert out.rule_adjusted_label == LLMLabel.IRRELEVANT, (
            f"LLM-CONFLICT with no topical link must be demoted to IRRELEVANT, "
            f"not {out.rule_adjusted_label.value}; actions={out.verifier_actions}"
        )
        assert "demote_conflict_negation_no_topic" in (out.verifier_actions or []), (
            f"Expected demote_conflict_negation_no_topic action; "
            f"got {out.verifier_actions}"
        )


class TestRule5PartialZeroEntityOverlap:
    """Rule 5: LLM-PARTIAL with near-zero entity overlap must be demoted
    to IRRELEVANT. Real-package failure mode (Polyakov PZ false PARTIALs):
    competitive-analysis sections have vocabulary overlap with TZ
    requirements but share no actual named entities."""

    def setup_method(self):
        self.verifier = PairVerifier()

    # ── helpers ──────────────────────────────────────────────────────────

    def _req_with_entities(self, text: str, entities: list,
                           req_type=RequirementType.FUNCTIONAL) -> RequirementUnit:
        from app.application.use_cases.build_requirements import _normalize_text
        return RequirementUnit(
            req_id="r1", source_document_id="doc-tz",
            text=text, normalized_text=_normalize_text(text),
            requirement_type=req_type,
            entities=entities,
        )

    def _unit_with_entities(self, text: str, entities: list,
                             role: str = "pz") -> CoverageUnit:
        from app.application.use_cases.build_requirements import _normalize_text
        return CoverageUnit(
            unit_id="u1", target_document_id=f"doc-{role}",
            target_doc_role=role,
            text=text, normalized_text=_normalize_text(text),
            entities=entities,
        )

    # ── basic rule fires ─────────────────────────────────────────────────

    def test_partial_zero_overlap_demoted_to_irrelevant(self):
        """LLM says PARTIAL, both sides have ≥ 2 entities, no overlap
        → rule must fire and set IRRELEVANT."""
        req = self._req_with_entities(
            "Исходные коды должны быть написаны на TypeScript с Angular.",
            entities=["TypeScript", "Angular"],
            req_type=RequirementType.ARCHITECTURE_IMPLEMENTATION,
        )
        unit = self._unit_with_entities(
            "Сравнительный анализ: разрабатываемая система должна "
            "полноценно функционировать как репозиторий DSpace.",
            entities=["DSpace", "репозиторий"],
        )
        j = _judgment(LLMLabel.PARTIAL, conf=0.70)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.IRRELEVANT, (
            f"PARTIAL with zero entity overlap must become IRRELEVANT; "
            f"actions={out.verifier_actions}"
        )
        assert "demote_partial_zero_entity_overlap" in (out.verifier_actions or [])

    def test_partial_zero_overlap_figma_vs_competitor(self):
        """Regression: 0.15::sent2 PZ — 'макет разработан в Figma' vs
        competitor description with no Figma reference."""
        req = self._req_with_entities(
            "Макет интерфейса должен быть разработан в Figma.",
            entities=["Figma", "макет интерфейса"],
            req_type=RequirementType.INTERFACE,
        )
        unit = self._unit_with_entities(
            "Repo.hse.ru является текущим решением ВШЭ. "
            "Пользовательский интерфейс реализован в фирменном стиле НИУ ВШЭ.",
            entities=["Repo.hse.ru", "НИУ ВШЭ"],
        )
        j = _judgment(LLMLabel.PARTIAL, conf=0.70)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.IRRELEVANT, (
            f"Figma req vs competitor text must become IRRELEVANT; "
            f"actions={out.verifier_actions}"
        )

    # ── rule does NOT fire when entities overlap ─────────────────────────

    def test_genuine_partial_entity_overlap_kept(self):
        """When entities overlap (genuine PARTIAL), rule must NOT fire.
        Regression: 0.11::sent1 PMI — 'регистрация, авторизация'
        req vs PMI test procedure mentioning the same."""
        req = self._req_with_entities(
            "Система должна предоставить функции: регистрация, авторизация, аутентификация.",
            entities=["регистрация", "авторизация", "аутентификация"],
            req_type=RequirementType.SECURITY,
        )
        unit = self._unit_with_entities(
            "Система должна обеспечивать возможность регистрации и авторизации.",
            entities=["регистрация", "авторизация"],
            role="pmi",
        )
        j = _judgment(LLMLabel.PARTIAL, conf=0.80)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label != LLMLabel.IRRELEVANT, (
            f"Genuine PARTIAL with entity overlap must not be demoted; "
            f"actions={out.verifier_actions}"
        )

    # ── entity-count guard ───────────────────────────────────────────────

    def test_rule5_not_fired_when_req_has_too_few_entities(self):
        """When requirement has < 2 entities, rule must be silent
        (extraction may have failed)."""
        req = self._req_with_entities(
            "Время отклика не должно превышать 3 секунд.",
            entities=["время отклика"],   # only 1 entity
            req_type=RequirementType.PERFORMANCE,
        )
        unit = self._unit_with_entities(
            "Перед разработкой выбирали стек технологий.",
            entities=["стек технологий", "технологии"],
        )
        j = _judgment(LLMLabel.PARTIAL, conf=0.65)
        out = self.verifier.verify(j, req, unit)
        # Rule 5 must not fire — insufficient entity data.
        assert "demote_partial_zero_entity_overlap" not in (out.verifier_actions or []), (
            f"Rule 5 must not fire with < 2 req entities; "
            f"actions={out.verifier_actions}"
        )

    def test_rule5_not_fired_when_unit_has_too_few_entities(self):
        """When unit has < 2 entities, rule must be silent."""
        req = self._req_with_entities(
            "Макет интерфейса должен быть разработан в Figma.",
            entities=["Figma", "макет интерфейса"],
            req_type=RequirementType.INTERFACE,
        )
        unit = self._unit_with_entities(
            "Изменить параметры в конфигурационном файле.",
            entities=["конфигурационный файл"],  # only 1 entity
        )
        j = _judgment(LLMLabel.PARTIAL, conf=0.70)
        out = self.verifier.verify(j, req, unit)
        assert "demote_partial_zero_entity_overlap" not in (out.verifier_actions or []), (
            f"Rule 5 must not fire with < 2 unit entities; "
            f"actions={out.verifier_actions}"
        )

    # ── confidence guard ─────────────────────────────────────────────────

    def test_rule5_not_fired_when_llm_very_confident(self):
        """When conf ≥ 0.85 the verifier trusts the LLM verdict
        even if entity overlap is near zero."""
        req = self._req_with_entities(
            "Макет интерфейса должен быть разработан в Figma.",
            entities=["Figma", "макет интерфейса"],
            req_type=RequirementType.INTERFACE,
        )
        unit = self._unit_with_entities(
            "Сравнительный анализ: DSpace, Repo.hse.ru.",
            entities=["DSpace", "Repo.hse.ru"],
        )
        j = _judgment(LLMLabel.PARTIAL, conf=0.90)   # very confident
        out = self.verifier.verify(j, req, unit)
        assert "demote_partial_zero_entity_overlap" not in (out.verifier_actions or []), (
            f"Rule 5 must not fire at conf=0.90; actions={out.verifier_actions}"
        )

    # ── only fires on LLM-native PARTIAL ─────────────────────────────────

    def test_rule5_not_fired_on_covered_label(self):
        """Rule 5 only fires when llm_label == PARTIAL, not on COVERED."""
        req = self._req_with_entities(
            "Исходный код на TypeScript с Angular.",
            entities=["TypeScript", "Angular"],
            req_type=RequirementType.ARCHITECTURE_IMPLEMENTATION,
        )
        unit = self._unit_with_entities(
            "Сравнительный анализ: DSpace, репозиторий.",
            entities=["DSpace", "репозиторий"],
        )
        j = _judgment(LLMLabel.COVERED, conf=0.75)   # LLM says COVERED, not PARTIAL
        out = self.verifier.verify(j, req, unit)
        assert "demote_partial_zero_entity_overlap" not in (out.verifier_actions or [])

    # ── Polyakov audit fix: lexical-jaccard escape hatch ─────────────────

    def test_rule5_preserves_partial_when_lexical_paraphrase(self):
        """Audit (Polyakov 0.41::sent4): the entity extractor missed
        nominal phrases ("перечень функций", "методы испытаний",
        "технические средства"), so entity_overlap = 0 even though the
        texts share a lot of content vocabulary. The lex_jac guard must
        preserve the LLM-PARTIAL verdict in this case."""
        req = self._req_with_entities(
            "Программа и методика испытаний: перечень функций программы, "
            "перечень требований к функциям, методы испытаний, технические "
            "средства, порядок проведения испытаний.",
            entities=["программа", "методика"],
            req_type=RequirementType.DOCUMENTATION_REQUIREMENT,
        )
        unit = self._unit_with_entities(
            "Порядок проведения испытаний. Испытание проверки выполнения "
            "требований к программной документации. Испытание проверки "
            "выполнения требований к функциональным характеристикам.",
            entities=["испытание", "документация"],
            role="pmi",
        )
        j = _judgment(LLMLabel.PARTIAL, conf=0.80)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.PARTIAL, (
            f"PARTIAL with high lex_jac (paraphrase) must be preserved; "
            f"got {out.rule_adjusted_label}, actions={out.verifier_actions}"
        )
        assert "preserve_partial_low_entity_high_lex" in (out.verifier_actions or [])


class TestQuantifierProhibitionNotNormative:
    """Regression: «не должно превышать» / «не более» / «не позднее» are
    NUMERIC-BOUND quantifiers, not action-banning prohibitions. They must
    not feed `_negation_contradiction`, otherwise any positive-phrasing
    coverage unit (off-topic or not) ends up flipping the verdict to
    CONFLICT just because the requirement carries the word «не должно».
    Numeric-bound conflicts on the same metric are Rule 1's job — that
    rule has its own topical guards.

    Real-package symptom: Polyakov 0.20::sent1 returned a CONFLICT when
    one of the lower-scored evidence units happened to be «Система должна
    корректно обрабатывать неверные запросы…» — a positive-phrasing
    sentence with no «не должн». The negation-rule then false-confirmed.
    """

    def setup_method(self):
        self.verifier = PairVerifier()

    def test_quantifier_prohibition_with_positive_unit_no_conflict(self):
        """TZ «время восстановления не должно превышать общее время на
        перезагрузку» (quantifier) vs PMI «Система должна корректно
        обрабатывать…» (positive, off-topic). Must NOT fire CONFLICT
        through Rule 2 — req's «не должно» is a numeric bound, not an
        action ban."""
        req = _req(
            "Время восстановления после отказа работы системы не должно "
            "превышать общее время, необходимое на перезагрузку "
            "составляющих системы.",
            req_type=RequirementType.RELIABILITY,
        )
        unit = _unit(
            "Система должна корректно обрабатывать неверные запросы любого "
            "вида и выдавать информативные сообщения об ошибках, а также "
            "уведомлять о них пользователя в случае необходимости."
        )
        j = _judgment(LLMLabel.PARTIAL, conf=0.40)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label != LLMLabel.CONFLICT, (
            f"quantifier-only «не должно превышать» + positive-phrasing "
            f"unit produced CONFLICT; actions={out.verifier_actions}"
        )

    def test_action_ban_prohibition_still_fires_with_topic_link(self):
        """Sanity check: a real action-banning prohibition («не должна
        аварийно завершать») paired with a positive unit that talks about
        the same topic should still be evaluated by the negation rule —
        only the quantifier carve-out is new. Whether the rule fires
        CONFLICT or suppresses depends on existing same-outcome /
        topical-link guards; here we just assert the carve-out doesn't
        accidentally exempt action bans."""
        req = _req(
            "Система не должна аварийно завершать свою работу в случае "
            "возникновения ошибки.",
            req_type=RequirementType.RELIABILITY,
        )
        unit = _unit(
            "В случае ошибки при обработке запроса система должна "
            "продолжать корректно функционировать."
        )
        # _negation_contradiction itself must still see this as a
        # modality mismatch — same-outcome compatibility is what
        # demotes / suppresses, not the carve-out.
        from app.application.use_cases.verify_pairs import _negation_contradiction
        assert _negation_contradiction(req, unit) is True, (
            "action-banning prohibition («не должна аварийно завершать») "
            "must still register as prohibitive — only quantifier-only "
            "prohibitions are carved out."
        )


class TestPartialPreservedOnSharedDomainAnchor:
    """Regression: PARTIAL must be preserved when req and unit share at
    least one substantive content noun, even if entity overlap and
    lex-jaccard are individually low. Aggressive demotion produced
    false MISSING for genuine partial coverage where the entity
    extractor missed a head noun.

    Real-package symptom (Polyakov 0.11::sent11): TZ
    «Комплексная система фильтрации поиска по репозиторию. (Авторы,
    темы, ключевые слова, дата и другие в случае необходимости)» vs
    PMI «Система должна обеспечивать поиск по публикациям и проектам.»
    Shared content lemma «поиск» — search-as-feature is covered, the
    specific filters aren't. PARTIAL is the correct verdict.
    """

    def setup_method(self):
        self.verifier = PairVerifier()

    def _req_with_entities(self, text: str, entities: list[str]) -> RequirementUnit:
        from app.application.use_cases.build_requirements import _normalize_text
        return RequirementUnit(
            req_id="r1", source_document_id="doc-tz",
            text=text, normalized_text=_normalize_text(text),
            requirement_type=RequirementType.FUNCTIONAL,
            modality=Modality.MUST,
            constraints=[], entities=entities,
        )

    def _unit_with_entities(self, text: str, entities: list[str], role: str = "pmi") -> CoverageUnit:
        from app.application.use_cases.build_requirements import _normalize_text
        return CoverageUnit(
            unit_id="u1", target_document_id=f"doc-{role}", target_doc_role=role,
            text=text, normalized_text=_normalize_text(text),
            entities=entities,
        )

    def test_partial_with_shared_anchor_noun_preserved(self):
        """Polyakov 0.11::sent11 case: shared lemma «поиск» is a real
        topical anchor, so PARTIAL must survive the demotion rule even
        with low entity overlap and low lex_jac."""
        req = self._req_with_entities(
            "Комплексная система фильтрации поиска по репозиторию. "
            "Авторы, темы, ключевые слова, дата и другие в случае необходимости.",
            entities=["фильтрация", "репозиторий", "автор", "тема"],
        )
        unit = self._unit_with_entities(
            "Система должна обеспечивать поиск по публикациям и проектам.",
            entities=["публикация", "проект"],
        )
        j = _judgment(LLMLabel.PARTIAL, conf=0.70)
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.PARTIAL, (
            f"PARTIAL with shared substantive anchor must be preserved; "
            f"got {out.rule_adjusted_label}, actions={out.verifier_actions}"
        )

    def test_partial_no_anchor_still_demoted(self):
        """Sanity: when there is genuinely no shared substantive content
        token, PARTIAL is still demoted. Off-topic pairs should not
        survive the new soft-anchor rule."""
        req = self._req_with_entities(
            "Время отклика приложения не должно превышать 3 секунд.",
            entities=["время", "отклик", "приложение"],
        )
        unit = self._unit_with_entities(
            "Климатические условия эксплуатации должны соответствовать "
            "стандартам ВЦ.",
            entities=["климат", "эксплуатация", "стандарт"],
        )
        j = _judgment(LLMLabel.PARTIAL, conf=0.50)
        out = self.verifier.verify(j, req, unit)
        # No substantive shared token → demoted as before.
        assert out.rule_adjusted_label == LLMLabel.IRRELEVANT, (
            f"off-topic PARTIAL must still demote; got "
            f"{out.rule_adjusted_label}, actions={out.verifier_actions}"
        )
