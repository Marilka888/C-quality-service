"""
PR-K follow-up regression tests:

  * Extended _PROHIBITION_RE — recognises all four gender/number forms
    of "не должен/должна/должно/должны", verb forms запрещается /
    запрещаются, без возможности.
  * Numeric-conflict topical-link guard — same-topic same-unit
    different-value pairs raise CONFLICT; different-topic pairs with
    same-unit-class numeric coincidence stay IRRELEVANT.
  * End-to-end negation conflict via the verifier (LLM-PARTIAL/COVERED
    label + clear prohibition mismatch → CONFLICT_VERIFIED).

Driven by smoke-time symptom on TZ#2 ("не должно превышать 2 секунд")
where the old regex missed neuter "не должно" → false-negative on the
modality-prohibition probe.
"""
from __future__ import annotations

import pytest

from app.application.use_cases.verify_pairs import (
    PairVerifier,
    _PROHIBITION_RE,
    _negation_contradiction,
    _find_numeric_conflict,
)
from app.domain.c_quality_enums import LLMLabel, Modality
from app.domain.c_quality_models import (
    Constraint,
    CoverageUnit,
    PairJudgment,
    RequirementUnit,
)


# ── _PROHIBITION_RE coverage ────────────────────────────────────────────


class TestProhibitionRegex:
    """Every prohibition form the verifier needs to recognise as a
    strict prohibition (modality MUST_NOT)."""

    @pytest.mark.parametrize("text", [
        # Russian "не должен/должна/должно/должны"
        "Система не должен передавать данные третьим лицам.",
        "Программа не должна сохранять пароль в открытом виде.",
        "Время отклика не должно превышать двух секунд.",   # ← was missing in old regex
        "Пользователи не должны иметь доступ к этим данным.",
        # запрещён/запрещена/запрещены/запрещается/запрещаются
        "Запрещено хранить пароли пользователей в логах.",
        "Запрещена передача данных без шифрования.",
        "Запрещены любые попытки обхода аутентификации.",
        "Запрещается экспортировать персональные данные.",   # ← verb form
        "Запрещаются операции записи в системные таблицы.",
        # недопустим/недопустима/недопустимы
        "Недопустимо логирование чувствительных данных.",
        "Передача пароля по HTTP недопустима.",
        "Незащищённые соединения недопустимы в production.",
        # не допускается / не разрешается + plural
        "Не допускается обработка персональных данных без согласия.",
        "Не допускаются обращения к внешним сервисам.",
        "Не разрешается изменение конфигурации в runtime.",
        "Не разрешаются прямые обращения к базе данных из UI.",
        # без возможности
        "Журнал событий хранится без возможности изменения.",
        "Транзакции должны быть выполнены без возможности отката.",
        # English
        "User passwords must be stored not allowed in plaintext.",
        "Direct database access is forbidden from the UI layer.",
        "Cross-domain requests are prohibited without authorisation.",
    ])
    def test_prohibition_recognised(self, text: str):
        assert _PROHIBITION_RE.search(text), (
            f"prohibition not recognised: {text!r}"
        )

    @pytest.mark.parametrize("text", [
        # Quantifiers — must NOT match (false-positive guard).
        "Время отклика должно быть не более 2 секунд.",
        "Система должна обрабатывать не менее 100 запросов в секунду.",
        "Журнал хранится не больше 365 дней.",
        # Positive affirmations.
        "Система должна обеспечивать доступ к данным.",
        "Программа должна корректно завершать работу.",
        "Backend реализован на Python.",
        # Other words that contain "должн"/"запрещ" as a substring.
        "Долженствование как философская категория.",
        # "недопустимое" inside a different word — \b boundary protects.
        "Недопустимое поведение фиксируется в логах.",
    ])
    def test_quantifier_and_positive_not_matched(self, text: str):
        assert not _PROHIBITION_RE.search(text), (
            f"false-positive prohibition match on: {text!r}"
        )


# ── _negation_contradiction across all four genders ─────────────────────


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


class TestNegationContradictionAllForms:
    """All four "не должен/должна/должно/должны" gender/number forms in
    the requirement, paired with a positive affirmation in the unit, must
    raise the negation-contradiction signal."""

    @pytest.mark.parametrize("req_text, unit_text", [
        # masculine
        ("Сервис не должен передавать пароли в открытом виде.",
         "Сервис передаёт пароли по защищённому каналу с TLS."),
        # feminine
        ("Программа не должна сохранять пароли в логах.",
         "Программа сохраняет введённые пароли в системный лог."),
        # neuter — was missing from the old regex (the smoke-time bug)
        ("Время отклика не должно превышать двух секунд.",
         "Время отклика составляет 5 секунд при типовой нагрузке."),
        # plural
        ("Пользователи не должны видеть данные других клиентов.",
         "Пользователи видят данные всех клиентов в общем списке."),
    ])
    def test_negation_detected_for_all_forms(self, req_text, unit_text):
        req = _req(req_text)
        unit = _unit(unit_text)
        assert _negation_contradiction(req, unit) is True, (
            f"negation_contradiction missed for req={req_text!r} unit={unit_text!r}"
        )


# ── End-to-end via verifier: prohibition mismatch → CONFLICT ────────────


class TestEndToEndProhibitionConflict:
    """When LLM labels PARTIAL/COVERED but the texts have a clear
    "X не должно делать Y" / "X делает Y" mismatch, the verifier must
    promote the verdict to CONFLICT_VERIFIED with conflict_confirmed_negation."""

    def setup_method(self):
        self.verifier = PairVerifier()

    @pytest.mark.parametrize("req_text, unit_text", [
        ("Программа не должна сохранять пароль в открытом виде.",
         "Программа сохраняет пароль пользователя в журнале аудита."),
        ("Время отклика не должно превышать 2 секунд.",
         "Время отклика составляет 5 секунд при пиковой нагрузке."),
    ])
    def test_prohibition_vs_affirmation_promotes_to_conflict(self, req_text, unit_text):
        req = _req(req_text)
        unit = _unit(unit_text)
        # LLM thinks they're related (PARTIAL) but doesn't catch the
        # prohibition mismatch — verifier must.
        j = PairJudgment(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            llm_label=LLMLabel.PARTIAL,
            rule_adjusted_label=LLMLabel.PARTIAL,
            llm_confidence=0.55,
        )
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.CONFLICT, (
            f"verifier didn't promote to CONFLICT; "
            f"actions={out.verifier_actions}"
        )
        assert "conflict_confirmed_negation" in out.verifier_actions
        # Confidence bumped so the aggregator's CONFLICT gate passes.
        assert out.llm_confidence >= 0.85


# ── Numeric conflict — same topic vs different topic ────────────────────


class TestNumericTopicalLink:
    """The numeric-conflict rule's topical-link guard (PR-K). Same-topic
    same-unit-class different-value pairs must raise CONFLICT; different-
    topic pairs that happen to share a numeric value or unit class must
    NOT raise CONFLICT."""

    def setup_method(self):
        self.verifier = PairVerifier()

    def _build_pair(
        self, req_text, unit_text, req_constraints, unit_constraints,
        llm_label=LLMLabel.IRRELEVANT, llm_conf=0.30,
    ):
        from app.application.use_cases.build_requirements import _normalize_text

        req = RequirementUnit(
            req_id="r1", source_document_id="doc-tz",
            text=req_text, normalized_text=_normalize_text(req_text),
            constraints=req_constraints,
        )
        unit = CoverageUnit(
            unit_id="u1", target_document_id="doc-pmi", target_doc_role="pmi",
            text=unit_text, normalized_text=_normalize_text(unit_text),
            constraints=unit_constraints,
        )
        j = PairJudgment(
            req_id="r1", unit_id="u1", target_document_id="doc-pmi",
            llm_label=llm_label, rule_adjusted_label=llm_label,
            llm_confidence=llm_conf,
        )
        return req, unit, j

    def test_same_topic_logs_90_vs_30_days_raises_conflict(self):
        """Same topic (журнал/логи), same kind (retention_period),
        same unit class (days), different values → CONFLICT."""
        req, unit, j = self._build_pair(
            req_text="Журнал событий безопасности должен храниться не менее 90 дней.",
            unit_text="Журнал событий хранится за последние 30 суток.",
            req_constraints=[
                Constraint(kind="retention_period", operator=">=", value=90, unit="days"),
            ],
            unit_constraints=[
                Constraint(kind="retention_period", operator="=", value=30, unit="days"),
            ],
        )
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.CONFLICT, (
            f"expected CONFLICT, got {out.rule_adjusted_label}; "
            f"actions={out.verifier_actions}"
        )
        assert "conflict_confirmed_numeric" in out.verifier_actions

    def test_different_topic_logs_90_vs_blocking_30_days_no_conflict(self):
        """Different topics: req is about LOG retention (90 days), unit
        is about USER-blocking duration (30 days). Same unit-class but
        different `kind` and zero topical token overlap → must NOT raise
        CONFLICT (topical-link guard rejects)."""
        req, unit, j = self._build_pair(
            req_text="Журнал событий должен храниться не менее 90 дней.",
            unit_text="Учётная запись блокируется на 30 дней при попытке несанкционированного доступа.",
            req_constraints=[
                Constraint(kind="retention_period", operator=">=", value=90, unit="days"),
            ],
            unit_constraints=[
                Constraint(kind="block_duration", operator="=", value=30, unit="days"),
            ],
        )
        out = self.verifier.verify(j, req, unit)
        # Different declared kinds → values_conflict returns None →
        # numeric_conflicts is empty → no promotion.
        assert out.rule_adjusted_label == LLMLabel.IRRELEVANT, (
            f"different-topic 90/30 days promoted; actions={out.verifier_actions}"
        )

    def test_same_topic_response_time_2_vs_5_sec_raises_conflict(self):
        """Same topic (response time), same kind, different values."""
        req, unit, j = self._build_pair(
            req_text="Время отклика API не должно превышать 2 секунд.",
            unit_text="Время отклика API составляет 5 секунд при типовой нагрузке.",
            req_constraints=[
                Constraint(kind="response_time", operator="<=", value=2, unit="sec"),
            ],
            unit_constraints=[
                Constraint(kind="response_time", operator="=", value=5, unit="sec"),
            ],
            llm_label=LLMLabel.PARTIAL, llm_conf=0.45,
        )
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.CONFLICT
        assert "conflict_confirmed_numeric" in out.verifier_actions

    def test_unitless_numbers_no_conflict(self):
        """Unitless coincidence ('section 30' vs 'page 90') must NOT
        produce CONFLICT — both unit=None pairs are skipped by guard."""
        req, unit, j = self._build_pair(
            req_text="Раздел 30 содержит требования к интерфейсу.",
            unit_text="На странице 90 описан внешний вид окна.",
            req_constraints=[
                Constraint(kind="generic", operator="=", value=30, unit=None),
            ],
            unit_constraints=[
                Constraint(kind="generic", operator="=", value=90, unit=None),
            ],
        )
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.IRRELEVANT, (
            f"unitless number coincidence promoted; actions={out.verifier_actions}"
        )

    def test_same_unit_class_different_kind_explicit_no_conflict(self):
        """Both sides have time-class units but different `kind`
        (response_time vs retention_period). values_conflict() must
        return None for explicit different-kind pairs."""
        req, unit, j = self._build_pair(
            req_text="Время отклика должно быть не более 2 секунд.",
            unit_text="Журнал хранится 90 секунд для тестирования.",
            req_constraints=[
                Constraint(kind="response_time", operator="<=", value=2, unit="sec"),
            ],
            unit_constraints=[
                Constraint(kind="retention_period", operator="=", value=90, unit="sec"),
            ],
        )
        out = self.verifier.verify(j, req, unit)
        assert out.rule_adjusted_label == LLMLabel.IRRELEVANT
