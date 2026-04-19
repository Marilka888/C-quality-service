from __future__ import annotations

import unittest

from app.core.config import ServiceConfig
from app.domain.entities import Requirement, TestCase
from app.service.traceability import TraceabilityService


class TraceabilityServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TraceabilityService(ServiceConfig())

    def test_marks_adequate_when_requirement_is_clearly_covered(self) -> None:
        report = self.service.evaluate(
            requirements=[
                Requirement(
                    id="REQ-1",
                    text="Система должна блокировать пользователя после 3 неудачных попыток входа.",
                    section="Безопасность",
                    source_doc_id="TZ-1",
                    page=5,
                )
            ],
            test_cases=[
                TestCase(
                    id="TC-1",
                    text="Выполнить 3 неудачные попытки входа и проверить блокировку пользователя.",
                    expected_result="После 3 попыток пользователь блокируется.",
                    section="Испытания безопасности",
                    source_doc_id="PMI-1",
                    page=7,
                )
            ],
        )

        finding = report["detailed_findings"][0]
        self.assertEqual("ADEQUATE", finding["final_status"])
        self.assertEqual(1, report["summary"]["adequate_count"])

    def test_conflict_is_not_hidden_by_more_similar_inadequate_candidate(self) -> None:
        report = self.service.evaluate(
            requirements=[
                Requirement(
                    id="REQ-2",
                    text="Система должна ограничивать размер файла 100 МБ и блокировать загрузку после 3 неудачных попыток.",
                    section="Загрузка",
                    source_doc_id="TZ-1",
                )
            ],
            test_cases=[
                TestCase(
                    id="TC-2A",
                    text="Проверить загрузку файла и обработку ошибок пользователя.",
                    expected_result="Система сообщает об ошибке загрузки.",
                    section="Загрузка",
                    source_doc_id="PMI-1",
                ),
                TestCase(
                    id="TC-2B",
                    text="Выполнить 5 неудачных попыток загрузки файла размером 200 МБ и проверить блокировку.",
                    expected_result="После 5 попыток загрузка блокируется.",
                    section="Загрузка",
                    source_doc_id="PMI-1",
                ),
            ],
        )

        finding = report["detailed_findings"][0]
        self.assertEqual("CONFLICT", finding["final_status"])
        self.assertEqual("TC-2B", finding["selected_best_match"]["test_case_id"])

    def test_marks_missing_when_no_candidate_found(self) -> None:
        report = self.service.evaluate(
            requirements=[
                Requirement(
                    id="REQ-3",
                    text="Система должна поддерживать экспорт отчета в PDF.",
                    section="Отчеты",
                    source_doc_id="TZ-1",
                )
            ],
            test_cases=[
                TestCase(
                    id="TC-3",
                    text="Проверить авторизацию администратора.",
                    expected_result="Администратор входит в систему.",
                    section="Безопасность",
                    source_doc_id="PMI-1",
                )
            ],
        )

        self.assertEqual(1, report["summary"]["missing_count"])
        self.assertEqual(["REQ-3"], report["uncovered_requirements"])

    def test_detects_orphan_tests_and_penalizes_score(self) -> None:
        report = self.service.evaluate(
            requirements=[
                Requirement(
                    id="REQ-4",
                    text="Система должна поддерживать экспорт отчета в PDF.",
                    section="Отчеты",
                    source_doc_id="TZ-1",
                )
            ],
            test_cases=[
                TestCase(
                    id="TC-4",
                    text="Проверить экспорт отчета в PDF.",
                    expected_result="Отчет выгружается в PDF.",
                    section="Отчеты",
                    source_doc_id="PMI-1",
                ),
                TestCase(
                    id="TC-5",
                    text="Проверить отображение логотипа на стартовом экране.",
                    expected_result="Логотип отображается.",
                    section="Интерфейс",
                    source_doc_id="PMI-1",
                ),
            ],
        )

        self.assertEqual(1, report["summary"]["orphan_test_count"])
        self.assertLess(report["aggregated_metrics"]["score_c"], 1.0)

    def test_does_not_mark_meaningful_candidate_as_orphan(self) -> None:
        report = self.service.evaluate(
            requirements=[
                Requirement(
                    id="REQ-5",
                    text="Система должна поддерживать экспорт отчета в PDF с предпросмотром перед выгрузкой.",
                    section="Отчеты",
                    source_doc_id="TZ-1",
                )
            ],
            test_cases=[
                TestCase(
                    id="TC-5A",
                    text="Проверить экспорт отчета в PDF с открытием окна предпросмотра.",
                    expected_result="Перед выгрузкой отображается предпросмотр и доступен экспорт в PDF.",
                    section="Отчеты",
                    source_doc_id="PMI-1",
                ),
                TestCase(
                    id="TC-5B",
                    text="Проверить экспорт отчета в PDF.",
                    expected_result="Отчет выгружается в PDF.",
                    section="Отчеты",
                    source_doc_id="PMI-1",
                ),
                TestCase(
                    id="TC-5C",
                    text="Проверить смену темы интерфейса.",
                    expected_result="Тема интерфейса переключается.",
                    section="Интерфейс",
                    source_doc_id="PMI-1",
                ),
            ],
        )

        orphan_ids = {item["test_id"] for item in report["orphan_test_cases"]}
        self.assertNotIn("TC-5B", orphan_ids)
        self.assertIn("TC-5C", orphan_ids)

    def test_marks_partial_when_expected_result_is_missing(self) -> None:
        report = self.service.evaluate(
            requirements=[
                Requirement(
                    id="REQ-6",
                    text="Система должна уведомлять пользователя о завершении импорта файла.",
                    section="Импорт",
                    source_doc_id="TZ-1",
                )
            ],
            test_cases=[
                TestCase(
                    id="TC-6",
                    text="Проверить завершение импорта файла и наличие уведомления пользователю.",
                    expected_result=None,
                    section="Импорт",
                    source_doc_id="PMI-1",
                )
            ],
        )

        finding = report["detailed_findings"][0]
        self.assertEqual("PARTIAL", finding["final_status"])
        self.assertIn("EXPECTED_RESULT_MISSING", finding["rule_flags"])

    def test_thematically_similar_but_weak_test_is_not_adequate(self) -> None:
        report = self.service.evaluate(
            requirements=[
                Requirement(
                    id="REQ-7",
                    text="Система должна поддерживать двухфакторную аутентификацию для администратора.",
                    section="Безопасность",
                    source_doc_id="TZ-1",
                )
            ],
            test_cases=[
                TestCase(
                    id="TC-7",
                    text="Проверить вход администратора в систему.",
                    expected_result="Администратор успешно входит.",
                    section="Безопасность",
                    source_doc_id="PMI-1",
                )
            ],
        )

        finding = report["detailed_findings"][0]
        self.assertEqual("INADEQUATE", finding["final_status"])

    def test_partial_numeric_mismatch_is_detected_without_strong_conflict(self) -> None:
        report = self.service.evaluate(
            requirements=[
                Requirement(
                    id="REQ-8",
                    text="Система должна ограничивать размер файла 100 МБ и блокировать загрузку после 3 неудачных попыток.",
                    section="Загрузка",
                    source_doc_id="TZ-1",
                )
            ],
            test_cases=[
                TestCase(
                    id="TC-8",
                    text="Выполнить 3 неудачные попытки загрузки файла размером 200 МБ и проверить блокировку.",
                    expected_result="После 3 попыток загрузка блокируется.",
                    section="Загрузка",
                    source_doc_id="PMI-1",
                )
            ],
        )

        finding = report["detailed_findings"][0]
        self.assertEqual("PARTIAL", finding["final_status"])
        self.assertIn("NUMERIC_PARTIAL_MISMATCH", finding["rule_flags"])
        self.assertNotIn("NUMERIC_MISMATCH", finding["rule_flags"])

    def test_thresholds_are_taken_from_config(self) -> None:
        config = ServiceConfig()
        config.scoring.thresholds.adequate_overlap_threshold = 0.9
        config.scoring.thresholds.inadequate_score_threshold = 0.05
        service = TraceabilityService(config)

        report = service.evaluate(
            requirements=[
                Requirement(
                    id="REQ-9",
                    text="Сервис должен формировать отчет по операциям пользователя.",
                    section="Отчеты",
                    source_doc_id="TZ-1",
                )
            ],
            test_cases=[
                TestCase(
                    id="TC-9",
                    text="Проверить формирование отчета по операциям пользователя.",
                    expected_result="Отчет формируется и отображается пользователю.",
                    section="Отчеты",
                    source_doc_id="PMI-1",
                )
            ],
        )

        finding = report["detailed_findings"][0]
        self.assertEqual("PARTIAL", finding["final_status"])


if __name__ == "__main__":
    unittest.main()
