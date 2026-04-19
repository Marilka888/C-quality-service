"""
Prompt builder for the coverage judge.
PROMPT_VERSION is embedded in each prompt for traceability.
"""
from __future__ import annotations

from app.domain.c_quality_models import CoverageUnit, RequirementUnit

PROMPT_VERSION = "v1"

_SYSTEM_PROMPT_V1 = """Ты эксперт по анализу программной документации.
Твоя задача — оценить, покрывает ли фрагмент документа (ПМИ/ПЗ) требование из ТЗ.

Верни ТОЛЬКО JSON в строго следующем формате (без дополнительного текста):
{
  "label": "<COVERED|PARTIAL|CONFLICT|IRRELEVANT>",
  "confidence": <0.0-1.0>,
  "matched_aspects": ["<aspect1>", ...],
  "missing_aspects": ["<aspect1>", ...],
  "conflict_aspects": ["<aspect1>", ...],
  "explanation": "<краткое объяснение на русском>"
}

Значения label:
- COVERED: фрагмент полностью покрывает требование (все числовые ограничения совпадают)
- PARTIAL: фрагмент частично покрывает, но некоторые аспекты или ограничения не проверены
- CONFLICT: фрагмент относится к тому же предмету, но явно противоречит требованию (другое число, другая граница, другая модальность)
- IRRELEVANT: фрагмент не относится к данному требованию

Важно: MISSING не используется на этом уровне; используй только COVERED/PARTIAL/CONFLICT/IRRELEVANT.
"""


def build_judge_prompt(req: RequirementUnit, unit: CoverageUnit) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given pair."""
    system = _SYSTEM_PROMPT_V1

    user = (
        f"[prompt_version={PROMPT_VERSION}]\n\n"
        f"Тип документа-источника фрагмента: {unit.target_doc_role.upper()}\n"
        f"Тип требования: {req.requirement_type.value}\n\n"
        f"=== ТРЕБОВАНИЕ (из ТЗ) ===\n{req.text}\n\n"
        f"=== ФРАГМЕНТ ({unit.target_doc_role.upper()}) ===\n{unit.text}\n\n"
        "Оцени покрытие и верни JSON."
    )
    return system, user
