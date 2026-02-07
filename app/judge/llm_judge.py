import json
import os
import re
import subprocess
from typing import Any, Dict, Optional


# ====== SETTINGS ======
# On Windows... Use an absolute path to ollama.exe (PowerShell often has PATH quirks).
OLLAMA_PATH = os.getenv(
    "OLLAMA_PATH",
    r"C:\Users\Marilka\AppData\Local\Programs\Ollama\ollama.exe",
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
TIMEOUT = int(os.getenv("TIMEOUT", "120"))


SYSTEM_PROMPT = """
Ты эксперт по программной документации.

Тебе дают требование из ТЗ (TZ) и тест из ПМИ (PMI).
Нужно определить отношение между ними.

Вердикт (ТОЛЬКО одно значение):
- COVERED  : тест явно проверяет требование
- MISSING  : тест для требования отсутствует (если PMI == null)
- CONFLICT : тест противоречит требованию
- EXTRA    : тест не относится к требованию

Правила:
1) Если PMI == null, вердикт ВСЕГДА MISSING.
2) Пиши кратко, без воды.
3) Ответ СТРОГО в JSON (без markdown, без пояснений вокруг JSON).

Формат ответа:
{"verdict": "COVERED", "explanation": "..."}
""".strip()


def _build_prompt(tz_text: str, pmi_text: Optional[str]) -> str:
    pmi_part = pmi_text if pmi_text is not None else "null"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"TZ: {tz_text}\n"
        f"PMI: {pmi_part}\n"
        "\nОтвет (JSON):"
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_VERDICT_RE = re.compile(r"\b(COVERED|MISSING|CONFLICT|EXTRA)\b")


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract first JSON object from model output."""
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(m.group(0))


def _normalize_verdict(value: Any) -> str:
    """Coerce verdict into one of allowed tokens."""
    if value is None:
        return "MISSING"
    s = str(value).upper()
    m = _VERDICT_RE.search(s)
    return m.group(1) if m else "MISSING"


def call_llm(tz_text: str, pmi_text: Optional[str]) -> Dict[str, Any]:
    if not os.path.exists(OLLAMA_PATH):
        raise FileNotFoundError(
            f"OLLAMA_PATH does not exist: {OLLAMA_PATH}. "
            "Set OLLAMA_PATH env var to a full path to ollama.exe"
        )

    prompt = _build_prompt(tz_text, pmi_text)

    # IMPORTANT on Windows: force UTF-8 for stdin/stdout to avoid cp1251 encode errors.
    proc = subprocess.run(
        [OLLAMA_PATH, "run", OLLAMA_MODEL],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT,
    )

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        raise RuntimeError(f"Ollama failed (code={proc.returncode}). stderr={stderr[:500]}")

    # Parse JSON; if model produced extra tokens, extract JSON object.
    data = _extract_json(stdout)
    data["verdict"] = _normalize_verdict(data.get("verdict"))
    if "explanation" not in data:
        data["explanation"] = ""
    return data
