from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]+", re.UNICODE)

STOP_WORDS = {
    "и",
    "или",
    "в",
    "во",
    "на",
    "по",
    "для",
    "при",
    "к",
    "ко",
    "с",
    "со",
    "от",
    "до",
    "из",
    "под",
    "над",
    "не",
    "это",
    "как",
    "а",
    "но",
    "если",
    "то",
    "ли",
    "система",
    "должен",
    "должна",
    "должно",
    "должны",
    "обеспечивать",
    "обеспечивает",
    "обеспечить",
    "проверить",
    "проверяет",
    "проверка",
    "выполнить",
    "выполняется",
    "выполняет",
    "наличие",
    "результат",
    "ожидаемый",
    "ожидается",
}


def tokenize_content(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if token and token.lower() not in STOP_WORDS
    }


def tokenize_raw(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "")}
