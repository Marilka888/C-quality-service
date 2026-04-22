from __future__ import annotations

import re

from app.core.lemmatize import lemma

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]+", re.UNICODE)

# Stop-words. Stored as both surface forms (fast path) and lemmas — we
# drop a token if EITHER its raw lowercased form or its lemma is listed.
# Keeping both catches (a) non-Russian function words pymorphy3 can't
# lemmatise and (b) lemmas directly (more forgiving to dictionary
# drift across pymorphy3 versions).
STOP_WORDS = {
    # Prepositions / conjunctions / particles
    "и", "или", "в", "во", "на", "по", "для", "при", "к", "ко",
    "с", "со", "от", "до", "из", "под", "над", "не", "это", "как",
    "а", "но", "если", "то", "ли", "что", "чтобы", "чем",
    # Domain-generic nouns/verbs that appear in almost every req and PMI
    # step — they carry no discriminative signal for retrieval.
    "система", "системы", "системе", "систему", "системой", "системах",
    "должен", "должна", "должно", "должны", "должный",
    "обеспечивать", "обеспечивает", "обеспечить", "обеспечение",
    "проверить", "проверяет", "проверка", "проверить",
    "выполнить", "выполняется", "выполняет", "выполнение",
    "наличие", "результат", "ожидаемый", "ожидается",
    "программа", "программы", "программе", "программу",
    "быть", "являться", "использовать", "использоваться",
}


def _norm_token(token: str) -> str:
    """Lowercased lemma if pymorphy3 is available, else lowercase."""
    return lemma(token)


def tokenize_content(text: str) -> set[str]:
    """
    Tokenise `text` into a set of content lemmas.

    Drops stop-words (checked both against the raw lowercase form and the
    lemma) so retrieval and entity overlap are form-invariant:
        "хранение данных" and "хранить данные"  →  same tokens.
    """
    tokens: set[str] = set()
    for raw in TOKEN_RE.findall(text or ""):
        low = raw.lower()
        if low in STOP_WORDS:
            continue
        lem = _norm_token(raw)
        if lem in STOP_WORDS:
            continue
        # Prefer lemma in the returned set — that is the form the rest of
        # the pipeline will match against.
        tokens.add(lem)
    return tokens


def tokenize_raw(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "")}
