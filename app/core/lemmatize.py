"""
Russian lemmatization helper backed by pymorphy3.

Used by retrieval (lexical scoring) and entity matching so that surface
forms of the same word do not break overlap metrics:
    "хранение" / "хранить" / "хранится" / "хранит"   →   "хранить"
    "требования" / "требование"                       →   "требование"

Characteristics:

  - **Lazy singleton.** pymorphy3 imports a ~13 MB dictionary on first use;
    defer until someone asks. Tests that do not touch retrieval pay nothing.
  - **LRU-cached per token.** Morphological analysis is not cheap (~hundreds
    of μs per token); caching trivially-repeated TZ vocabulary brings
    full-document lemmatization down to a few ms.
  - **Safe fallback.** If pymorphy3 is unavailable we lowercase and return
    the token unchanged, so the rest of the pipeline keeps working.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import List

from app.core.logging import get_logger

logger = get_logger(__name__)

_LOCK = threading.Lock()
_MORPH = None
_ANALYZER_AVAILABLE: bool | None = None  # tri-state: None = not yet probed


def _get_analyzer():
    """Return the pymorphy3 MorphAnalyzer or None if not installed."""
    global _MORPH, _ANALYZER_AVAILABLE
    if _ANALYZER_AVAILABLE is False:
        return None
    if _MORPH is not None:
        return _MORPH
    with _LOCK:
        if _MORPH is not None:
            return _MORPH
        try:
            import pymorphy3  # type: ignore

            _MORPH = pymorphy3.MorphAnalyzer()
            _ANALYZER_AVAILABLE = True
            logger.info("pymorphy3 lemmatizer initialised")
        except Exception as exc:  # noqa: BLE001 — third-party import guard
            logger.warning(
                "pymorphy3 not available (%s); lemmatization disabled, "
                "falling back to lowercase-only matching", exc,
            )
            _ANALYZER_AVAILABLE = False
            _MORPH = None
        return _MORPH


@lru_cache(maxsize=50_000)
def lemma(token: str) -> str:
    """
    Return the base (dictionary) form of a single lowercased token.

    Non-Russian tokens, numbers, and single-character tokens are passed
    through unchanged (lowercased). pymorphy3 otherwise returns the most
    probable parse's `.normal_form`.
    """
    if not token:
        return ""
    low = token.lower()
    # Skip short / non-alphabetic tokens — pymorphy spends time on them
    # and returns them unchanged anyway.
    if len(low) < 2 or not any("а" <= c <= "я" or c == "ё" for c in low):
        return low
    analyzer = _get_analyzer()
    if analyzer is None:
        return low
    try:
        parses = analyzer.parse(low)
    except Exception:
        return low
    if not parses:
        return low
    return parses[0].normal_form


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Vectorised version that preserves token order."""
    return [lemma(t) for t in tokens]


def is_available() -> bool:
    """Return True if pymorphy3 is installed and initialised successfully."""
    _get_analyzer()
    return _ANALYZER_AVAILABLE is True
