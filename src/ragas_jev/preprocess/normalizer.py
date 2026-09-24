"""Unit normalization and duplicate removal (plan section 7.1).

Exact duplicates are dropped after whitespace/case normalization; near
duplicates are dropped when their character-trigram Jaccard similarity is at
least `near_threshold` (no embeddings needed).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragas_jev.preprocess.base import ExtractedUnit

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\s.,!?;:\"'()\[\]{}·…“”‘’]+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _trigrams(text: str) -> set[str]:
    compact = _PUNCT.sub("", text.casefold())
    return {compact[i : i + 3] for i in range(max(1, len(compact) - 2))}


def jaccard(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


def dedupe(texts: Iterable[str], near_threshold: float = 0.9) -> list[str]:
    kept: list[str] = []
    for text in texts:
        norm = normalize(text)
        if norm and all(jaccard(norm, k) < near_threshold for k in kept):
            kept.append(norm)
    return kept


def dedupe_units(units: Iterable[ExtractedUnit], near_threshold: float = 0.9) -> list[ExtractedUnit]:
    kept: list[ExtractedUnit] = []
    for unit in units:
        text = normalize(unit.text)
        if text and all(jaccard(text, k.text) < near_threshold for k in kept):
            kept.append(unit.model_copy(update={"text": text}))
    return kept
