"""Deterministic numeric consistency check for claims (plan Phase 2).

JEV is weak at numeric comparison (TypeSafe jaggedness docs), so numbers are
compared in code: every number in a claim should also appear in the contexts,
after normalizing separators and magnitude words (만/억/조, thousand/million/...).
The result is a diagnostic; the pipeline reports it next to the JEV score and
can optionally cap the support probability of mismatching claims.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_MAGNITUDE = {
    "천": 1e3, "만": 1e4, "억": 1e8, "조": 1e12,
    "thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6,
    "billion": 1e9, "bn": 1e9, "trillion": 1e12, "tn": 1e12,
}
_NUMBER = re.compile(
    # Not part of an identifier such as "COVID-19", "B2B", or "v1.2".
    r"(?<![\w.\-/])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*"
    # Single-letter "m"/"b" are left out: too often meters/bytes rather than million/billion.
    r"(천|만|억|조|thousand|million|billion|trillion|mn|bn|tn|k)?(?![A-Za-z])",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(r"\[PII_[A-Z]+_\d+\]")


@dataclass(frozen=True)
class NumberToken:
    raw: str
    value: float


@dataclass
class NumericCheck:
    numbers: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def has_numbers(self) -> bool:
        return bool(self.numbers)

    @property
    def mismatch(self) -> bool:
        return bool(self.missing)

    def as_dict(self) -> dict[str, list[str]]:
        return {"numbers": self.numbers, "missing": self.missing}


def extract_numbers(text: str) -> list[NumberToken]:
    text = _PLACEHOLDER.sub(" ", text)  # PII placeholders contain digits
    tokens = []
    for m in _NUMBER.finditer(text):
        integer, fraction, magnitude = m.group(1), m.group(2) or "", m.group(3)
        value = float(integer.replace(",", "") + fraction)
        if magnitude:
            value *= _MAGNITUDE[magnitude.lower()]
        tokens.append(NumberToken(raw=m.group(0).strip(), value=value))
    return tokens


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def check_claim(claim: str, contexts: list[str]) -> NumericCheck:
    claim_numbers = extract_numbers(claim)
    if not claim_numbers:
        return NumericCheck()
    context_values = [t.value for c in contexts for t in extract_numbers(c)]
    # A bare value also matches when the context writes it with a magnitude word
    # split off (e.g. claim "5,000억" vs context "5000 억"): compare values only.
    missing = [t.raw for t in claim_numbers if not any(_close(t.value, v) for v in context_values)]
    return NumericCheck(numbers=[t.raw for t in claim_numbers], missing=missing)
