"""Mask PII before any text leaves the process (plan section 3.2).

Every external call path (LLM proxy, JEV) is outside the country, so masking
is a mandatory gate, not an option. One `PiiMasker` is used per sample so the
same value maps to the same placeholder across question, answer, contexts, and
reference, which keeps entailment judgments intact. The reverse mapping lives
only in memory and is never written anywhere.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ragas_jev.schemas import RagSample

# Order matters: more specific patterns run first so that, e.g., a phone
# number is not swallowed by the generic account-number pattern.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # 주민등록번호 / 외국인등록번호: YYMMDD-GNNNNNN
    ("RRN", re.compile(r"(?<!\d)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\s?-\s?[1-8]\d{6}(?!\d)")),
    ("CARD", re.compile(r"(?<!\d)\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}(?!\d)|(?<!\d)\d{16}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+82[- ]?)?0\d{1,2}[- .]?\d{3,4}[- .]?\d{4}(?!\d)")),
    # 사업자등록번호: NNN-NN-NNNNN
    ("BRN", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{5}(?!\d)")),
    # 계좌번호: dash-separated digit groups with at least 10 digits in total
    ("ACCOUNT", re.compile(r"(?<!\d)\d{2,6}(?:-\d{2,7}){2,3}(?!\d)")),
]
_ACCOUNT_MIN_DIGITS = 10


class PiiMaskingError(RuntimeError):
    """Masking failed; the sample must not be sent to any external service."""


@dataclass
class PiiMasker:
    custom_terms: list[str] = field(default_factory=list)
    _mapping: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)
    _counts: Counter[str] = field(default_factory=Counter)

    def mask(self, text: str) -> str:
        try:
            for term in sorted(set(self.custom_terms), key=len, reverse=True):
                if term:
                    text = text.replace(term, self._placeholder("TERM", term))
            for kind, pattern in _PATTERNS:
                text = pattern.sub(lambda m, kind=kind: self._replace(kind, m.group(0)), text)
        except Exception as exc:  # noqa: BLE001 - any failure must block sending
            raise PiiMaskingError(str(exc)) from exc
        return text

    def _replace(self, kind: str, value: str) -> str:
        if kind == "ACCOUNT" and sum(c.isdigit() for c in value) < _ACCOUNT_MIN_DIGITS:
            return value
        return self._placeholder(kind, value)

    def _placeholder(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key not in self._mapping:
            self._counts[kind] += 1
            self._mapping[key] = f"[PII_{kind}_{self._counts[kind]}]"
        return self._mapping[key]

    @property
    def report(self) -> dict[str, int]:
        """Number of distinct masked values per type; never includes the values."""
        return dict(self._counts)


def mask_sample(sample: RagSample, custom_terms: list[str] | None = None) -> tuple[RagSample, dict[str, int]]:
    masker = PiiMasker(custom_terms=custom_terms or [])
    masked = sample.model_copy(
        update={
            "question": masker.mask(sample.question),
            "answer": masker.mask(sample.answer),
            "contexts": [masker.mask(c) for c in sample.contexts],
            "reference": masker.mask(sample.reference) if sample.reference is not None else None,
        }
    )
    return masked, masker.report
