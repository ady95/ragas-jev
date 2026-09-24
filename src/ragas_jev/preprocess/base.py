"""Layer 1 extractor interface and an offline sentence-split baseline.

`LlmExtractor` (llm_extractor.py) is the real preprocessor. The sentence
splitter lets the pipeline run without any LLM call; its units are whole
sentences, not atomic claims.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel

from ragas_jev.preprocess.normalizer import dedupe_units

_SENTENCE_END = re.compile(r"(?<=[.!?。])\s+|\n+")


class ExtractedUnit(BaseModel):
    text: str
    # Verbatim substring of the source text the unit came from (for tracing
    # and for span-level evaluation against labeled hallucinations).
    source_span: str | None = None


class Extractor(Protocol):
    @property
    def version(self) -> str: ...

    async def claims(self, question: str, answer: str) -> list[ExtractedUnit]: ...

    async def ref_claims(self, question: str, reference: str) -> list[ExtractedUnit]: ...

    async def statements(self, question: str, answer: str) -> list[ExtractedUnit]: ...


def split_sentences(text: str) -> list[ExtractedUnit]:
    return dedupe_units(ExtractedUnit(text=s, source_span=s.strip()) for s in _SENTENCE_END.split(text) if s.strip())


class SentenceSplitExtractor:
    @property
    def version(self) -> str:
        return "sentence-split.v1"

    async def claims(self, question: str, answer: str) -> list[ExtractedUnit]:
        return split_sentences(answer)

    async def ref_claims(self, question: str, reference: str) -> list[ExtractedUnit]:
        return split_sentences(reference)

    async def statements(self, question: str, answer: str) -> list[ExtractedUnit]:
        return split_sentences(answer)
