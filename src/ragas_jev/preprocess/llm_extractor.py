"""Layer 1 preprocessor: LLM-based claim / statement extraction (plan section 7.1).

LLM outputs are not deterministic (GPT-5+ models take no `temperature`), so
extractions are cached by (model, prompt id, masked input). Re-running an
evaluation therefore reuses the same units, which is what makes the final
scores reproducible. The cache holds masked text only and stays local.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ragas_jev.cache import AnswerCache, stable_hash
from ragas_jev.config import Settings
from ragas_jev.llm_json import loads_first
from ragas_jev.preprocess.base import ExtractedUnit
from ragas_jev.preprocess.normalizer import dedupe_units
from ragas_jev.preprocess.prompts import (
    CLAIM_EXTRACTION_V1,
    REF_CLAIM_EXTRACTION_V1,
    STATEMENT_EXTRACTION_V1,
    Prompt,
)

logger = logging.getLogger(__name__)


class ExtractionError(ValueError):
    """The LLM response could not be turned into a list of units."""


class _Units(BaseModel):
    units: list[ExtractedUnit]


def parse_units(content: str | None, key: str) -> list[ExtractedUnit]:
    """Accept {"<key>": [...]}, a bare list, or items given as strings or objects."""
    data: Any = loads_first(content)
    if isinstance(data, dict):
        if key in data:
            data = data[key]
        elif len(data) == 1:
            data = next(iter(data.values()))
    if not isinstance(data, list):
        raise ExtractionError(f"expected a list of {key}, got {type(data).__name__}")
    items = []
    for item in data:
        if isinstance(item, str):
            items.append({"text": item})
        elif isinstance(item, dict):
            text = item.get("text") or item.get("claim") or item.get("statement")
            items.append({"text": text, "source_span": item.get("source_span")})
        else:
            raise ExtractionError(f"unexpected item type {type(item).__name__}")
    return _Units.model_validate({"units": items}).units


class LlmExtractor:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        cache: AnswerCache | None = None,
        max_concurrency: int = 8,
        seed: int = 7,
        max_units: int = 30,
    ) -> None:
        self._client = client
        self._model = model
        self._cache = cache
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._seed = seed
        self.max_units = max_units
        self.calls = 0

    @classmethod
    def from_settings(cls, settings: Settings, cache: AnswerCache | None = None) -> LlmExtractor:
        client = AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key.get_secret_value())
        return cls(client, settings.preprocessor_model, cache=cache)

    @property
    def version(self) -> str:
        return f"{self._model}:{CLAIM_EXTRACTION_V1.id}"

    async def claims(self, question: str, answer: str) -> list[ExtractedUnit]:
        return await self._extract(CLAIM_EXTRACTION_V1, "claims", question, answer)

    async def ref_claims(self, question: str, reference: str) -> list[ExtractedUnit]:
        return await self._extract(REF_CLAIM_EXTRACTION_V1, "claims", question, reference)

    async def statements(self, question: str, answer: str) -> list[ExtractedUnit]:
        return await self._extract(STATEMENT_EXTRACTION_V1, "statements", question, answer)

    async def _extract(self, prompt: Prompt, key: str, question: str, text: str) -> list[ExtractedUnit]:
        if not text.strip():
            return []
        cache_key = stable_hash("extract", self._model, prompt.id, question, text)
        cached = self._cache.get(cache_key) if self._cache else None
        if cached is not None:
            units = [ExtractedUnit.model_validate(u) for u in cached["units"]]
        else:
            units = await self._call(prompt, key, question, text)
            if self._cache:
                self._cache.set(cache_key, {"units": [u.model_dump() for u in units]})
        units = dedupe_units(u for u in units if u.text.strip())
        if len(units) > self.max_units:
            logger.warning("%s: %d units exceed max_units=%d", prompt.id, len(units), self.max_units)
        return units

    async def _call(self, prompt: Prompt, key: str, question: str, text: str) -> list[ExtractedUnit]:
        user = json.dumps({"question": question, "answer": text}, ensure_ascii=False)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=1, max=20),
            retry=retry_if_exception_type((APIError, ValidationError, ExtractionError, json.JSONDecodeError)),
            reraise=True,
        ):
            with attempt:
                async with self._semaphore:
                    # No `temperature`: GPT-5 and later models do not accept it.
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "system", "content": prompt.system}, {"role": "user", "content": user}],
                        response_format={"type": "json_object"},
                        seed=self._seed,
                    )
                self.calls += 1
                return parse_units(response.choices[0].message.content, key)
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self._client.close()
