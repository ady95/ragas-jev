"""LLM-as-a-Judge behind the same Judge interface as JEV.

Used as the Phase 1 comparison baseline and, from Phase 5, as the auditor for
uncertain JEV decisions. One chat completion answers every question of a
request, mirroring JEV's one-state-many-questions shape. Only Noul (yes/no)
questions are supported so far.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ragas_jev.config import Settings
from ragas_jev.llm_json import loads_first
from ragas_jev.judge.base import JudgeQuestion, JudgeResponseError, JudgeResult, RawAnswer

SYSTEM_PROMPT = """You are a careful evaluation judge.
You receive a JSON object `state` and a list of yes/no questions. Questions refer to
fields of `state` by name in backticks, e.g. `contexts[2]` or `question`.
Answer every question independently, using only the information in `state`.
For each question return `p_yes`: your probability (0 to 1) that the answer is yes.
Use values near 0 or 1 only when you are sure; use values near 0.5 when unsure."""

RESPONSE_SCHEMA = {
    "name": "judge_answers",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "p_yes": {"type": "number"}},
                    "required": ["id", "p_yes"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    },
}


class _Answer(BaseModel):
    id: str
    p_yes: float


class _Answers(BaseModel):
    answers: list[_Answer]


class LlmJudge:
    def __init__(self, client: AsyncOpenAI, model: str, *, max_concurrency: int = 8, seed: int = 7) -> None:
        self._client = client
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._seed = seed

    @classmethod
    def from_settings(cls, settings: Settings, model: str | None = None, max_concurrency: int = 8) -> LlmJudge:
        client = AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key.get_secret_value())
        return cls(client, model or settings.ragas_jev_auditor_model, max_concurrency=max_concurrency)

    @property
    def model_id(self) -> str:
        return self._model

    async def evaluate(self, state: dict[str, Any], questions: list[JudgeQuestion]) -> JudgeResult:
        unsupported = [q.unit_id for q in questions if q.primitive != "noul"]
        if unsupported:
            raise NotImplementedError(f"LlmJudge supports noul questions only: {unsupported}")
        if not questions:
            return JudgeResult()
        user = json.dumps(
            {
                "state": state,
                "questions": [{"id": q.unit_id, "question": q.instructions, "criteria": q.criteria} for q in questions],
            },
            ensure_ascii=False,
        )
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=1, max=20),
            retry=retry_if_exception_type((APIError, ValidationError, JudgeResponseError)),
            reraise=True,
        ):
            with attempt:
                async with self._semaphore:
                    response = await self._complete(user)
                by_id = {a.id: a.p_yes for a in _parse(response.choices[0].message.content)}
                missing = [q.unit_id for q in questions if q.unit_id not in by_id]
                if missing:
                    raise JudgeResponseError(f"LLM judge omitted answers: {missing}")
        result = JudgeResult(requests=1, input_tokens=response.usage.prompt_tokens if response.usage else 0)
        for q in questions:
            result.answers[q.unit_id] = RawAnswer(
                unit_id=q.unit_id,
                primitive="noul",
                noul=min(1.0, max(0.0, by_id[q.unit_id])),
                model=response.model or self._model,
                request_id=response.id,
            )
        return result

    async def _complete(self, user: str) -> Any:
        # No `temperature`: GPT-5 and later models do not accept it (the proxy
        # answers 500), so LLM judge outputs are not deterministic.
        return await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
            seed=self._seed,
        )

    async def aclose(self) -> None:
        await self._client.close()


def _parse(content: str | None) -> list[_Answer]:
    """Accept the shapes models actually return for the requested schema.

    The proxy accepts `json_schema` but does not enforce it. Observed shapes:
      {"answers": [{"id", "p_yes"}]}   (requested)
      [{"id", "p_yes"}]                 (gpt-5.6-sol, sometimes)
      {"<id>": {"p_yes": x}}            (gpt-6-luna)
      {"<id>": x}
      {"answers": {"<id>": x}}          (gpt-6-sol, occasionally)
      {"<other key>": [...]}           (gpt-6-sol, occasionally)
    Anything after the first JSON value (seen occasionally) is ignored.
    """
    data = loads_first(content)
    if isinstance(data, dict):
        if "answers" in data:
            data = data["answers"]
        elif len(data) == 1 and isinstance(next(iter(data.values())), list):
            data = next(iter(data.values()))  # e.g. {"results": [...]}
    if isinstance(data, dict):  # {"<id>": x} or {"<id>": {"p_yes": x}}
        data = [
            {"id": key, "p_yes": value["p_yes"] if isinstance(value, dict) and "p_yes" in value else value}
            for key, value in data.items()
        ]
    return _Answers.model_validate({"answers": data}).answers
