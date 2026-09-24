"""JEV Primary Judge backed by the official `typesafe-sdk` (plan section 7.2)."""

from __future__ import annotations

import asyncio
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from ragas_jev.config import Settings
from ragas_jev.judge.base import JudgeQuestion, JudgeResponseError, JudgeResult, RawAnswer


class JevJudge:
    """Sends one state with many questions per request (speculative fan-out)."""

    def __init__(
        self,
        client: AsyncTypeSafeClient,
        model: str,
        *,
        max_concurrency: int = 8,
        max_questions_per_request: int = 40,
    ) -> None:
        self._client = client
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_questions = max_questions_per_request

    @classmethod
    def from_settings(cls, settings: Settings) -> JevJudge:
        if settings.typesafe_api_key is None:
            raise ValueError("TYPESAFE_API_KEY is not set")
        client = AsyncTypeSafeClient(
            api_key=settings.typesafe_api_key.get_secret_value(),
            base_url=settings.typesafe_base_url,
            model=settings.typesafe_default_model,
            timeout=settings.jev_timeout_sec,
            retry=RetryPolicy(max_retries=4, backoff_max=10.0),
        )
        return cls(
            client,
            settings.typesafe_default_model,
            max_concurrency=settings.jev_max_concurrency,
            max_questions_per_request=settings.jev_max_questions_per_request,
        )

    @property
    def model_id(self) -> str:
        return self._model

    async def evaluate(self, state: dict[str, Any], questions: list[JudgeQuestion]) -> JudgeResult:
        ids = [q.unit_id for q in questions]
        if len(set(ids)) != len(ids):
            raise ValueError("question unit_ids must be unique within one request")
        result = JudgeResult()
        if not questions:
            return result
        batches = [questions[i : i + self._max_questions] for i in range(0, len(questions), self._max_questions)]
        for partial in await asyncio.gather(*(self._call(state, batch) for batch in batches)):
            result.answers.update(partial.answers)
            result.requests += partial.requests
            result.input_tokens += partial.input_tokens
        return result

    async def _call(self, state: dict[str, Any], batch: list[JudgeQuestion]) -> JudgeResult:
        async with self._semaphore:
            response = await self._client.system_one(
                state, {q.unit_id: q.to_payload() for q in batch}, model=self._model
            )
        result = JudgeResult(requests=1, input_tokens=response.usage.input_tokens or 0)
        for question in batch:
            answer = response.answers.get(question.unit_id)
            if answer is None:
                raise JudgeResponseError(f"missing answer for {question.unit_id}")
            if answer.type != question.primitive:
                raise JudgeResponseError(f"{question.unit_id}: expected {question.primitive}, got {answer.type}")
            result.answers[question.unit_id] = _to_raw(question, answer, response.model, response.request_id)
        return result

    async def list_models(self) -> list[str]:
        listing = await self._client.models.list()
        return [m.name for m in listing.models]

    async def aclose(self) -> None:
        await self._client.aclose()


def _to_raw(question: JudgeQuestion, answer: Any, model: str, request_id: str | None) -> RawAnswer:
    raw = RawAnswer(unit_id=question.unit_id, primitive=question.primitive, model=model, request_id=request_id)
    if answer.type == "noul":
        raw.noul = answer.noul
    elif answer.type == "choice":
        raw.choice = answer.choice
        raw.probabilities = dict(answer.probabilities)
        raw.confidence = answer.confidence
    else:
        raw.score = answer.score
        raw.probabilities = {str(level): prob for level, prob in answer.probabilities.items()}
        raw.confidence = answer.confidence
    return raw
