"""Judge wrapper that answers repeated questions from the SQLite cache."""

from __future__ import annotations

from typing import Any

from ragas_jev.cache import AnswerCache, stable_hash
from ragas_jev.judge.base import Judge, JudgeQuestion, JudgeResult, RawAnswer


class CachedJudge:
    def __init__(self, inner: Judge, cache: AnswerCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    async def evaluate(self, state: dict[str, Any], questions: list[JudgeQuestion]) -> JudgeResult:
        result = JudgeResult()
        misses: list[JudgeQuestion] = []
        keys: dict[str, str] = {}
        for q in questions:
            key = stable_hash(self.model_id, state, q.to_payload())
            keys[q.unit_id] = key
            hit = self._cache.get(key)
            if hit is None:
                misses.append(q)
            else:
                result.answers[q.unit_id] = RawAnswer.model_validate({**hit, "unit_id": q.unit_id})
                result.cached += 1
        if misses:
            fresh = await self._inner.evaluate(state, misses)
            for unit_id, answer in fresh.answers.items():
                self._cache.set(keys[unit_id], answer.model_dump(exclude={"unit_id"}))
            result.answers.update(fresh.answers)
            result.requests += fresh.requests
            result.input_tokens += fresh.input_tokens
        return result

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()
