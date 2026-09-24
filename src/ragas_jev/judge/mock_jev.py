"""Deterministic offline judge for tests and development without API calls."""

from __future__ import annotations

from typing import Any

from ragas_jev.cache import stable_hash
from ragas_jev.judge.base import JudgeQuestion, JudgeResult, RawAnswer
from ragas_jev.scoring.uncertainty import spread_confidence


class MockJudge:
    """Returns hash-derived probabilities; `overrides` pins values per unit_id.

    An override is a float for Noul questions or a probability map for
    Score/Choice questions.
    """

    def __init__(self, overrides: dict[str, float | dict[str, float]] | None = None) -> None:
        self._overrides = overrides or {}
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "mock-jev"

    async def evaluate(self, state: dict[str, Any], questions: list[JudgeQuestion]) -> JudgeResult:
        result = JudgeResult(requests=1 if questions else 0)
        self.calls += result.requests
        for q in questions:
            result.answers[q.unit_id] = self._answer(state, q)
        return result

    def _answer(self, state: dict[str, Any], q: JudgeQuestion) -> RawAnswer:
        raw = RawAnswer(unit_id=q.unit_id, primitive=q.primitive, model=self.model_id)
        override = self._overrides.get(q.unit_id)
        seed = stable_hash(q.question_id, q.to_payload(), state)
        if q.primitive == "noul":
            raw.noul = float(override) if isinstance(override, (int, float)) else _unit_float(seed)
            return raw
        options = _options(q)
        if isinstance(override, dict):
            probs = {k: float(v) for k, v in override.items()}
        else:
            weights = [_unit_float(seed + str(i)) ** 4 for i in range(len(options))]
            total = sum(weights)
            probs = {opt: w / total for opt, w in zip(options, weights)}
        raw.probabilities = probs
        raw.confidence = spread_confidence(list(probs.values()))
        best = max(probs, key=probs.__getitem__)
        if q.primitive == "choice":
            raw.choice = best
        else:
            raw.score = sum(int(level) * prob for level, prob in probs.items())
        return raw


def _options(q: JudgeQuestion) -> list[str]:
    if q.primitive == "score":
        return [str(i) for i in range(len(q.criteria or []))]
    return list((q.criteria or {}).keys())


def _unit_float(seed: str) -> float:
    return int(stable_hash(seed)[:8], 16) / 0xFFFFFFFF
