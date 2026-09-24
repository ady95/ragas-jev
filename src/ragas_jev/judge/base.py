"""Judge interface shared by JEV, LLM judges, and the offline mock.

The interface mirrors the JEV request shape: one `state` evaluated against many
named questions in a single call (plan section 7.2.3).
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from ragas_jev.schemas import Primitive


class JudgeQuestion(BaseModel):
    unit_id: str
    question_id: str
    primitive: Primitive
    instructions: str | dict[str, Any]
    criteria: dict[str, Any] | list[Any] | None = None
    # For Choice questions: the option whose probability becomes `p`.
    positive_option: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.primitive, "instructions": self.instructions}
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        return payload


class RawAnswer(BaseModel):
    unit_id: str
    primitive: Primitive
    noul: float | None = None
    score: float | None = None
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    model: str
    request_id: str | None = None


class JudgeResult(BaseModel):
    answers: dict[str, RawAnswer] = Field(default_factory=dict)
    requests: int = 0
    input_tokens: int = 0
    cached: int = 0


class JudgeResponseError(ValueError):
    """The judge returned an answer that cannot be turned into a decision."""


class Judge(Protocol):
    @property
    def model_id(self) -> str: ...

    async def evaluate(self, state: dict[str, Any], questions: list[JudgeQuestion]) -> JudgeResult: ...
