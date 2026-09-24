"""Turn raw judge answers into decisions with p / confidence / uncertainty (plan 7.3).

Noul answers carry no confidence, so it is derived as max(p, 1 - p).
Choice and Score answers keep the API-provided confidence, which measures how
spread out the distribution is and is on a different scale than Noul's.
"""

from __future__ import annotations

import math

from ragas_jev.judge.base import JudgeQuestion, JudgeResponseError, RawAnswer
from ragas_jev.schemas import Decision

_PROB_TOLERANCE = 1e-3


def binary_confidence(p: float) -> float:
    return max(p, 1.0 - p)


def spread_confidence(probs: list[float]) -> float:
    """Approximation of TypeSafe's confidence: (k * max - 1) / (k - 1), clamped to [0, 1]."""
    k = len(probs)
    if k < 2:
        return 1.0
    return min(1.0, max(0.0, (k * max(probs) - 1.0) / (k - 1)))


def normalized_entropy(probs: list[float]) -> float:
    k = len(probs)
    if k < 2:
        return 0.0
    h = -math.fsum(p * math.log(p) for p in probs if p > 0)
    return h / math.log(k)


def to_decision(question: JudgeQuestion, raw: RawAnswer) -> Decision:
    if raw.primitive == "noul":
        p = _check_unit_interval(raw.noul, raw.unit_id)
        distribution = {"true": p, "false": 1.0 - p}
        confidence = binary_confidence(p)
    else:
        distribution = _normalize(raw.probabilities, raw.unit_id)
        confidence = raw.confidence if raw.confidence is not None else spread_confidence(list(distribution.values()))
        confidence = _check_unit_interval(confidence, raw.unit_id)
        if raw.primitive == "score":
            max_level = len(distribution) - 1
            expected = raw.score if raw.score is not None else math.fsum(int(k) * v for k, v in distribution.items())
            p = min(1.0, max(0.0, expected / max_level)) if max_level > 0 else 0.0
        else:
            target = question.positive_option or raw.choice
            if target not in distribution:
                raise JudgeResponseError(f"{raw.unit_id}: option {target!r} missing from probabilities")
            p = distribution[target]
    return Decision(
        unit_id=raw.unit_id,
        primitive=raw.primitive,
        question_id=question.question_id,
        distribution=distribution,
        p=p,
        confidence=confidence,
        uncertainty=1.0 - confidence,
        entropy=normalized_entropy(list(distribution.values())),
        jev_model=raw.model,
        jev_request_id=raw.request_id,
        jev_p=p,
    )


def _check_unit_interval(value: float | None, unit_id: str) -> float:
    if value is None or math.isnan(value):
        raise JudgeResponseError(f"{unit_id}: missing or NaN value")
    if value < -_PROB_TOLERANCE or value > 1.0 + _PROB_TOLERANCE:
        raise JudgeResponseError(f"{unit_id}: value {value} outside [0, 1]")
    return min(1.0, max(0.0, value))


def _normalize(probs: dict[str, float] | None, unit_id: str) -> dict[str, float]:
    if not probs:
        raise JudgeResponseError(f"{unit_id}: missing probabilities")
    if any(math.isnan(v) or v < 0 for v in probs.values()):
        raise JudgeResponseError(f"{unit_id}: negative or NaN probability")
    total = math.fsum(probs.values())
    if total <= 0:
        raise JudgeResponseError(f"{unit_id}: probabilities sum to zero")
    if abs(total - 1.0) <= _PROB_TOLERANCE:
        return dict(probs)
    return {k: v / total for k, v in probs.items()}
