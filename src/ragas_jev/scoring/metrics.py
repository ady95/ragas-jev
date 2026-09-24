"""Deterministic metric engine (plan section 7.5).

Every function here is pure: decisions in, MetricResult out, no I/O.
Sums use math.fsum, which is exactly rounded, so results do not depend on
input order and are bit-for-bit reproducible.
"""

from __future__ import annotations

import math

from ragas_jev.schemas import Decision, MetricResult


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _summarize(
    name: str,
    decisions: list[Decision],
    threshold: float,
    conf_accept: float,
    *,
    reason_if_empty: str,
) -> MetricResult:
    if not decisions:
        return MetricResult(name=name, score=None, reason=reason_if_empty)
    ps = [d.p for d in decisions]
    confidences = [d.confidence for d in decisions]
    mean_conf = _mean(confidences)
    return MetricResult(
        name=name,
        score=_mean(ps),
        score_binary=sum(p >= threshold for p in ps) / len(ps),
        confidence=mean_conf,
        confidence_min=min(confidences),
        uncertainty=1.0 - mean_conf,
        low_confidence_ratio=sum(c < conf_accept for c in confidences) / len(confidences),
        n_units=len(decisions),
    )


def faithfulness(claims: list[Decision], threshold: float = 0.5, conf_accept: float = 0.85) -> MetricResult:
    return _summarize("faithfulness", claims, threshold, conf_accept, reason_if_empty="no_claims")


def context_recall(ref_claims: list[Decision], threshold: float = 0.5, conf_accept: float = 0.85) -> MetricResult:
    return _summarize("context_recall", ref_claims, threshold, conf_accept, reason_if_empty="no_ref_claims")


def rank_weights(k: int) -> list[float]:
    """Linear rank weights (K - i + 1) / K; K=5 gives 1.0, 0.8, 0.6, 0.4, 0.2."""
    return [(k - i) / k for i in range(k)]


def average_precision(relevant: list[bool]) -> float:
    """Original RAGAS context precision (AP@K over binary relevance)."""
    hits = 0
    total = []
    for i, is_rel in enumerate(relevant, start=1):
        if is_rel:
            hits += 1
            total.append(hits / i)
    return math.fsum(total) / hits if hits else 0.0


def context_precision(
    chunks_in_rank_order: list[Decision], threshold: float = 0.5, conf_accept: float = 0.85
) -> MetricResult:
    result = _summarize(
        "context_precision", chunks_in_rank_order, threshold, conf_accept, reason_if_empty="no_contexts"
    )
    if result.score is None:
        return result
    ps = [d.p for d in chunks_in_rank_order]
    weights = rank_weights(len(ps))
    result.extra["context_precision_rank"] = math.fsum(w * p for w, p in zip(weights, ps)) / math.fsum(weights)
    result.extra["context_precision_ap"] = average_precision([p >= threshold for p in ps])
    return result


def answer_relevancy(
    statements: list[Decision],
    scale: Decision | None = None,
    threshold: float = 0.5,
    conf_accept: float = 0.85,
) -> MetricResult:
    result = _summarize("answer_relevancy", statements, threshold, conf_accept, reason_if_empty="no_statements")
    result.extra["answer_relevancy_scale"] = scale.p if scale is not None else None
    return result
