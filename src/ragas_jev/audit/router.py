"""Confidence-gated escalation of JEV decisions (plan section 7.4).

  confidence >= accept          keep the JEV decision
  audit <= confidence < accept  re-judge with the LLM auditor
  confidence < audit            re-judge with the strong judge

accept == 0 disables routing; audit == accept leaves only the strong band;
audit == 0 leaves only the auditor band. A re-judged unit is flagged for human
review when the LLM verdict is on the other side of the decision threshold
from JEV's.

DEFAULT_METRIC_POLICIES come from the Phase 5 simulation
(docs/reports/phase5_routing_calibration.md): for each metric, the cheapest
policy within 0.01 kappa of the best one, on calibrated JEV probabilities.

Only Noul decisions are escalated (the LLM judges answer yes/no questions).
Escalated units of one request go to the LLM in a single call. If the LLM
call fails, the JEV decision is kept and flagged for human review, so a
routing failure never loses a sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ragas_jev.judge.base import Judge, JudgeQuestion
from ragas_jev.schemas import Decision, UnitRecord
from ragas_jev.scoring.uncertainty import binary_confidence, normalized_entropy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingPolicy:
    accept: float = 0.85
    audit: float = 0.60

    def band(self, confidence: float) -> str:
        if confidence >= self.accept:
            return "accept"
        return "audit" if confidence >= self.audit else "strong"


DEFAULT_METRIC_POLICIES: dict[str, RoutingPolicy] = {
    "context_precision": RoutingPolicy(accept=0.60, audit=0.60),  # strong judge only
    "faithfulness": RoutingPolicy(accept=0.65, audit=0.0),  # auditor only
    "context_recall": RoutingPolicy(accept=0.0, audit=0.0),  # no routing: calibration is enough
    "answer_relevancy": RoutingPolicy(accept=0.0, audit=0.0),  # not validated at unit level
}


@dataclass
class RoutingStats:
    audited: int = 0
    strong_judged: int = 0
    failed: int = 0
    llm_requests: int = 0
    llm_input_tokens: int = 0

    def add(self, other: RoutingStats) -> None:
        for name in ("audited", "strong_judged", "failed", "llm_requests", "llm_input_tokens"):
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class Router:
    auditor: Judge
    strong: Judge | None = None
    policy: RoutingPolicy = field(default_factory=RoutingPolicy)
    # Per-metric overrides, e.g. {"context_precision": RoutingPolicy(0.9, 0.7)}
    metric_policies: dict[str, RoutingPolicy] = field(default_factory=dict)
    decision_threshold: float = 0.5

    def policy_for(self, metric: str) -> RoutingPolicy:
        return self.metric_policies.get(metric, self.policy)

    async def route(
        self, metric: str, state: dict[str, Any], records: list[UnitRecord], questions: list[JudgeQuestion]
    ) -> RoutingStats:
        policy = self.policy_for(metric)
        by_band: dict[str, list[tuple[UnitRecord, JudgeQuestion]]] = {"audit": [], "strong": []}
        for record, question in zip(records, questions):
            d = record.decision
            if d is None or d.primitive != "noul":
                continue
            band = policy.band(d.confidence)
            if band != "accept":
                by_band[band].append((record, question))

        stats = RoutingStats()
        for band, items in by_band.items():
            if not items:
                continue
            judge = self.auditor if band == "audit" or self.strong is None else self.strong
            source = "llm_auditor" if judge is self.auditor else "strong_judge"
            try:
                result = await judge.evaluate(state, [q for _, q in items])
            except Exception as exc:  # noqa: BLE001 - keep JEV, ask a human
                logger.warning("%s escalation failed for %s: %s", source, metric, exc)
                for record, _ in items:
                    record.decision = record.decision.model_copy(update={"needs_human_review": True})
                stats.failed += len(items)
                continue
            stats.llm_requests += result.requests
            stats.llm_input_tokens += result.input_tokens
            for record, question in items:
                answer = result.answers.get(question.unit_id)
                if answer is None or answer.noul is None:
                    record.decision = record.decision.model_copy(update={"needs_human_review": True})
                    stats.failed += 1
                    continue
                record.decision = self._merge(record.decision, answer.noul, answer.model, source, band, policy)
                if source == "llm_auditor":
                    stats.audited += 1
                else:
                    stats.strong_judged += 1
        return stats

    def _merge(
        self, jev: Decision, p: float, model: str, source: str, band: str, policy: RoutingPolicy
    ) -> Decision:
        confidence = binary_confidence(p)
        t = self.decision_threshold
        needs_review = (jev.p >= t) != (p >= t)
        return jev.model_copy(
            update={
                "p": p,
                "distribution": {"true": p, "false": 1.0 - p},
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
                "entropy": normalized_entropy([p, 1.0 - p]),
                "decision_source": source,
                "auditor_model": model,
                "needs_human_review": needs_review,
            }
        )
