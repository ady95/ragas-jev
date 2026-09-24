"""Input, intermediate, and result models (implementation plan section 6)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

UnitKind = Literal["claim", "statement", "ref_claim", "chunk", "answer"]
Primitive = Literal["noul", "score", "choice"]
DecisionSource = Literal["jev", "llm_auditor", "strong_judge", "human"]
SampleStatus = Literal["ok", "partial", "blocked_pii", "error"]

METRICS = ("context_precision", "faithfulness", "context_recall", "answer_relevancy")


class RagSample(BaseModel):
    sample_id: str
    question: str
    answer: str
    contexts: list[str]
    reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalUnit(BaseModel):
    unit_id: str
    kind: UnitKind
    index: int
    text: str
    source_span: str | None = None


class Decision(BaseModel):
    unit_id: str
    primitive: Primitive
    question_id: str
    distribution: dict[str, float]
    p: float
    confidence: float
    uncertainty: float
    entropy: float
    jev_model: str
    jev_request_id: str | None = None
    decision_source: DecisionSource = "jev"
    jev_p: float                     # raw JEV value, kept after calibration / escalation
    calibrated: bool = False         # p was mapped through a Calibrator
    auditor_model: str | None = None # set when an LLM judge replaced the JEV decision
    audit_rationale: str | None = None
    needs_human_review: bool = False


class UnitRecord(BaseModel):
    metric: str
    unit: EvalUnit
    decision: Decision | None = None
    # Deterministic side checks, e.g. {"numeric": {"numbers": [...], "missing": [...]}}
    checks: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    name: str
    score: float | None
    score_binary: float | None = None
    extra: dict[str, float | None] = Field(default_factory=dict)
    confidence: float | None = None
    confidence_min: float | None = None
    uncertainty: float | None = None
    low_confidence_ratio: float | None = None
    n_units: int = 0
    reason: str | None = None


class SampleResult(BaseModel):
    sample_id: str
    evaluation_model: dict[str, Any]
    retrieval: dict[str, float | None] = Field(default_factory=dict)
    generation: dict[str, float | None] = Field(default_factory=dict)
    confidence: dict[str, float | None] = Field(default_factory=dict)
    uncertainty: dict[str, float | None] = Field(default_factory=dict)
    escalation: dict[str, int] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    units: list[UnitRecord] = Field(default_factory=list)
    status: SampleStatus = "ok"
    reasons: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
