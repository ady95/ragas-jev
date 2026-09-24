"""Evaluation orchestration (plan section 7.6).

load → PII mask → extract units → one JEV request per sample × metric
→ decisions → deterministic scoring → result JSON.
Escalation to LLM judges (Phase 5) is not wired in yet; the result already
reports how many decisions fall below each routing threshold.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from ragas_jev.checks.numeric import check_claim
from ragas_jev.config import Settings
from ragas_jev.judge import questions as Q
from ragas_jev.judge.base import Judge, JudgeQuestion, JudgeResult
from ragas_jev.preprocess.base import ExtractedUnit, Extractor
from ragas_jev.privacy.pii_guard import PiiMaskingError, mask_sample
from ragas_jev.schemas import METRICS, EvalUnit, MetricResult, RagSample, SampleResult, UnitKind, UnitRecord
from ragas_jev.scoring import metrics as M
from ragas_jev.scoring.uncertainty import to_decision

SupportMode = Literal["joint", "per_chunk"]


@dataclass
class _MetricOutput:
    result: MetricResult
    records: list[UnitRecord] = field(default_factory=list)
    usage: JudgeResult = field(default_factory=JudgeResult)


def _merge_usage(parts: Iterable[JudgeResult]) -> JudgeResult:
    total = JudgeResult()
    for part in parts:
        total.requests += part.requests
        total.input_tokens += part.input_tokens
        total.cached += part.cached
    return total


class Evaluator:
    def __init__(
        self,
        judge: Judge,
        extractor: Extractor,
        *,
        threshold: float = 0.5,
        conf_accept: float = 0.85,
        conf_audit: float = 0.60,
        pii_masking: bool = True,
        custom_pii_terms: list[str] | None = None,
        chunk_relevance_version: str = Q.CHUNK_RELEVANCE_V2,
        claim_support_version: str = Q.CLAIM_SUPPORT,
        statement_relevance_version: str = Q.STATEMENT_RELEVANCE_V2,
        answer_scale_version: str | None = Q.ANSWER_RELEVANCE_SCALE_V2,
        support_mode: SupportMode = "joint",
        state_char_budget: int = 20_000,
        numeric_cap: float = 0.2,
    ) -> None:
        self.judge = judge
        self.extractor = extractor
        self.threshold = threshold
        self.conf_accept = conf_accept
        self.conf_audit = conf_audit
        self.pii_masking = pii_masking
        self.custom_pii_terms = custom_pii_terms or []
        self.chunk_relevance_version = chunk_relevance_version
        self.claim_support_version = claim_support_version
        self.statement_relevance_version = statement_relevance_version
        # None skips the whole-answer Score question (e.g. for judges without Score support).
        self.answer_scale_version = answer_scale_version
        # joint: all contexts in one state (split only past state_char_budget);
        # per_chunk: one state per chunk. Either way a claim's p is the max over states.
        self.support_mode = support_mode
        self.state_char_budget = state_char_budget
        # Cap applied to p of claims whose numbers are absent from the contexts,
        # for the diagnostic `*_numeric_guarded` score only.
        self.numeric_cap = numeric_cap

    @classmethod
    def from_settings(cls, settings: Settings, judge: Judge, extractor: Extractor) -> Evaluator:
        return cls(
            judge,
            extractor,
            threshold=settings.ragas_jev_decision_threshold,
            conf_accept=settings.ragas_jev_conf_accept,
            conf_audit=settings.ragas_jev_conf_audit,
            pii_masking=settings.ragas_jev_pii_masking,
            state_char_budget=settings.jev_state_char_budget,
        )

    async def evaluate(
        self,
        samples: Iterable[RagSample],
        metrics: tuple[str, ...] = METRICS,
        *,
        concurrency: int = 4,
        on_result: Callable[[SampleResult], Awaitable[None] | None] | None = None,
    ) -> list[SampleResult]:
        semaphore = asyncio.Semaphore(concurrency)

        async def run(sample: RagSample) -> SampleResult:
            async with semaphore:
                result = await self.evaluate_sample(sample, metrics)
            if on_result is not None:
                maybe = on_result(result)
                if asyncio.iscoroutine(maybe):
                    await maybe
            return result

        return list(await asyncio.gather(*(run(s) for s in samples)))

    async def evaluate_sample(self, sample: RagSample, metrics: tuple[str, ...] = METRICS) -> SampleResult:
        result = SampleResult(sample_id=sample.sample_id, evaluation_model=self._model_info())
        if self.pii_masking:
            try:
                sample, pii_report = mask_sample(sample, self.custom_pii_terms)
            except PiiMaskingError:
                result.status = "blocked_pii"
                result.error = "PII masking failed; sample was not sent to any external service"
                return result
            result.evaluation_model["pii_masked"] = pii_report
        runners = {
            "context_precision": self._context_precision,
            "faithfulness": self._faithfulness,
            "context_recall": self._context_recall,
            "answer_relevancy": self._answer_relevancy,
        }
        unknown = set(metrics) - runners.keys()
        if unknown:
            raise ValueError(f"unknown metrics: {sorted(unknown)}")
        try:
            outputs = await asyncio.gather(*(runners[m](sample) for m in metrics))
        except Exception as exc:  # noqa: BLE001 - isolate per-sample failures
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        self._assemble(result, outputs)
        return result

    # --- metric runners -------------------------------------------------

    @staticmethod
    def _units(kind: UnitKind, extracted: list[ExtractedUnit]) -> list[EvalUnit]:
        return [
            EvalUnit(unit_id=f"{kind}_{i}", kind=kind, index=i, text=u.text, source_span=u.source_span)
            for i, u in enumerate(extracted)
        ]

    async def _judge(
        self, metric: str, state: dict, units: list[EvalUnit], questions: list[JudgeQuestion]
    ) -> tuple[list[UnitRecord], JudgeResult]:
        usage = await self.judge.evaluate(state, questions)
        records = []
        for unit, question in zip(units, questions):
            decision = to_decision(question, usage.answers[question.unit_id])
            records.append(UnitRecord(metric=metric, unit=unit, decision=decision))
        return records, usage

    def _context_groups(self, contexts: list[str]) -> list[list[str]]:
        if self.support_mode == "per_chunk":
            return [[c] for c in contexts]
        groups: list[list[str]] = []
        current: list[str] = []
        size = 0
        for context in contexts:
            if current and size + len(context) > self.state_char_budget:
                groups.append(current)
                current, size = [], 0
            current.append(context)
            size += len(context)
        groups.append(current)
        return groups

    async def _support(
        self, metric: str, contexts: list[str], units: list[EvalUnit], questions: list[JudgeQuestion]
    ) -> tuple[list[UnitRecord], JudgeResult]:
        """Judge claim support against the contexts; p = max over context groups."""
        groups = self._context_groups(contexts)
        results = await asyncio.gather(*(self.judge.evaluate({"contexts": g}, questions) for g in groups))
        records = []
        for unit, question in zip(units, questions):
            candidates = [to_decision(question, r.answers[question.unit_id]) for r in results]
            best = max(range(len(candidates)), key=lambda i: candidates[i].p)
            record = UnitRecord(metric=metric, unit=unit, decision=candidates[best])
            if len(groups) > 1:
                record.checks["support_groups"] = len(groups)
                record.checks["best_group"] = best
            numeric = check_claim(unit.text, contexts)
            if numeric.has_numbers:
                record.checks["numeric"] = numeric.as_dict()
            records.append(record)
        return records, _merge_usage(results)

    def _numeric_extras(self, prefix: str, records: list[UnitRecord]) -> dict[str, float | None]:
        with_numbers = [r for r in records if "numeric" in r.checks]
        mismatched = [r for r in with_numbers if r.checks["numeric"]["missing"]]
        guarded = [
            min(r.decision.p, self.numeric_cap) if r.checks.get("numeric", {}).get("missing") else r.decision.p
            for r in records
        ]
        return {
            f"{prefix}_numeric_mismatch_ratio": len(mismatched) / len(with_numbers) if with_numbers else None,
            f"{prefix}_numeric_guarded": sum(guarded) / len(guarded) if guarded else None,
        }

    async def _context_precision(self, s: RagSample) -> _MetricOutput:
        units = [EvalUnit(unit_id=f"chunk_{i}", kind="chunk", index=i, text=c) for i, c in enumerate(s.contexts)]
        if not units:
            return _MetricOutput(M.context_precision([]))
        questions = [Q.chunk_relevance(u.unit_id, u.index, self.chunk_relevance_version) for u in units]
        state = {"question": s.question, "contexts": s.contexts}
        records, usage = await self._judge("context_precision", state, units, questions)
        decisions = [r.decision for r in records]
        return _MetricOutput(M.context_precision(decisions, self.threshold, self.conf_accept), records, usage)

    async def _faithfulness(self, s: RagSample) -> _MetricOutput:
        if not s.contexts:
            return _MetricOutput(MetricResult(name="faithfulness", score=None, reason="no_contexts"))
        units = self._units("claim", await self.extractor.claims(s.question, s.answer))
        if not units:
            return _MetricOutput(M.faithfulness([]))
        questions = [Q.claim_support(u.unit_id, u.text, self.claim_support_version) for u in units]
        records, usage = await self._support("faithfulness", s.contexts, units, questions)
        result = M.faithfulness([r.decision for r in records], self.threshold, self.conf_accept)
        result.extra.update(self._numeric_extras("faithfulness", records))
        return _MetricOutput(result, records, usage)

    async def _context_recall(self, s: RagSample) -> _MetricOutput:
        if not s.reference:
            return _MetricOutput(MetricResult(name="context_recall", score=None, reason="no_reference"))
        if not s.contexts:
            return _MetricOutput(MetricResult(name="context_recall", score=None, reason="no_contexts"))
        units = self._units("ref_claim", await self.extractor.ref_claims(s.question, s.reference))
        if not units:
            return _MetricOutput(M.context_recall([]))
        version = f"ref_claim_coverage.{self.claim_support_version.rsplit('.', 1)[-1]}"
        questions = [Q.ref_claim_coverage(u.unit_id, u.text, version) for u in units]
        records, usage = await self._support("context_recall", s.contexts, units, questions)
        result = M.context_recall([r.decision for r in records], self.threshold, self.conf_accept)
        result.extra.update(self._numeric_extras("context_recall", records))
        return _MetricOutput(result, records, usage)

    async def _answer_relevancy(self, s: RagSample) -> _MetricOutput:
        units = self._units("statement", await self.extractor.statements(s.question, s.answer))
        questions = [Q.statement_relevance(u.unit_id, u.text, self.statement_relevance_version) for u in units]
        if self.answer_scale_version is not None:
            units.append(EvalUnit(unit_id="answer_scale", kind="answer", index=0, text=s.answer))
            questions.append(Q.answer_relevance_scale("answer_scale", self.answer_scale_version))
        if not questions:
            return _MetricOutput(M.answer_relevancy([]))
        state = {"question": s.question, "answer": s.answer}
        records, usage = await self._judge("answer_relevancy", state, units, questions)
        statement_decisions = [r.decision for r in records if r.unit.kind == "statement"]
        scale = next((r.decision for r in records if r.unit.kind == "answer"), None)
        result = M.answer_relevancy(statement_decisions, scale, self.threshold, self.conf_accept)
        return _MetricOutput(result, records, usage)

    # --- assembly -------------------------------------------------------

    def _model_info(self) -> dict:
        return {
            "primary_judge": "JEV",
            "judge_model": self.judge.model_id,
            "preprocessor": self.extractor.version,
            "secondary_judge": None,
            "question_versions": [
                self.chunk_relevance_version,
                self.claim_support_version,
                self.statement_relevance_version,
                *([self.answer_scale_version] if self.answer_scale_version else []),
            ],
            "support_mode": self.support_mode,
            "thresholds": {
                "decision": self.threshold,
                "conf_accept": self.conf_accept,
                "conf_audit": self.conf_audit,
            },
        }

    def _assemble(self, result: SampleResult, outputs: list[_MetricOutput]) -> None:
        by_name = {o.result.name: o.result for o in outputs}
        for name, metric in by_name.items():
            result.confidence[name] = metric.confidence
            result.uncertainty[name] = metric.uncertainty
            if metric.score is None:
                result.reasons[name] = metric.reason or "not_computed"
        if cp := by_name.get("context_precision"):
            result.retrieval.update(
                context_precision=cp.score,
                context_precision_binary=cp.score_binary,
                context_precision_rank=cp.extra.get("context_precision_rank"),
                context_precision_ap=cp.extra.get("context_precision_ap"),
            )
        if cr := by_name.get("context_recall"):
            result.retrieval.update(context_recall=cr.score, context_recall_binary=cr.score_binary)
            result.retrieval.update({k: v for k, v in cr.extra.items() if k.startswith("context_recall_numeric")})
        if ff := by_name.get("faithfulness"):
            result.generation.update(faithfulness=ff.score, faithfulness_binary=ff.score_binary)
            result.generation.update({k: v for k, v in ff.extra.items() if k.startswith("faithfulness_numeric")})
        if ar := by_name.get("answer_relevancy"):
            result.generation.update(
                answer_relevancy=ar.score,
                answer_relevancy_binary=ar.score_binary,
                answer_relevancy_scale=ar.extra.get("answer_relevancy_scale"),
            )
        decisions = [r.decision for o in outputs for r in o.records if r.decision is not None]
        result.units = [r for o in outputs for r in o.records]
        result.escalation = {
            "total_decisions": len(decisions),
            "below_conf_accept": sum(d.confidence < self.conf_accept for d in decisions),
            "below_conf_audit": sum(d.confidence < self.conf_audit for d in decisions),
            "audited": 0,
            "strong_judged": 0,
            "human_review": 0,
        }
        result.usage = {
            "jev_requests": sum(o.usage.requests for o in outputs),
            "jev_input_tokens": sum(o.usage.input_tokens for o in outputs),
            "jev_cached_answers": sum(o.usage.cached for o in outputs),
        }
        result.status = "partial" if result.reasons else "ok"
