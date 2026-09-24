"""Router + calibration + human review wired through the pipeline (mock judges only)."""

import pytest

from ragas_jev.audit.review_queue import apply_labels, export_review, read_labels
from ragas_jev.audit.router import Router, RoutingPolicy
from ragas_jev.judge.mock_jev import MockJudge
from ragas_jev.pipeline import Evaluator
from ragas_jev.preprocess.base import SentenceSplitExtractor
from ragas_jev.schemas import RagSample
from ragas_jev.scoring.calibration import Calibrator, IsotonicMap

SAMPLE = RagSample(
    sample_id="s1",
    question="아인슈타인은 어디서 태어났나?",
    answer="아인슈타인은 울름에서 태어났다. 그는 특허청에서 일했다. 그는 바이올린을 켰다.",
    contexts=["아인슈타인은 울름에서 태어났다.", "그는 베른 특허청에서 일했다."],
)


class NamedMock(MockJudge):
    def __init__(self, name, overrides=None, fail=False):
        super().__init__(overrides)
        self.name, self.fail, self.asked = name, fail, []

    @property
    def model_id(self):
        return self.name

    async def evaluate(self, state, questions):
        self.asked.extend(q.unit_id for q in questions)
        if self.fail:
            raise RuntimeError("proxy down")
        result = await super().evaluate(state, questions)
        for a in result.answers.values():
            a.model = self.name
        return result


def _evaluator(auditor, strong, jev_values, **kw):
    router = Router(auditor=auditor, strong=strong, policy=RoutingPolicy(accept=0.85, audit=0.60))
    return Evaluator(MockJudge(overrides=jev_values), SentenceSplitExtractor(), router=router, **kw)


async def test_units_are_routed_by_confidence_band():
    # claim_0 confident, claim_1 in the audit band, claim_2 in the strong band
    auditor = NamedMock("auditor", overrides={"claim_1": 0.05})
    strong = NamedMock("strong", overrides={"claim_2": 0.02})
    evaluator = _evaluator(auditor, strong, {"claim_0": 0.95, "claim_1": 0.7, "claim_2": 0.55})
    result = await evaluator.evaluate_sample(SAMPLE, ("faithfulness",))
    by_id = {r.unit.unit_id: r.decision for r in result.units}
    assert auditor.asked == ["claim_1"] and strong.asked == ["claim_2"]
    assert by_id["claim_0"].decision_source == "jev" and by_id["claim_0"].p == 0.95
    assert by_id["claim_1"].decision_source == "llm_auditor" and by_id["claim_1"].p == 0.05
    assert by_id["claim_1"].jev_p == 0.7 and by_id["claim_1"].auditor_model == "auditor"
    assert by_id["claim_2"].decision_source == "strong_judge" and by_id["claim_2"].needs_human_review
    assert result.generation["faithfulness"] == pytest.approx((0.95 + 0.05 + 0.02) / 3)
    assert result.escalation["audited"] == 1 and result.escalation["strong_judged"] == 1
    # both LLM verdicts are on the other side of 0.5 from JEV's, so both go to review
    assert result.escalation["human_review"] == 2
    assert result.evaluation_model["secondary_judge"] == "auditor"


async def test_only_disagreements_are_flagged_for_review():
    # claim_1: auditor disagrees with JEV (0.7 -> 0.3); claim_2: strong judge agrees (0.55 -> 0.9)
    auditor = NamedMock("auditor", overrides={"claim_1": 0.3})
    strong = NamedMock("strong", overrides={"claim_2": 0.9})
    evaluator = _evaluator(auditor, strong, {"claim_0": 0.95, "claim_1": 0.7, "claim_2": 0.55})
    result = await evaluator.evaluate_sample(SAMPLE, ("faithfulness",))
    d = {r.unit.unit_id: r.decision for r in result.units}
    assert d["claim_1"].needs_human_review
    assert d["claim_2"].decision_source == "strong_judge" and not d["claim_2"].needs_human_review


async def test_policy_bands():
    from ragas_jev.audit.router import DEFAULT_METRIC_POLICIES

    assert RoutingPolicy(0.0, 0.0).band(0.51) == "accept"  # routing off
    assert RoutingPolicy(0.6, 0.6).band(0.59) == "strong"  # strong only
    assert RoutingPolicy(0.65, 0.0).band(0.51) == "audit"  # auditor only
    assert DEFAULT_METRIC_POLICIES["context_recall"].band(0.5) == "accept"


async def test_routing_failure_keeps_jev_decision_and_flags_review():
    auditor = NamedMock("auditor", fail=True)
    evaluator = _evaluator(auditor, None, {"claim_0": 0.95, "claim_1": 0.7, "claim_2": 0.95})
    result = await evaluator.evaluate_sample(SAMPLE, ("faithfulness",))
    d = {r.unit.unit_id: r.decision for r in result.units}["claim_1"]
    assert result.status == "ok"
    assert d.decision_source == "jev" and d.p == 0.7 and d.needs_human_review
    assert result.escalation["routing_failures"] == 1


async def test_calibration_runs_before_routing():
    # JEV 0.7 is calibrated down to 0.95-confident "no", so it is not escalated
    cal = Calibrator({"claim_support.v1:ko": IsotonicMap((0.0, 0.7, 1.0), (0.01, 0.05, 0.99))})
    auditor = NamedMock("auditor")
    evaluator = _evaluator(auditor, None, {"claim_0": 0.99, "claim_1": 0.7, "claim_2": 0.99}, calibrator=cal)
    result = await evaluator.evaluate_sample(SAMPLE, ("faithfulness",))
    d = {r.unit.unit_id: r.decision for r in result.units}["claim_1"]
    assert auditor.asked == []
    assert d.calibrated and d.p == pytest.approx(0.05) and d.jev_p == 0.7


async def test_human_review_roundtrip_rescores(tmp_path):
    strong = NamedMock("strong", overrides={"claim_2": 0.4})
    evaluator = _evaluator(NamedMock("auditor"), strong, {"claim_0": 0.95, "claim_1": 0.95, "claim_2": 0.55})
    result = await evaluator.evaluate_sample(SAMPLE, ("faithfulness",))
    csv_path = tmp_path / "review.csv"
    assert export_review([result], csv_path) == 1
    text = csv_path.read_text(encoding="utf-8-sig")
    assert "claim_2" in text and "바이올린" in text
    csv_path.write_text(text.replace("strong,", "strong,").rstrip("\n").rstrip("\r") + "1\n", encoding="utf-8-sig")
    labels = read_labels(csv_path)
    assert labels == {("s1", "faithfulness", "claim_2"): 1}
    updated, applied = apply_labels(result, labels)
    rescored = evaluator.rescore(updated)
    d = {r.unit.unit_id: r.decision for r in rescored.units}["claim_2"]
    assert applied == 1 and d.decision_source == "human" and d.p == 1.0 and not d.needs_human_review
    assert rescored.generation["faithfulness"] == pytest.approx((0.95 + 0.95 + 1.0) / 3)
    assert rescored.escalation["human_labeled"] == 1 and rescored.escalation["human_review"] == 0


async def test_rescore_without_changes_is_identical():
    evaluator = Evaluator(MockJudge(), SentenceSplitExtractor())
    sample = SAMPLE.model_copy(update={"reference": "아인슈타인은 울름에서 태어났다."})
    result = await evaluator.evaluate_sample(sample)
    rescored = evaluator.rescore(result)
    assert rescored.retrieval == result.retrieval and rescored.generation == result.generation
