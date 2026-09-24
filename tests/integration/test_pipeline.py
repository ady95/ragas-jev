"""End-to-end pipeline runs with MockJudge and the offline sentence splitter."""

import json

import pytest

from ragas_jev.cache import AnswerCache
from ragas_jev.judge.cached import CachedJudge
from ragas_jev.judge.mock_jev import MockJudge
from ragas_jev.pipeline import Evaluator
from ragas_jev.preprocess.base import SentenceSplitExtractor
from ragas_jev.schemas import RagSample

SAMPLE = RagSample(
    sample_id="einstein",
    question="아인슈타인은 어디서 태어났나?",
    answer="아인슈타인은 1879년 독일 울름에서 태어났다. 이후 스위스 특허청에서 근무했다.",
    contexts=[
        "알베르트 아인슈타인은 1879년 3월 14일 독일 울름에서 태어났다.",
        "그는 1902년부터 스위스 베른의 특허청에서 일했다.",
    ],
    reference="아인슈타인은 독일 울름에서 태어났다.",
)


def _evaluator(judge=None) -> Evaluator:
    return Evaluator(judge or MockJudge(), SentenceSplitExtractor())


async def test_full_result_schema():
    result = await _evaluator().evaluate_sample(SAMPLE)
    assert result.status == "ok", result.error
    assert set(result.retrieval) == {
        "context_precision", "context_precision_binary", "context_precision_rank",
        "context_precision_ap", "context_recall", "context_recall_binary",
        "context_recall_numeric_mismatch_ratio", "context_recall_numeric_guarded",
    }
    assert set(result.generation) == {
        "faithfulness", "faithfulness_binary", "answer_relevancy",
        "answer_relevancy_binary", "answer_relevancy_scale",
        "faithfulness_numeric_mismatch_ratio", "faithfulness_numeric_guarded",
    }
    assert set(result.confidence) == set(result.uncertainty) == {
        "context_precision", "faithfulness", "context_recall", "answer_relevancy",
    }
    for value in [*result.retrieval.values(), *result.generation.values()]:
        assert value is None or 0.0 <= value <= 1.0
    # 2 chunks + 2 claims + 1 ref claim + 2 statements + 1 scale
    assert result.escalation["total_decisions"] == 8
    assert result.usage["jev_requests"] == 4  # one request per metric
    for record in result.units:
        d = record.decision
        assert d is not None and d.decision_source == "jev" and d.jev_model == "mock-jev"
    json.loads(result.model_dump_json())


async def test_pinned_judge_values_flow_into_scores():
    judge = MockJudge(overrides={"chunk_0": 0.99, "chunk_1": 0.06, "claim_0": 0.98, "claim_1": 0.10})
    result = await _evaluator(judge).evaluate_sample(SAMPLE, ("context_precision", "faithfulness"))
    assert result.retrieval["context_precision"] == pytest.approx((0.99 + 0.06) / 2)
    assert result.generation["faithfulness"] == pytest.approx((0.98 + 0.10) / 2)
    assert result.generation["faithfulness_binary"] == pytest.approx(0.5)
    assert "context_recall" not in result.retrieval


async def test_repeated_runs_are_bit_for_bit_identical():
    first = await _evaluator().evaluate_sample(SAMPLE)
    second = await _evaluator().evaluate_sample(SAMPLE)
    assert first.model_dump_json() == second.model_dump_json()


async def test_missing_reference_is_partial():
    result = await _evaluator().evaluate_sample(SAMPLE.model_copy(update={"reference": None}))
    assert result.status == "partial"
    assert result.reasons == {"context_recall": "no_reference"}
    assert result.retrieval["context_recall"] is None


async def test_pii_is_masked_before_reaching_the_judge():
    seen_states = []

    class SpyJudge(MockJudge):
        async def evaluate(self, state, questions):
            seen_states.append(json.dumps([state, [q.to_payload() for q in questions]], ensure_ascii=False))
            return await super().evaluate(state, questions)

    sample = SAMPLE.model_copy(update={"answer": "담당자 연락처는 010-1234-5678 입니다."})  # synthetic
    result = await _evaluator(SpyJudge()).evaluate_sample(sample)
    assert all("010-1234-5678" not in s for s in seen_states)
    assert "010-1234-5678" not in result.model_dump_json()
    assert result.evaluation_model["pii_masked"] == {"PHONE": 1}


async def test_judge_failure_is_isolated_per_sample():
    class BrokenJudge(MockJudge):
        async def evaluate(self, state, questions):
            raise RuntimeError("boom")

    results = await _evaluator(BrokenJudge()).evaluate([SAMPLE, SAMPLE.model_copy(update={"sample_id": "b"})])
    assert [r.status for r in results] == ["error", "error"]
    assert results[0].error == "RuntimeError: boom"


async def test_cache_answers_repeated_questions(tmp_path):
    inner = MockJudge()
    cache = AnswerCache(tmp_path / "cache.sqlite")
    evaluator = _evaluator(CachedJudge(inner, cache))
    first = await evaluator.evaluate_sample(SAMPLE)
    calls_after_first = inner.calls
    second = await evaluator.evaluate_sample(SAMPLE)
    cache.close()
    assert inner.calls == calls_after_first
    assert second.usage["jev_requests"] == 0
    assert second.usage["jev_cached_answers"] == first.escalation["total_decisions"]
    assert second.retrieval == first.retrieval and second.generation == first.generation


class _RecordingJudge(MockJudge):
    def __init__(self, overrides=None):
        super().__init__(overrides)
        self.states = []

    async def evaluate(self, state, questions):
        self.states.append(state)
        return await super().evaluate(state, questions)


async def test_per_chunk_mode_takes_max_over_chunks():
    judge = _RecordingJudge()
    evaluator = Evaluator(judge, SentenceSplitExtractor(), support_mode="per_chunk")
    result = await evaluator.evaluate_sample(SAMPLE, ("faithfulness",))
    assert [len(s["contexts"]) for s in judge.states] == [1, 1]
    record = result.units[0]
    assert record.checks["support_groups"] == 2
    assert result.usage["jev_requests"] == 2


async def test_joint_mode_splits_only_past_char_budget():
    judge = _RecordingJudge()
    evaluator = Evaluator(judge, SentenceSplitExtractor(), state_char_budget=10)
    await evaluator.evaluate_sample(SAMPLE, ("faithfulness",))
    assert len(judge.states) == 2  # each context alone exceeds the tiny budget
    judge = _RecordingJudge()
    await Evaluator(judge, SentenceSplitExtractor()).evaluate_sample(SAMPLE, ("faithfulness",))
    assert len(judge.states) == 1


async def test_numeric_check_flags_numbers_missing_from_contexts():
    sample = SAMPLE.model_copy(update={"answer": "아인슈타인은 1880년에 태어났다. 그는 특허청에서 일했다."})
    judge = MockJudge(overrides={"claim_0": 0.9, "claim_1": 0.9})
    result = await Evaluator(judge, SentenceSplitExtractor(), numeric_cap=0.2).evaluate_sample(sample, ("faithfulness",))
    first = result.units[0]
    assert first.checks["numeric"] == {"numbers": ["1880"], "missing": ["1880"]}
    assert "numeric" not in result.units[1].checks
    assert result.generation["faithfulness"] == pytest.approx(0.9)
    assert result.generation["faithfulness_numeric_mismatch_ratio"] == 1.0
    assert result.generation["faithfulness_numeric_guarded"] == pytest.approx((0.2 + 0.9) / 2)


async def test_source_span_is_kept_on_units():
    result = await _evaluator().evaluate_sample(SAMPLE, ("faithfulness",))
    assert all(r.unit.source_span and r.unit.source_span in SAMPLE.answer for r in result.units)


async def test_chunk_relevance_v3_uses_reference_and_falls_back_without_it():
    judge = _RecordingJudge()
    evaluator = Evaluator(judge, SentenceSplitExtractor(), chunk_relevance_version="chunk_relevance.v3")
    with_ref = await evaluator.evaluate_sample(SAMPLE, ("context_precision",))
    assert judge.states[-1]["reference"] == SAMPLE.reference
    assert {r.decision.question_id for r in with_ref.units} == {"chunk_relevance.v3"}
    no_ref = await evaluator.evaluate_sample(SAMPLE.model_copy(update={"reference": None}), ("context_precision",))
    assert "reference" not in judge.states[-1]
    assert {r.decision.question_id for r in no_ref.units} == {"chunk_relevance.v2"}
