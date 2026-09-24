"""Scorer tests built from the worked examples in RAGAS-JEV_설계방향.md."""

import pytest

from helpers import noul_decision, noul_decisions
from ragas_jev.scoring import metrics as M


def test_faithfulness_design_doc_example():
    # 7.3: (0.98 + 0.94 + 0.08) / 3 = 0.667
    result = M.faithfulness(noul_decisions("claim", [0.98, 0.94, 0.08]))
    assert result.score == pytest.approx(0.6667, abs=1e-4)
    assert result.score_binary == pytest.approx(2 / 3)
    assert result.n_units == 3


def test_context_precision_soft_and_rank_design_doc_example():
    chunks = noul_decisions("chunk", [0.98, 0.76, 0.08, 0.43, 0.91])
    result = M.context_precision(chunks)
    # 8.3: soft mean = 0.632
    assert result.score == pytest.approx(0.632)
    # 8.4: weights 1.0, 0.8, 0.6, 0.4, 0.2
    expected_rank = (0.98 * 1.0 + 0.76 * 0.8 + 0.08 * 0.6 + 0.43 * 0.4 + 0.91 * 0.2) / 3.0
    assert result.extra["context_precision_rank"] == pytest.approx(expected_rank)
    # binary relevance: T T F F T → AP = (1/1 + 2/2 + 3/5) / 3
    assert result.extra["context_precision_ap"] == pytest.approx((1 + 1 + 0.6) / 3)


def test_rank_weights_match_design_doc():
    assert M.rank_weights(5) == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2])


def test_rank_precision_rewards_relevant_chunks_on_top():
    top = M.context_precision(noul_decisions("c", [0.9, 0.9, 0.1, 0.1]))
    bottom = M.context_precision(noul_decisions("c", [0.1, 0.1, 0.9, 0.9]))
    assert top.score == bottom.score
    assert top.extra["context_precision_rank"] > bottom.extra["context_precision_rank"]


def test_empty_inputs_yield_null_scores_with_reason():
    assert M.faithfulness([]).score is None
    assert M.faithfulness([]).reason == "no_claims"
    assert M.context_recall([]).reason == "no_ref_claims"
    assert M.context_precision([]).reason == "no_contexts"


def test_confidence_summary():
    result = M.context_recall(noul_decisions("r", [0.98, 0.51]))
    assert result.confidence == pytest.approx((0.98 + 0.51) / 2)
    assert result.confidence_min == pytest.approx(0.51)
    assert result.uncertainty == pytest.approx(1 - (0.98 + 0.51) / 2)
    assert result.low_confidence_ratio == pytest.approx(0.5)


def test_scores_are_order_independent_bit_for_bit():
    ps = [0.1, 0.2, 0.3, 0.7, 0.123456789, 0.987654321]
    forward = M.faithfulness(noul_decisions("c", ps)).score
    backward = M.faithfulness(noul_decisions("c", list(reversed(ps)))).score
    assert forward == backward


def test_answer_relevancy_keeps_scale_separately():
    scale = noul_decision("answer_scale", 0.9)
    result = M.answer_relevancy(noul_decisions("s", [0.98, 0.96, 0.21, 0.93]), scale)
    assert result.score == pytest.approx((0.98 + 0.96 + 0.21 + 0.93) / 4)
    assert result.extra["answer_relevancy_scale"] == 0.9
