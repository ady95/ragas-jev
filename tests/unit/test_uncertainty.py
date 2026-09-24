import pytest

from ragas_jev.judge.base import JudgeQuestion, JudgeResponseError, RawAnswer
from ragas_jev.judge.questions import answer_relevance_scale
from ragas_jev.scoring.uncertainty import binary_confidence, normalized_entropy, spread_confidence, to_decision


def _noul_q():
    return JudgeQuestion(unit_id="u", question_id="q.v1", primitive="noul", instructions="?")


def test_binary_confidence_design_doc_cases():
    # 12: Case A vs Case B share the same verdict but not the same confidence
    assert binary_confidence(0.98) == pytest.approx(0.98)
    assert binary_confidence(0.51) == pytest.approx(0.51)
    assert binary_confidence(0.02) == pytest.approx(0.98)


def test_spread_confidence_matches_typesafe_docs_example():
    # docs: probabilities {0.88, 0.12, 0.0} → confidence 0.81 (approximation gives 0.82)
    assert spread_confidence([0.88, 0.12, 0.0]) == pytest.approx(0.82)
    assert spread_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert spread_confidence([1.0, 0.0]) == 1.0


def test_normalized_entropy_bounds():
    assert normalized_entropy([1.0, 0.0]) == 0.0
    assert normalized_entropy([0.5, 0.5]) == pytest.approx(1.0)


def test_noul_decision():
    d = to_decision(_noul_q(), RawAnswer(unit_id="u", primitive="noul", noul=0.94, model="jev-1.13.0"))
    assert d.p == 0.94
    assert d.distribution == {"true": 0.94, "false": pytest.approx(0.06)}
    assert d.confidence == 0.94
    assert d.uncertainty == pytest.approx(0.06)
    assert d.jev_p == d.p and d.decision_source == "jev"


def test_score_decision_expected_value_design_doc_example():
    # 10.3: (0×0.01 + 1×0.03 + 2×0.20 + 3×0.76) / 3
    probs = {"0": 0.01, "1": 0.03, "2": 0.20, "3": 0.76}
    raw = RawAnswer(unit_id="a", primitive="score", score=2.71, probabilities=probs, confidence=0.7, model="m")
    d = to_decision(answer_relevance_scale("a"), raw)
    assert d.p == pytest.approx(2.71 / 3)
    assert d.confidence == 0.7  # API-provided confidence is kept as-is


def test_choice_decision_uses_positive_option():
    q = JudgeQuestion(
        unit_id="c", question_id="q", primitive="choice", instructions="?",
        criteria={"supports": None, "contradicts": None}, positive_option="supports",
    )
    raw = RawAnswer(unit_id="c", primitive="choice", choice="contradicts",
                    probabilities={"supports": 0.3, "contradicts": 0.7}, confidence=0.4, model="m")
    assert to_decision(q, raw).p == pytest.approx(0.3)


def test_unnormalized_probabilities_are_renormalized():
    raw = RawAnswer(unit_id="a", primitive="score", probabilities={"0": 1.0, "1": 1.0, "2": 2.0, "3": 0.0},
                    confidence=0.2, model="m")
    d = to_decision(answer_relevance_scale("a"), raw)
    assert sum(d.distribution.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [None, float("nan"), 1.5, -0.2])
def test_invalid_noul_values_raise(value):
    with pytest.raises(JudgeResponseError):
        to_decision(_noul_q(), RawAnswer(unit_id="u", primitive="noul", noul=value, model="m"))
