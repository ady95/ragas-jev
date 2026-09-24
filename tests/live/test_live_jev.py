"""Calls the real JEV API with synthetic data. Run with: pytest -m live"""

import pytest

from ragas_jev.config import Settings
from ragas_jev.judge.jev_client import JevJudge
from ragas_jev.pipeline import Evaluator
from ragas_jev.preprocess.base import SentenceSplitExtractor
from ragas_jev.schemas import RagSample

pytestmark = pytest.mark.live


async def test_live_synthetic_korean_sample():
    settings = Settings()
    if settings.typesafe_api_key is None:
        pytest.skip("TYPESAFE_API_KEY not set")
    judge = JevJudge.from_settings(settings)
    sample = RagSample(
        sample_id="live-einstein",
        question="아인슈타인은 어디서 태어났나?",
        answer="아인슈타인은 독일 울름에서 태어났다. 그는 영국 특허청에서 근무했다.",
        contexts=[
            "알베르트 아인슈타인은 1879년 3월 14일 독일 울름에서 태어났다.",
            "그는 1902년부터 스위스 베른의 특허청에서 일했다.",
        ],
        reference="아인슈타인은 독일 울름에서 태어났다.",
    )
    try:
        result = await Evaluator(judge, SentenceSplitExtractor()).evaluate_sample(sample)
    finally:
        await judge.aclose()
    assert result.status == "ok", result.error
    claims = {r.unit.text: r.decision.p for r in result.units if r.metric == "faithfulness"}
    assert claims["아인슈타인은 독일 울름에서 태어났다."] > 0.8
    assert claims["그는 영국 특허청에서 근무했다."] < 0.2
    chunks = [r.decision.p for r in result.units if r.metric == "context_precision"]
    assert chunks[0] > chunks[1]
    assert all(r.decision.jev_model.startswith("jev-") for r in result.units)
