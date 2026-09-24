"""JevJudge against a mocked HTTP transport: request shape and answer conversion."""

import json

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from ragas_jev.judge import questions as Q
from ragas_jev.judge.base import JudgeResponseError
from ragas_jev.judge.jev_client import JevJudge


def _client(handler) -> AsyncTypeSafeClient:
    return AsyncTypeSafeClient(
        api_key="test-key",
        base_url="https://jev.test",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),
    )


def _response(answers: dict, request_id: str = "req_1") -> httpx2.Response:
    body = {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 5}}
    return httpx2.Response(200, json=body, headers={"x-typesafe-request-id": request_id})


async def test_sends_one_request_with_all_questions_and_converts_answers():
    seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return _response(
            {
                "claim_0": {"type": "noul", "noul": 0.98},
                "answer_scale": {
                    "type": "score", "score": 2.8, "confidence": 0.9,
                    "legend": {"0": "a", "1": "b", "2": "c", "3": "d"},
                    "probabilities": {"0": 0.0, "1": 0.0, "2": 0.2, "3": 0.8},
                },
            }
        )

    judge = JevJudge(_client(handler), "jev-1.13.0")
    questions = [Q.claim_support("claim_0", "아인슈타인은 1879년에 태어났다."), Q.answer_relevance_scale("answer_scale")]
    result = await judge.evaluate({"contexts": ["..."]}, questions)
    await judge.aclose()

    assert len(seen) == 1
    body = seen[0]
    assert body["model"] == "jev-1.13.0"
    assert body["state"] == {"contexts": ["..."]}
    assert body["questions"]["claim_0"]["instructions"]["claim"] == "아인슈타인은 1879년에 태어났다."
    assert body["questions"]["answer_scale"]["criteria"] == Q.ANSWER_RELEVANCE_LEVELS

    assert result.requests == 1 and result.input_tokens == 100
    claim = result.answers["claim_0"]
    assert claim.noul == 0.98 and claim.model == "jev-1.13.0" and claim.request_id == "req_1"
    scale = result.answers["answer_scale"]
    assert scale.score == 2.8 and scale.probabilities == {"0": 0.0, "1": 0.0, "2": 0.2, "3": 0.8}
    assert scale.confidence == 0.9


async def test_splits_large_question_sets_into_batches():
    calls = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        qs = json.loads(request.content)["questions"]
        calls.append(len(qs))
        return _response({k: {"type": "noul", "noul": 0.5} for k in qs})

    judge = JevJudge(_client(handler), "jev-1.13.0", max_questions_per_request=2)
    result = await judge.evaluate({"question": "q", "contexts": list("abcde")},
                                  [Q.chunk_relevance(f"chunk_{i}", i) for i in range(5)])
    await judge.aclose()
    assert sorted(calls) == [1, 2, 2]
    assert result.requests == 3 and len(result.answers) == 5


async def test_missing_answer_raises():
    judge = JevJudge(_client(lambda r: _response({})), "jev-1.13.0")
    with pytest.raises(JudgeResponseError):
        await judge.evaluate({"question": "q", "contexts": ["a"]}, [Q.chunk_relevance("chunk_0", 0)])
    await judge.aclose()


async def test_duplicate_unit_ids_are_rejected():
    judge = JevJudge(_client(lambda r: _response({})), "jev-1.13.0")
    with pytest.raises(ValueError):
        await judge.evaluate({}, [Q.chunk_relevance("x", 0), Q.chunk_relevance("x", 1)])
    await judge.aclose()
