import pytest
from pydantic import ValidationError

from ragas_jev.audit.llm_judge import _parse


def test_parse_accepts_wrapped_object():
    answers = _parse('{"answers": [{"id": "chunk_0", "p_yes": 0.9}]}')
    assert [(a.id, a.p_yes) for a in answers] == [("chunk_0", 0.9)]


def test_parse_accepts_bare_list():
    # the proxy accepts json_schema but does not enforce it
    answers = _parse('[{"id": "chunk_0", "p_yes": 0.1}, {"id": "chunk_1", "p_yes": 0.7}]')
    assert [a.id for a in answers] == ["chunk_0", "chunk_1"]


def test_parse_accepts_id_keyed_objects():
    # gpt-6-luna ignores the schema and keys answers by question id
    answers = _parse('{"chunk_0": {"p_yes": 0.1}, "chunk_1": {"p_yes": 0.9}}')
    assert [(a.id, a.p_yes) for a in answers] == [("chunk_0", 0.1), ("chunk_1", 0.9)]
    assert [(a.id, a.p_yes) for a in _parse('{"chunk_0": 0.4}')] == [("chunk_0", 0.4)]


def test_parse_accepts_other_observed_shapes():
    assert [(a.id, a.p_yes) for a in _parse('{"answers": {"s_0": 0.99, "s_1": 0.1}}')] == [("s_0", 0.99), ("s_1", 0.1)]
    assert [a.id for a in _parse('{"results": [{"id": "s_0", "p_yes": 0.2}]}')] == ["s_0"]


def test_parse_rejects_malformed():
    with pytest.raises(ValidationError):
        _parse('{"answers": [{"id": "chunk_0"}]}')


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        content = '{"answers": [{"id": "chunk_0", "p_yes": 0.8}]}'
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10),
            model=kwargs["model"],
            id="resp_1",
        )


async def test_requests_never_send_temperature():
    # GPT-5 and later models reject `temperature`
    from types import SimpleNamespace

    from ragas_jev.audit.llm_judge import LlmJudge
    from ragas_jev.judge import questions as Q

    completions = _FakeCompletions()
    judge = LlmJudge(SimpleNamespace(chat=SimpleNamespace(completions=completions)), "gpt-6-sol")
    result = await judge.evaluate({"question": "q", "contexts": ["c"]}, [Q.chunk_relevance("chunk_0", 0)])
    assert result.answers["chunk_0"].noul == 0.8
    assert "temperature" not in completions.calls[0]
    assert completions.calls[0]["seed"] == 7
