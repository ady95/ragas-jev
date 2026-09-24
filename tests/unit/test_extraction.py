import pytest

from ragas_jev.preprocess.llm_extractor import ExtractionError, parse_units
from ragas_jev.preprocess.normalizer import dedupe, jaccard


def test_parse_units_shapes():
    wrapped = parse_units('{"claims": [{"text": "A", "source_span": "a"}]}', "claims")
    assert [(u.text, u.source_span) for u in wrapped] == [("A", "a")]
    assert [u.text for u in parse_units('["A", "B"]', "claims")] == ["A", "B"]
    assert [u.text for u in parse_units('{"items": [{"claim": "A"}]}', "claims")] == ["A"]
    assert parse_units('```json\n{"claims": []}\n```', "claims") == []


def test_parse_units_rejects_non_list():
    with pytest.raises(ExtractionError):
        parse_units('{"claims": "A"}', "claims")


def test_near_duplicates_are_removed():
    assert jaccard("아인슈타인은 울름에서 태어났다.", "아인슈타인은 울름에서 태어났다") == 1.0
    kept = dedupe(["Einstein was born in Ulm.", "Einstein was born in Ulm", "Einstein worked in Bern."])
    assert kept == ["Einstein was born in Ulm.", "Einstein worked in Bern."]
