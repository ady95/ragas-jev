import pytest

from ragas_jev.checks.numeric import check_claim, extract_numbers


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("A사는 2025년에 매출 3조원을 기록했다", [2025, 3e12]),
        ("영업이익은 5,000억원으로 20% 증가", [5e11, 20]),
        ("about $23.70 per hour or $49,400 per year", [23.7, 49400]),
        ("3.5 million users", [3.5e6]),
        ("$5k budget", [5000]),
        ("COVID-19 and B2B, v1.2", []),
        ("[PII_PHONE_1] 번호", []),
    ],
)
def test_extract_numbers(text, values):
    assert [t.value for t in extract_numbers(text)] == pytest.approx(values)


def test_check_claim_matches_values_across_formats():
    assert check_claim("매출은 3조원이다", ["매출 3조 원을 기록"]).missing == []
    assert check_claim("revenue was 5,000 dollars", ["revenue of 5000 dollars"]).missing == []
    assert check_claim("영업이익은 20% 증가했다", ["영업이익은 15% 증가"]).missing == ["20"]
    assert not check_claim("no numbers here", ["1, 2, 3"]).has_numbers
