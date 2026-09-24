import pytest

from ragas_jev.privacy.pii_guard import PiiMasker, mask_sample
from ragas_jev.schemas import RagSample

# All values below are synthetic test fixtures, not real personal data.


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("문의: test.user@example.com 로 회신", "EMAIL"),
        ("주민번호 900101-1234567 확인", "RRN"),
        ("외국인등록번호 900101-5234567", "RRN"),
        ("연락처 010-1234-5678", "PHONE"),
        ("연락처 01012345678", "PHONE"),
        ("대표번호 02-123-4567", "PHONE"),
        ("카드 1234-5678-9012-3456 결제", "CARD"),
        ("사업자번호 123-45-67890", "BRN"),
        ("계좌 110-123-456789 입금", "ACCOUNT"),
    ],
)
def test_detects_each_pii_type(text, kind):
    masker = PiiMasker()
    masked = masker.mask(text)
    assert f"[PII_{kind}_1]" in masked
    assert masker.report == {kind: 1}


@pytest.mark.parametrize(
    "text",
    [
        "A사는 2025년에 매출 3조원을 기록했다.",
        "기준일은 2025-01-01 이다.",
        "영업이익은 5,000억원으로 20% 증가했다.",
        "아인슈타인은 1879년 독일 울름에서 태어났다.",
    ],
)
def test_leaves_ordinary_facts_untouched(text):
    assert PiiMasker().mask(text) == text


def test_same_value_gets_same_placeholder_across_sample_fields():
    sample = RagSample(
        sample_id="s1",
        question="010-1234-5678 번호의 가입자는?",
        answer="010-1234-5678 번호는 홍길동 고객입니다.",
        contexts=["고객 홍길동, 연락처 010-1234-5678", "다른 번호 010-9999-8888"],
        reference="홍길동",
    )
    masked, report = mask_sample(sample, custom_terms=["홍길동"])
    assert "010-1234-5678" not in masked.model_dump_json()
    assert "홍길동" not in masked.model_dump_json()
    assert masked.question.startswith("[PII_PHONE_1]")
    assert "[PII_PHONE_1]" in masked.answer and "[PII_PHONE_1]" in masked.contexts[0]
    assert "[PII_PHONE_2]" in masked.contexts[1]
    assert masked.reference == "[PII_TERM_1]"
    assert report == {"PHONE": 2, "TERM": 1}


def test_report_never_contains_values():
    masker = PiiMasker()
    masker.mask("010-1234-5678")
    assert "010" not in repr(masker)
    assert "010" not in str(masker.report)
