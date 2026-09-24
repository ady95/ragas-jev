"""Versioned JEV question templates (plan section 7.2.4).

Instructions refer to `state` fields by name in backticks. Change a template's
wording only by adding a new version id, so old results stay comparable.
"""

from __future__ import annotations

from ragas_jev.judge.base import JudgeQuestion

CHUNK_RELEVANCE = "chunk_relevance.v1"
CLAIM_SUPPORT = "claim_support.v1"
REF_CLAIM_COVERAGE = "ref_claim_coverage.v1"
STATEMENT_RELEVANCE = "statement_relevance.v1"
ANSWER_RELEVANCE_SCALE = "answer_relevance_scale.v1"

ANSWER_RELEVANCE_LEVELS = [
    "Unrelated",
    "Partially relevant",
    "Mostly relevant",
    "Direct and complete",
]


CHUNK_RELEVANCE_V2 = "chunk_relevance.v2"
CHUNK_RELEVANCE_V3 = "chunk_relevance.v3"


def chunk_relevance(unit_id: str, index: int, version: str = CHUNK_RELEVANCE) -> JudgeQuestion:
    if version == CHUNK_RELEVANCE:
        return JudgeQuestion(
            unit_id=unit_id,
            question_id=CHUNK_RELEVANCE,
            primitive="noul",
            instructions=f"Is `contexts[{index}]` useful for answering `question`?",
        )
    if version == CHUNK_RELEVANCE_V2:
        # Aligned with MIRACL's annotation criterion: the passage must answer, not just share a topic.
        return JudgeQuestion(
            unit_id=unit_id,
            question_id=CHUNK_RELEVANCE_V2,
            primitive="noul",
            instructions=f"Does `contexts[{index}]` contain information that answers `question`?",
            criteria={
                "true": "The passage states facts that answer the question, fully or in part",
                "false": "The passage is only on the same topic or mentions the same entities without answering the question",
            },
        )
    if version == CHUNK_RELEVANCE_V3:
        # Uses the reference answer, like ragas' reference-based ContextPrecision.
        # The pipeline falls back to v2 for samples without a reference.
        return JudgeQuestion(
            unit_id=unit_id,
            question_id=CHUNK_RELEVANCE_V3,
            primitive="noul",
            instructions=f"Does `contexts[{index}]` contain information used in `reference`, the answer to `question`?",
            criteria={
                "true": "The passage states at least one fact that appears in or directly supports the reference answer",
                "false": "None of the reference answer's facts come from this passage",
            },
        )
    raise ValueError(f"unknown chunk relevance version: {version}")


CLAIM_SUPPORT_V2 = "claim_support.v2"
REF_CLAIM_COVERAGE_V2 = "ref_claim_coverage.v2"

_SUPPORT_CRITERIA_V2 = {
    "true": "Every part of the claim, including numbers, dates, and names, is stated in or directly implied by the contexts",
    "false": "Some part of the claim is missing from, different from, or contradicted by the contexts",
}


def _support(unit_id: str, claim: str, question_id: str) -> JudgeQuestion:
    version = question_id.rsplit(".", 1)[-1]
    if version not in ("v1", "v2"):
        raise ValueError(f"unknown support question version: {question_id}")
    return JudgeQuestion(
        unit_id=unit_id,
        question_id=question_id,
        primitive="noul",
        instructions={"claim": claim, "question": "Is `claim` fully supported by `contexts`?"},
        criteria=_SUPPORT_CRITERIA_V2 if version == "v2" else None,
    )


def claim_support(unit_id: str, claim: str, version: str = CLAIM_SUPPORT) -> JudgeQuestion:
    return _support(unit_id, claim, version)


def ref_claim_coverage(unit_id: str, claim: str, version: str = REF_CLAIM_COVERAGE) -> JudgeQuestion:
    return _support(unit_id, claim, version)


STATEMENT_RELEVANCE_V2 = "statement_relevance.v2"
ANSWER_RELEVANCE_SCALE_V2 = "answer_relevance_scale.v2"

ANSWER_RELEVANCE_LEVELS_V2 = [
    "Unrelated: does not address the question at all",
    "Partially relevant: touches the topic but leaves the question mostly unanswered",
    "Mostly relevant: answers the question but misses parts of it or adds off-topic content",
    "Direct and complete: answers every part of the question and stays on topic",
]


def statement_relevance(unit_id: str, statement: str, version: str = STATEMENT_RELEVANCE) -> JudgeQuestion:
    if version == STATEMENT_RELEVANCE:
        return JudgeQuestion(
            unit_id=unit_id,
            question_id=STATEMENT_RELEVANCE,
            primitive="noul",
            instructions={"statement": statement, "question": "Does `statement` directly address `question`?"},
        )
    if version == STATEMENT_RELEVANCE_V2:
        # v1 marked correct supporting details as irrelevant (Phase 0 smoke run).
        # v2 asks whether the statement helps answer the question, regardless of truth.
        return JudgeQuestion(
            unit_id=unit_id,
            question_id=STATEMENT_RELEVANCE_V2,
            primitive="noul",
            instructions={
                "statement": statement,
                "question": "Is `statement` relevant to answering `question`? Judge relevance only, not whether it is true.",
            },
            criteria={
                "true": "It answers the question, or gives information about the question's subject that helps answer it",
                "false": "It is about a different subject, or is filler that does not help answer the question",
            },
        )
    raise ValueError(f"unknown statement relevance version: {version}")


def answer_relevance_scale(unit_id: str, version: str = ANSWER_RELEVANCE_SCALE) -> JudgeQuestion:
    if version == ANSWER_RELEVANCE_SCALE:
        return JudgeQuestion(
            unit_id=unit_id,
            question_id=ANSWER_RELEVANCE_SCALE,
            primitive="score",
            instructions="How directly and completely does `answer` address `question`?",
            criteria=ANSWER_RELEVANCE_LEVELS,
        )
    if version == ANSWER_RELEVANCE_SCALE_V2:
        # v1 appeared to penalize factually wrong answers; relevance must not depend on truth.
        return JudgeQuestion(
            unit_id=unit_id,
            question_id=ANSWER_RELEVANCE_SCALE_V2,
            primitive="score",
            instructions=(
                "How directly and completely does `answer` address `question`? "
                "Judge only relevance and completeness, not whether the answer is factually correct."
            ),
            criteria=ANSWER_RELEVANCE_LEVELS_V2,
        )
    raise ValueError(f"unknown answer relevance scale version: {version}")
