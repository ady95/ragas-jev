"""Versioned extraction prompts (plan section 7.1).

Change a prompt only by adding a new version; the version id is part of the
extraction cache key and of every result's `evaluation_model`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    id: str
    system: str


CLAIM_EXTRACTION_V1 = Prompt(
    id="claim_extraction.v1",
    system="""You decompose an answer into atomic factual claims so each can be fact-checked on its own.

Rules:
1. Each claim states exactly one fact that can be judged true or false by itself.
2. Resolve pronouns and omitted subjects using the answer and the question, so every claim is self-contained.
3. Copy numbers, dates, units, and names exactly as they appear in the answer.
4. Do not add information that is not in the answer, and do not judge whether a claim is true.
5. Skip content without a factual assertion: greetings, refusals such as "I don't know", questions,
   statements about the answer or its sources such as "Based on the passages" or
   "the passages do not mention X", and hedges without facts.
6. Write each claim in the same language as the answer.
7. For each claim give `source_span`: the shortest verbatim substring of the answer the claim comes from.

Input is JSON with `question` and `answer`.
Return JSON: {"claims": [{"text": "...", "source_span": "..."}]}
Return {"claims": []} when the answer makes no factual claims.""",
)

REF_CLAIM_EXTRACTION_V1 = Prompt(
    id="ref_claim_extraction.v1",
    system=CLAIM_EXTRACTION_V1.system.replace("an answer", "a reference answer"),
)

STATEMENT_EXTRACTION_V1 = Prompt(
    id="statement_extraction.v1",
    system="""You split an answer into statements so the relevance of each part to the question can be judged.

Rules:
1. Cover everything the answer says, including hedges, caveats, and remarks about sources.
2. Each statement expresses one point and is self-contained: resolve pronouns and omitted subjects.
3. Keep the wording close to the answer; do not add or judge information.
4. Write each statement in the same language as the answer.
5. For each statement give `source_span`: the shortest verbatim substring of the answer it comes from.

Input is JSON with `question` and `answer`.
Return JSON: {"statements": [{"text": "...", "source_span": "..."}]}""",
)
