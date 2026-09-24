"""Build a Korean hallucination-injection set from MIRACL-ko (synthetic labels).

No public Korean RAG dataset has human faithfulness labels, so this script:
  1. takes MIRACL-ko dev queries with their judged passages as contexts,
  2. asks a generator model for an answer grounded only in the relevant passages,
  3. asks it again to change exactly one fact in a copy of that answer
     (numeric value, entity swap, or an added unsupported sentence) and to
     report the changed span.

Labels are synthetic: the grounded answer is assumed faithful and the modified
copy has one known hallucinated span. The generator (default gpt-6-astra) is
kept different from the judge models to avoid self-agreement.

Requires benchmark/datasets/miracl_ko_dev_relevant (see prepare_miracl.py).
Usage:
  uv run python benchmark/prepare_ko_hallu.py --queries 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

from ragas_jev.config import get_settings
from ragas_jev.llm_json import loads_first

SRC = Path("benchmark/datasets/miracl_ko_dev_relevant")
KINDS = ["numeric", "entity", "fabricated"]

ANSWER_PROMPT = """질문에 한국어로 답하라. 반드시 주어진 passages에 있는 정보만 사용한다.
- 2~4문장으로 쓴다.
- passages에 숫자, 날짜, 이름 같은 구체적 사실이 있으면 포함한다.
- passages에 없는 정보는 절대 추가하지 않는다. 출처나 passages 자체를 언급하지 않는다.
JSON으로 반환: {"answer": "..."}"""

INJECT_PROMPTS = {
    "numeric": "답변에 있는 숫자나 날짜 하나만 다른 그럴듯한 값으로 바꿔라. 답변에 숫자나 날짜가 없으면 대신 이름이나 장소 하나를 바꿔라.",
    "entity": "답변에 있는 사람, 장소, 기관, 작품 이름 중 하나만 passages가 뒷받침하지 않는 다른 그럴듯한 이름으로 바꿔라.",
    "fabricated": "답변 끝에 passages에 없는 그럴듯한 구체적 사실 한 문장을 추가하라. 나머지는 그대로 둔다.",
}
INJECT_SYSTEM = """너는 fact-checking 평가용 데이터를 만든다. 주어진 답변을 지시대로 한 곳만 수정한다.
나머지 문장은 글자 그대로 유지한다.
JSON으로 반환: {"answer": "수정된 전체 답변", "changed_span": "수정된 답변에서 바뀐 부분을 담은 가장 짧은 원문 부분 문자열"}"""


async def chat(client: AsyncOpenAI, model: str, sem: asyncio.Semaphore, system: str, user: str) -> dict:
    for attempt in range(3):
        try:
            async with sem:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                )
            data = loads_first(r.choices[0].message.content)
            if isinstance(data, dict):
                return data
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"  retry after {type(exc).__name__}", file=sys.stderr)
    raise RuntimeError("no JSON object returned")


async def build(args: argparse.Namespace) -> None:
    settings = get_settings()
    client = AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key.get_secret_value())
    sem = asyncio.Semaphore(args.concurrency)
    samples = [json.loads(l) for l in (SRC / "samples.jsonl").open(encoding="utf-8")]
    labels = {d["sample_id"]: d["labels"] for d in map(json.loads, (SRC / "labels.jsonl").open(encoding="utf-8"))}
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(samples, args.queries), key=lambda s: s["sample_id"])

    async def one(i: int, s: dict) -> list[tuple[dict, dict]] | None:
        rel = [c for c, y in zip(s["contexts"], labels[s["sample_id"]]) if y][:3]
        non = [c for c, y in zip(s["contexts"], labels[s["sample_id"]]) if not y]
        contexts = rel + rng.sample(non, min(len(non), max(0, args.contexts - len(rel))))
        rng.shuffle(contexts)
        user = json.dumps({"question": s["question"], "passages": rel}, ensure_ascii=False)
        answer = (await chat(client, args.model, sem, ANSWER_PROMPT, user)).get("answer", "").strip()
        if not answer:
            return None
        kind = KINDS[i % len(KINDS)]
        inject_user = json.dumps(
            {"question": s["question"], "passages": rel, "answer": answer, "instruction": INJECT_PROMPTS[kind]},
            ensure_ascii=False,
        )
        data = await chat(client, args.model, sem, INJECT_SYSTEM, inject_user)
        modified, span = str(data.get("answer", "")).strip(), str(data.get("changed_span", "")).strip()
        start = modified.find(span) if span else -1
        if not modified or modified == answer or start < 0:
            print(f"  skip {s['sample_id']}: injection not usable", file=sys.stderr)
            return None
        base = {"question": s["question"], "contexts": contexts, "metadata": {"source": "miracl-ko-synthetic", "generator": args.model}}
        clean_id, hallu_id = f"{s['sample_id']}-clean", f"{s['sample_id']}-{kind}"
        return [
            ({**base, "sample_id": clean_id, "answer": answer}, {"sample_id": clean_id, "hallucinated": False, "spans": []}),
            (
                {**base, "sample_id": hallu_id, "answer": modified},
                {"sample_id": hallu_id, "hallucinated": True, "kind": kind,
                 "spans": [{"start": start, "end": start + len(span), "text": span, "label_type": kind}]},
            ),
        ]

    async def safe(i: int, s: dict) -> list[tuple[dict, dict]] | None:
        try:
            return await one(i, s)
        except Exception as exc:  # noqa: BLE001 - one bad query must not stop the build
            print(f"  skip {s['sample_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None

    results = await asyncio.gather(*(safe(i, s) for i, s in enumerate(picked)))
    await client.close()
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    with (args.out / "samples.jsonl").open("w", encoding="utf-8") as fs, (args.out / "labels.jsonl").open("w", encoding="utf-8") as fl:
        for pair in results:
            for sample, label in pair or []:
                fs.write(json.dumps(sample, ensure_ascii=False) + "\n")
                fl.write(json.dumps(label, ensure_ascii=False) + "\n")
                n += 1
    print(f"{args.out}: {n} responses from {sum(r is not None for r in results)}/{len(picked)} queries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--contexts", type=int, default=5, help="contexts per sample (relevant + distractors)")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out", type=Path, default=Path("benchmark/datasets/ko_hallu_miracl"))
    asyncio.run(build(parser.parse_args()))


if __name__ == "__main__":
    main()
