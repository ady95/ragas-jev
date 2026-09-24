"""Build a Context Recall validation set from MIRACL (synthetic references).

For each MIRACL dev query with >= 2 relevant passages:
  1. a generator model writes a reference answer from up to 3 relevant passages,
     one sentence per fact, and says which passage each sentence came from;
  2. three context variants are built with the same number of passages:
       full     all used relevant passages + distractors
       partial  one used relevant passage replaced by a distractor
       none     every relevant passage replaced by distractors
Distractors are passages MIRACL annotators judged non-relevant for the query.

Claim-level expectation: a reference claim should be supported iff the passage
its sentence came from is still in the contexts. This is noisy when passages
repeat each other's facts, which the report must keep in mind.

Requires benchmark/datasets/miracl_<lang>_dev_relevant (see prepare_miracl.py).
Usage:
  uv run python benchmark/prepare_recall.py --lang ko --queries 100
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

PROMPTS = {
    "ko": """질문에 대한 참고 답변(reference)을 한국어로 쓴다. 주어진 passages의 정보만 사용한다.
- 모든 passage에서 최소 한 문장씩 정보를 가져온다. 총 3~6문장.
- 각 문장은 passage 하나의 정보만 사용하고, 그 passage 번호(0부터)를 적는다.
- 숫자, 날짜, 이름 같은 구체적 사실을 포함한다. passages에 없는 정보는 추가하지 않는다.
JSON으로 반환: {"sentences": [{"text": "...", "passage": 0}]}""",
    "en": """Write a reference answer to the question in English, using only the given passages.
- Take at least one sentence of information from every passage; 3 to 6 sentences in total.
- Each sentence uses information from a single passage; give that passage's index (from 0).
- Include specific facts such as numbers, dates, and names. Add nothing that is not in the passages.
Return JSON: {"sentences": [{"text": "...", "passage": 0}]}""",
}


async def generate(client, model, sem, question: str, passages: list[str], lang: str) -> list[dict] | None:
    user = json.dumps({"question": question, "passages": passages}, ensure_ascii=False)
    for _ in range(3):
        try:
            async with sem:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": PROMPTS[lang]}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                )
            data = loads_first(r.choices[0].message.content)
            sentences = [
                {"text": str(s["text"]).strip(), "passage": int(s["passage"])}
                for s in data.get("sentences", [])
                if str(s.get("text", "")).strip() and 0 <= int(s.get("passage", -1)) < len(passages)
            ]
            if sentences and {s["passage"] for s in sentences} == set(range(len(passages))):
                return sentences
        except Exception as exc:  # noqa: BLE001
            print(f"  retry after {type(exc).__name__}", file=sys.stderr)
    return None


async def build(args: argparse.Namespace) -> None:
    settings = get_settings()
    client = AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key.get_secret_value())
    sem = asyncio.Semaphore(args.concurrency)
    src = Path(f"benchmark/datasets/miracl_{args.lang}_dev_relevant")
    samples = [json.loads(l) for l in (src / "samples.jsonl").open(encoding="utf-8")]
    labels = {d["sample_id"]: d["labels"] for d in map(json.loads, (src / "labels.jsonl").open(encoding="utf-8"))}
    eligible = [s for s in samples if sum(labels[s["sample_id"]]) >= 2 and len(labels[s["sample_id"]]) - sum(labels[s["sample_id"]]) >= args.contexts]
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(eligible, min(args.queries, len(eligible))), key=lambda s: s["sample_id"])

    async def one(s: dict) -> list[tuple[dict, dict]] | None:
        y = labels[s["sample_id"]]
        relevant = [c for c, v in zip(s["contexts"], y) if v]
        distractors = [c for c, v in zip(s["contexts"], y) if not v]
        used = relevant[:3]
        sentences = await generate(client, args.model, sem, s["question"], used, args.lang)
        if sentences is None:
            print(f"  skip {s['sample_id']}: no usable reference", file=sys.stderr)
            return None
        # Reference text with each sentence's character span.
        parts, spans, pos = [], [], 0
        for sent in sentences:
            if parts:
                pos += 1
            spans.append({"start": pos, "end": pos + len(sent["text"]), "passage": sent["passage"]})
            parts.append(sent["text"])
            pos += len(sent["text"])
        reference = " ".join(parts)
        local = random.Random(f"{args.seed}-{s['sample_id']}")
        pool = local.sample(distractors, len(distractors))
        removed = local.randrange(len(used))
        variants = {
            "full": list(range(len(used))),
            "partial": [i for i in range(len(used)) if i != removed],
            "none": [],
        }
        out = []
        for variant, kept in variants.items():
            contexts = [used[i] for i in kept] + pool[: args.contexts - len(kept)]
            order = local.sample(range(len(contexts)), len(contexts))
            contexts = [contexts[i] for i in order]
            sample_id = f"{s['sample_id']}-{variant}"
            out.append(
                (
                    {"sample_id": sample_id, "question": s["question"], "answer": "", "reference": reference,
                     "contexts": contexts, "metadata": {"source": "miracl-recall-synthetic", "generator": args.model}},
                    {"sample_id": sample_id, "variant": variant, "kept_passages": kept,
                     "n_used_passages": len(used), "sentences": spans},
                )
            )
        return out

    async def safe(s: dict):
        try:
            return await one(s)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {s['sample_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None

    results = await asyncio.gather(*(safe(s) for s in picked))
    await client.close()
    out_dir = args.out or Path(f"benchmark/datasets/recall_miracl_{args.lang}")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as fs, (out_dir / "labels.jsonl").open("w", encoding="utf-8") as fl:
        for group in results:
            for sample, label in group or []:
                fs.write(json.dumps(sample, ensure_ascii=False) + "\n")
                fl.write(json.dumps(label, ensure_ascii=False) + "\n")
                n += 1
    print(f"{out_dir}: {n} samples from {sum(r is not None for r in results)}/{len(picked)} queries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=sorted(PROMPTS), default="ko")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--contexts", type=int, default=5)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", type=Path, default=None)
    asyncio.run(build(parser.parse_args()))


if __name__ == "__main__":
    main()
