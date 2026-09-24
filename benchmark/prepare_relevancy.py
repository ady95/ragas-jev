"""Build Answer Relevancy validation sets.

wikieval (English, 50 questions; https://huggingface.co/datasets/explodinggradients/WikiEval,
no license declared, used for internal evaluation only and never committed):
  good   the grounded answer
  poor   the answer with poor relevance (incomplete / drifting)   -> should score lower than good
  wrong  the ungrounded answer (on topic, factually wrong)        -> should score about the same as good

miracl (ko / en, synthetic; base answers are the Phase 3 references):
  base       the reference answer
  offtopic   base + one sentence taken from another question's reference   -> that sentence is irrelevant
  wrong      base with the fact that answers the question changed           -> relevance should not drop
  nonanswer  on-topic background that does not answer the question         -> should score low

Usage:
  uv run --group benchmark python benchmark/prepare_relevancy.py wikieval
  uv run python benchmark/prepare_relevancy.py miracl --lang ko
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

WIKIEVAL_URL = (
    "https://huggingface.co/datasets/explodinggradients/WikiEval/resolve/main/data/"
    "train-00000-of-00001-385c01e94624e9b7.parquet"
)
WIKIEVAL_FILE = Path(".cache/downloads/wikieval/train.parquet")
_PREFIX = re.compile(r"^\s*(Question|Answer)\s*:\s*", re.IGNORECASE)

WRONG_PROMPT = {
    "ko": "답변에서 질문에 직접 답하는 핵심 사실 하나만 틀린 값으로 바꿔라 (숫자, 날짜, 이름 등). 나머지 문장은 글자 그대로 둔다.",
    "en": "Change only the one fact in the answer that directly answers the question to a wrong value (a number, date, or name). Keep every other sentence exactly as it is.",
}
WRONG_SYSTEM = {
    "ko": 'JSON으로 반환: {"answer": "수정된 전체 답변", "changed_span": "바뀐 부분을 담은 가장 짧은 원문 부분 문자열"}',
    "en": 'Return JSON: {"answer": "the full modified answer", "changed_span": "the shortest verbatim substring containing the change"}',
}
NONANSWER_PROMPT = {
    "ko": """주어진 질문에 **답하지 않는** 2~3문장을 한국어로 쓴다. 질문과 같은 주제의 배경 정보만 passages에서 가져온다.
질문의 답이 되는 정보(질문이 묻는 사람, 장소, 숫자, 날짜 등)는 절대 쓰지 않는다. "모른다"는 말도 쓰지 않는다.
JSON으로 반환: {"answer": "..."}""",
    "en": """Write 2 to 3 sentences in English that do **not** answer the question. Use only background on the same topic from the passages.
Never state the information the question asks for (the person, place, number, date, and so on). Do not say that you do not know.
Return JSON: {"answer": "..."}""",
}


def clean(text: str) -> str:
    return _PREFIX.sub("", text or "").strip()


def write(out: Path, rows: list[tuple[dict, dict]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as fs, (out / "labels.jsonl").open("w", encoding="utf-8") as fl:
        for sample, label in rows:
            fs.write(json.dumps(sample, ensure_ascii=False) + "\n")
            fl.write(json.dumps(label, ensure_ascii=False) + "\n")
    print(f"{out}: {len(rows)} samples")


def build_wikieval(out: Path) -> None:
    import pyarrow.parquet as pq

    if not WIKIEVAL_FILE.exists():
        WIKIEVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(WIKIEVAL_URL, timeout=120) as resp:
            WIKIEVAL_FILE.write_bytes(resp.read())
    rows = []
    for i, r in enumerate(pq.read_table(WIKIEVAL_FILE).to_pylist()):
        group = f"wikieval-{i:02d}"
        for variant, column in (("good", "answer"), ("poor", "poor_answer"), ("wrong", "ungrounded_answer")):
            sample_id = f"{group}-{variant}"
            rows.append(
                (
                    {"sample_id": sample_id, "question": clean(r["question"]), "answer": clean(r[column]),
                     "contexts": list(r["context_v1"]), "metadata": {"source": "wikieval"}},
                    {"sample_id": sample_id, "group": group, "variant": variant},
                )
            )
    write(out, rows)


async def build_miracl(lang: str, model: str, concurrency: int, seed: int, out: Path) -> None:
    from openai import AsyncOpenAI

    from ragas_jev.config import get_settings
    from ragas_jev.llm_json import loads_first

    src = Path(f"benchmark/datasets/recall_miracl_{lang}")
    samples = {d["sample_id"]: d for d in map(json.loads, (src / "samples.jsonl").open(encoding="utf-8"))}
    labels = {d["sample_id"]: d for d in map(json.loads, (src / "labels.jsonl").open(encoding="utf-8"))}
    groups = sorted({sid.rsplit("-", 1)[0] for sid in samples})
    settings = get_settings()
    client = AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key.get_secret_value())
    sem = asyncio.Semaphore(concurrency)
    rng = random.Random(seed)

    async def chat(system: str, user: dict) -> dict:
        for _ in range(3):
            try:
                async with sem:
                    r = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                        response_format={"type": "json_object"},
                    )
                data = loads_first(r.choices[0].message.content)
                if isinstance(data, dict):
                    return data
            except Exception as exc:  # noqa: BLE001
                print(f"  retry after {type(exc).__name__}", file=sys.stderr)
        raise RuntimeError("no JSON object returned")

    async def one(i: int, group: str) -> list[tuple[dict, dict]]:
        full = samples[f"{group}-full"]
        question, reference = full["question"], full["reference"]
        sentences = labels[f"{group}-full"]["sentences"]
        out_rows = []

        def row(variant: str, answer: str, **extra) -> tuple[dict, dict]:
            sample_id = f"{group}-{variant}"
            return (
                {"sample_id": sample_id, "question": question, "answer": answer, "contexts": full["contexts"],
                 "metadata": {"source": "miracl-relevancy-synthetic", "generator": model}},
                {"sample_id": sample_id, "group": group, "variant": variant, **extra},
            )

        out_rows.append(row("base", reference))

        # offtopic: a sentence from another question's reference, inserted at a sentence boundary
        other = groups[(i + 1 + rng.randrange(len(groups) - 1)) % len(groups)]
        other_label = labels[f"{other}-full"]["sentences"][0]
        injected = samples[f"{other}-full"]["reference"][other_label["start"] : other_label["end"]]
        cut = rng.choice([s["end"] for s in sentences])
        answer = (reference[:cut] + " " + injected + reference[cut:]).strip()
        start = answer.find(injected)
        out_rows.append(row("offtopic", answer, spans=[{"start": start, "end": start + len(injected), "text": injected}]))

        try:
            data = await chat(WRONG_SYSTEM[lang], {"question": question, "answer": reference, "instruction": WRONG_PROMPT[lang]})
            wrong, span = str(data.get("answer", "")).strip(), str(data.get("changed_span", "")).strip()
            if wrong and wrong != reference and span and span in wrong:
                s = wrong.find(span)
                out_rows.append(row("wrong", wrong, spans=[{"start": s, "end": s + len(span), "text": span}]))
        except Exception as exc:  # noqa: BLE001
            print(f"  {group} wrong skipped: {exc}", file=sys.stderr)

        try:
            distractors = samples[f"{group}-none"]["contexts"]
            data = await chat(NONANSWER_PROMPT[lang], {"question": question, "passages": distractors})
            nonanswer = str(data.get("answer", "")).strip()
            if nonanswer:
                out_rows.append(row("nonanswer", nonanswer))
        except Exception as exc:  # noqa: BLE001
            print(f"  {group} nonanswer skipped: {exc}", file=sys.stderr)
        return out_rows

    results = await asyncio.gather(*(one(i, g) for i, g in enumerate(groups)))
    await client.close()
    write(out, [r for group in results for r in group])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["wikieval", "miracl"])
    parser.add_argument("--lang", choices=["ko", "en"], default="ko")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.source == "wikieval":
        build_wikieval(args.out or Path("benchmark/datasets/relevancy_wikieval"))
    else:
        asyncio.run(build_miracl(args.lang, args.model, args.concurrency, args.seed,
                                 args.out or Path(f"benchmark/datasets/relevancy_miracl_{args.lang}")))


if __name__ == "__main__":
    main()
