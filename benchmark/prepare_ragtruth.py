"""Convert RAGTruth QA (test split) into RagSample JSONL with span labels.

Source: https://huggingface.co/datasets/wandb/RAGTruth-processed (MIT).
Expert annotators marked hallucinated character spans in each LLM answer.

Outputs (default benchmark/datasets/ragtruth_qa_test/):
  samples.jsonl  RagSample per response (contexts = passages)
  labels.jsonl   {"sample_id", "hallucinated": bool, "spans": [{"start", "end", "text", "label_type"}]}

Usage:
  uv run --group benchmark python benchmark/prepare_ragtruth.py --per-class 160
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/wandb/RAGTruth-processed/resolve/main/data/test-00000-of-00001.parquet"
DOWNLOAD = Path(".cache/downloads/ragtruth/test.parquet")
_PASSAGE_SPLIT = re.compile(r"\n\s*\n")


def load_rows() -> list[dict]:
    import pyarrow.parquet as pq

    if not DOWNLOAD.exists():
        DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
        print(f"download {URL}")
        with urllib.request.urlopen(URL, timeout=300) as resp:
            DOWNLOAD.write_bytes(resp.read())
    return pq.read_table(DOWNLOAD).to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=160, help="responses with / without hallucination")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("benchmark/datasets/ragtruth_qa_test"))
    args = parser.parse_args()

    rows = [r for r in load_rows() if r["task_type"] == "QA" and r["quality"] == "good"]
    for r in rows:
        r["spans"] = json.loads(r["hallucination_labels"] or "[]")
    positive = [r for r in rows if r["spans"]]
    negative = [r for r in rows if not r["spans"]]
    rng = random.Random(args.seed)
    picked = rng.sample(positive, min(args.per_class, len(positive))) + rng.sample(
        negative, min(args.per_class, len(negative))
    )
    picked.sort(key=lambda r: r["id"])

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "samples.jsonl").open("w", encoding="utf-8") as fs, (args.out / "labels.jsonl").open(
        "w", encoding="utf-8"
    ) as fl:
        for r in picked:
            sample_id = f"ragtruth-{r['id']}"
            contexts = [p.strip() for p in _PASSAGE_SPLIT.split(r["context"]) if p.strip()]
            sample = {
                "sample_id": sample_id,
                "question": r["query"],
                "answer": r["output"],
                "contexts": contexts,
                "metadata": {"source": "ragtruth", "model": r["model"]},
            }
            spans = [
                {"start": s["start"], "end": s["end"], "text": s["text"], "label_type": s["label_type"]}
                for s in r["spans"]
            ]
            fs.write(json.dumps(sample, ensure_ascii=False) + "\n")
            fl.write(json.dumps({"sample_id": sample_id, "hallucinated": bool(spans), "spans": spans}, ensure_ascii=False) + "\n")
    n_pos = sum(bool(r["spans"]) for r in picked)
    print(f"{args.out}: {len(picked)} responses ({n_pos} hallucinated, {len(picked) - n_pos} clean)")


if __name__ == "__main__":
    main()
