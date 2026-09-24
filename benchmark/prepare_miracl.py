"""Convert MIRACL / NoMIRACL human relevance judgments into RagSample JSONL.

Source: https://huggingface.co/datasets/miracl/nomiracl (Apache-2.0).
NoMIRACL's `dev.relevant` subset has the same qrels as MIRACL dev, and its
per-language corpus holds the text of every judged passage.

Outputs (under --out, default benchmark/datasets/miracl_<lang>_<split>_<subset>/):
  samples.jsonl  one RagSample per query; contexts = judged passages
  labels.jsonl   {"sample_id", "labels": [0|1 per context]} (native-speaker judgments)

MIRACL passages have no retrieval rank, so contexts are put in a seeded random
order. Rank-aware precision is therefore not meaningful on this data.

Usage:
  uv run python benchmark/prepare_miracl.py --lang ko
  uv run python benchmark/prepare_miracl.py --lang en --max-queries 213
  uv run python benchmark/prepare_miracl.py --lang ko --subset non_relevant --max-queries 100
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://huggingface.co/datasets/miracl/nomiracl/resolve/main/data"
LANGS = {"ko": "korean", "en": "english"}
DOWNLOAD_DIR = Path(".cache/downloads/nomiracl")


def fetch(lang_dir: str, rel_path: str) -> Path:
    target = DOWNLOAD_DIR / lang_dir / rel_path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/{lang_dir}/{rel_path}"
        print(f"download {url}")
        with urllib.request.urlopen(url, timeout=300) as resp:
            target.write_bytes(resp.read())
    return target


def load_corpus(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {d["docid"]: d for d in map(json.loads, f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=sorted(LANGS), default="ko")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--subset", choices=["relevant", "non_relevant"], default="relevant")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    lang_dir = LANGS[args.lang]
    name = f"{args.split}.{args.subset}"
    corpus = load_corpus(fetch(lang_dir, "corpus.jsonl.gz"))
    topics = {}
    for line in fetch(lang_dir, f"topics/{name}.tsv").read_text(encoding="utf-8").splitlines():
        qid, query = line.split("\t", 1)
        topics[qid] = query
    judgments: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for line in fetch(lang_dir, f"qrels/{name}.tsv").read_text(encoding="utf-8").splitlines():
        qid, _, docid, rel = line.split()
        judgments[qid].append((docid, int(rel) > 0))

    rng = random.Random(args.seed)
    qids = sorted(q for q in judgments if q in topics)
    if args.max_queries is not None and args.max_queries < len(qids):
        qids = sorted(rng.sample(qids, args.max_queries))

    out = args.out or Path(f"benchmark/datasets/miracl_{args.lang}_{args.split}_{args.subset}")
    out.mkdir(parents=True, exist_ok=True)
    n_ctx = n_rel = 0
    with (out / "samples.jsonl").open("w", encoding="utf-8") as fs, (out / "labels.jsonl").open(
        "w", encoding="utf-8"
    ) as fl:
        for qid in qids:
            passages = [(d, rel) for d, rel in judgments[qid] if d in corpus]
            rng.shuffle(passages)
            contexts = [f"[{corpus[d]['title']}] {corpus[d]['text']}" for d, _ in passages]
            labels = [int(rel) for _, rel in passages]
            sample_id = f"miracl-{args.lang}-{qid}"
            sample = {
                "sample_id": sample_id,
                "question": topics[qid],
                "answer": "",
                "contexts": contexts,
                "metadata": {"source": "nomiracl", "lang": args.lang, "split": name, "docids": [d for d, _ in passages]},
            }
            fs.write(json.dumps(sample, ensure_ascii=False) + "\n")
            fl.write(json.dumps({"sample_id": sample_id, "labels": labels}) + "\n")
            n_ctx += len(labels)
            n_rel += sum(labels)
    print(f"{out}: {len(qids)} queries, {n_ctx} judged passages, {n_rel} relevant")


if __name__ == "__main__":
    main()
