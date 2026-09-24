"""Context Precision set with reference answers, built from the Phase 3 recall sets.

ragas' ContextPrecision needs a reference answer, which MIRACL does not have.
The Phase 3 recall sets (prepare_recall.py) do: each has a synthetic reference
written from MIRACL-relevant passages, and contexts that mix those passages
with MIRACL-non-relevant ones in a fixed order. Every context therefore has a
native-speaker relevance label, and the ranking is fixed, so ragas'
rank-aware ContextPrecision and this project's chunk relevance can be compared
on the same definition.

Only the `full` and `partial` variants are used (`none` has no relevant context).

Outputs benchmark/datasets/cp_ref_miracl_<lang>/{samples,labels}.jsonl in the
format of prepare_miracl.py (labels = 0/1 per context, in context order).

Usage:
  uv run python benchmark/prepare_cp_reference.py --lang ko
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["ko", "en"], default="ko")
    args = parser.parse_args()

    miracl = Path(f"benchmark/datasets/miracl_{args.lang}_dev_relevant")
    relevant: dict[str, set[str]] = {}
    judged: dict[str, set[str]] = {}
    labels = {d["sample_id"]: d["labels"] for d in map(json.loads, (miracl / "labels.jsonl").open(encoding="utf-8"))}
    for s in map(json.loads, (miracl / "samples.jsonl").open(encoding="utf-8")):
        relevant[s["sample_id"]] = {c for c, y in zip(s["contexts"], labels[s["sample_id"]]) if y}
        judged[s["sample_id"]] = set(s["contexts"])

    src = Path(f"benchmark/datasets/recall_miracl_{args.lang}")
    out = Path(f"benchmark/datasets/cp_ref_miracl_{args.lang}")
    out.mkdir(parents=True, exist_ok=True)
    n = n_rel = n_ctx = 0
    with (out / "samples.jsonl").open("w", encoding="utf-8") as fs, (out / "labels.jsonl").open("w", encoding="utf-8") as fl:
        for s in map(json.loads, (src / "samples.jsonl").open(encoding="utf-8")):
            if s["sample_id"].endswith("-none"):
                continue
            query = s["sample_id"].rsplit("-", 1)[0]
            if not all(c in judged[query] for c in s["contexts"]):
                raise ValueError(f"{s['sample_id']}: context not found among MIRACL judged passages")
            y = [int(c in relevant[query]) for c in s["contexts"]]
            sample = {**s, "metadata": {**s["metadata"], "source": "miracl-recall-reference"}}
            fs.write(json.dumps(sample, ensure_ascii=False) + "\n")
            fl.write(json.dumps({"sample_id": s["sample_id"], "labels": y}) + "\n")
            n, n_rel, n_ctx = n + 1, n_rel + sum(y), n_ctx + len(y)
    print(f"{out}: {n} samples, {n_ctx} contexts, {n_rel} relevant")


if __name__ == "__main__":
    main()
