"""Simulate confidence-gated routing from saved Phase 1 runs (no API calls).

For each dataset it reports, per JEV confidence band, how accurate JEV and
each LLM judge are, and what the hybrid score would be if every JEV decision
below --accept were replaced by the LLM judge's decision (Phase 5 auditor).

Usage:
  uv run python benchmark/analyze_routing.py --models gpt-6-luna gpt-6-sol
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats import binary_report  # noqa: E402

RAW_DIR = Path(".cache/phase1")
DATASETS = ["miracl_ko_dev_relevant", "miracl_en_dev_relevant", "miracl_ko_dev_non_relevant"]


def load_decisions(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        for u in r["units"]:
            out[(r["sample_id"], u["unit"]["index"])] = (u["decision"]["p"], u["decision"]["confidence"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=["gpt-6-luna", "gpt-6-sol"])
    parser.add_argument("--question", default="chunk_relevance.v2")
    parser.add_argument("--accept", type=float, default=0.85)
    parser.add_argument("--audit", type=float, default=0.60)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t = args.threshold

    bands = [(f"≥ {args.accept}", args.accept, 2.0), (f"{args.audit}–{args.accept}", args.audit, args.accept),
             (f"< {args.audit}", 0.0, args.audit)]
    band_rows, hybrid_rows = [], []
    for name in args.datasets:
        labels = {d["sample_id"]: d["labels"] for d in map(json.loads, open(f"benchmark/datasets/{name}/labels.jsonl", encoding="utf-8"))}
        jev = load_decisions(RAW_DIR / f"{name}.{args.question}.jev.run0.jsonl")
        llms = {m: load_decisions(RAW_DIR / f"{name}.{args.question}.llm.{m}.jsonl") for m in args.models}
        keys = sorted(jev)
        y = {k: bool(labels[k[0]][k[1]]) for k in keys}

        for band, lo, hi in bands:
            ks = [k for k in keys if lo <= jev[k][1] < hi]
            accs = [sum((src[k][0] >= t) == y[k] for k in ks) / len(ks) if ks else None for src in (jev, *llms.values())]
            band_rows.append((name, band, len(ks), accs))

        def score(pick) -> dict:
            return binary_report([int(y[k]) for k in keys], [int(pick(k) >= t) for k in keys])

        escalated = sum(jev[k][1] < args.accept for k in keys) / len(keys)
        hybrid_rows.append((name, "JEV only", score(lambda k: jev[k][0])))
        for m, src in llms.items():
            hybrid_rows.append((name, f"JEV + {m}", score(lambda k, src=src: src[k][0] if jev[k][1] < args.accept else jev[k][0])))
        for m, src in llms.items():
            hybrid_rows.append((name, f"{m} only", score(lambda k, src=src: src[k][0])))
        hybrid_rows.append((name, f"(재검증 비율 {escalated * 100:.1f}%)", None))

    pct = lambda x: "–" if x is None else f"{x * 100:.1f}%"
    print("| 데이터셋 | JEV conf 구간 | unit 수 | JEV | " + " | ".join(args.models) + " |")
    print("|---|---|---|---|" + "---|" * len(args.models))
    for name, band, n, accs in band_rows:
        print(f"| {name} | {band} | {n} | " + " | ".join(pct(a) for a in accs) + " |")
    print(f"\n| 데이터셋 | 구성 (JEV conf < {args.accept}만 LLM으로 대체) | Accuracy | F1 | κ |")
    print("|---|---|---|---|---|")
    for name, label, b in hybrid_rows:
        if b is None:
            print(f"| {name} | {label} | | | |")
        elif b["tp"] + b["fn"]:
            print(f"| {name} | {label} | {b['accuracy']:.3f} | {b['f1']:.3f} | {b['kappa']:.3f} |")
        else:
            print(f"| {name} | {label} | {b['accuracy']:.3f} | – | – |")


if __name__ == "__main__":
    main()
