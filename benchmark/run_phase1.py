"""Phase 1 validation: JEV chunk relevance vs human labels (and an LLM judge).

Measures, per dataset:
  - unit-level agreement with human labels (accuracy / P / R / F1 / kappa, AUROC, ECE)
  - accuracy per confidence band (input for Phase 5 routing thresholds)
  - uncertainty quality (does high uncertainty predict errors?)
  - sample-level correlation of soft context precision with human precision
  - repeatability across uncached runs, latency, token cost
  - JEV vs LLM judge agreement

Usage:
  uv run python benchmark/prepare_miracl.py --lang ko   # etc., see that script
  uv run python benchmark/run_phase1.py --repeats 5 --llm-judge
Raw results go to .cache/phase1/, the report to docs/reports/phase1_context_precision.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats import auroc, binary_report, cohen_kappa, ece, pearson, percentile, reliability_table, spearman  # noqa: E402

from ragas_jev.audit.llm_judge import LlmJudge  # noqa: E402
from ragas_jev.config import get_settings  # noqa: E402
from ragas_jev.judge.jev_client import JevJudge  # noqa: E402
from ragas_jev.pipeline import Evaluator  # noqa: E402
from ragas_jev.preprocess.base import SentenceSplitExtractor  # noqa: E402
from ragas_jev.schemas import RagSample, SampleResult  # noqa: E402

JEV_PRICE_PER_MTOK = 0.042
RAW_DIR = Path(".cache/phase1")
DATASETS = ["miracl_ko_dev_relevant", "miracl_en_dev_relevant", "miracl_ko_dev_non_relevant"]


class TimedJudge:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.latencies: list[float] = []

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    async def evaluate(self, state, questions):
        started = perf_counter()
        result = await self.inner.evaluate(state, questions)
        self.latencies.append(perf_counter() - started)
        return result


def load(name: str) -> tuple[list[RagSample], dict[str, list[int]]]:
    root = Path("benchmark/datasets") / name
    samples = [RagSample.model_validate_json(line) for line in (root / "samples.jsonl").open(encoding="utf-8")]
    labels = {d["sample_id"]: d["labels"] for d in map(json.loads, (root / "labels.jsonl").open(encoding="utf-8"))}
    return samples, labels


CHUNK_QUESTION = "chunk_relevance.v1"


async def run_judge(judge: Any, samples: list[RagSample], concurrency: int) -> tuple[list[SampleResult], TimedJudge]:
    timed = TimedJudge(judge)
    settings = get_settings()
    evaluator = Evaluator.from_settings(settings, timed, SentenceSplitExtractor())
    evaluator.chunk_relevance_version = CHUNK_QUESTION
    results = await evaluator.evaluate(samples, ("context_precision",), concurrency=concurrency)
    return results, timed


def unit_table(results: list[SampleResult], labels: dict[str, list[int]]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        if r.status == "error":
            continue
        for rec in r.units:
            rows.append(
                {
                    "sample_id": r.sample_id,
                    "index": rec.unit.index,
                    "label": labels[r.sample_id][rec.unit.index],
                    "p": rec.decision.p,
                    "confidence": rec.decision.confidence,
                }
            )
    return rows


def analyze_units(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not rows:
        return {"error": "no successful units"}
    ys = [r["label"] for r in rows]
    ps = [r["p"] for r in rows]
    preds = [int(p >= threshold) for p in ps]
    out: dict[str, Any] = {"at_threshold": binary_report(ys, preds)}
    out["auroc"] = auroc(ps, ys)
    out["ece"] = ece(ps, ys) if any(ys) else None
    out["reliability"] = reliability_table(ps, ys)
    if any(ys):
        sweep = [(t / 100, binary_report(ys, [int(p >= t / 100) for p in ps])["f1"]) for t in range(5, 100, 5)]
        out["best_f1_threshold"], out["best_f1"] = max(sweep, key=lambda x: x[1])
    bands = [("conf ≥ 0.85", 0.85, 1.01), ("0.60 ≤ conf < 0.85", 0.60, 0.85), ("conf < 0.60", 0.0, 0.60)]
    out["bands"] = []
    for name, lo, hi in bands:
        idx = [i for i, r in enumerate(rows) if lo <= r["confidence"] < hi]
        acc = sum(preds[i] == ys[i] for i in idx) / len(idx) if idx else None
        out["bands"].append({"band": name, "n": len(idx), "share": len(idx) / len(rows), "accuracy": acc})
    errors = [int(preds[i] != ys[i]) for i in range(len(rows))]
    out["uncertainty_auroc"] = auroc([1 - r["confidence"] for r in rows], errors)
    return out


def analyze_samples(results: list[SampleResult], labels: dict[str, list[int]]) -> dict[str, Any]:
    ok = [r for r in results if r.status != "error"]
    if not ok:
        return {"n": 0}
    human = [sum(labels[r.sample_id]) / len(labels[r.sample_id]) for r in ok]
    soft = [r.retrieval["context_precision"] for r in ok]
    binary = [r.retrieval["context_precision_binary"] for r in ok]
    return {
        "n": len(ok),
        "human_mean": math.fsum(human) / len(human),
        "soft_mean": math.fsum(soft) / len(soft),
        "binary_mean": math.fsum(binary) / len(binary),
        "pearson_soft": pearson(soft, human),
        "spearman_soft": spearman(soft, human),
        "pearson_binary": pearson(binary, human),
        "mae_soft": math.fsum(abs(a - b) for a, b in zip(soft, human)) / len(ok),
        "any_chunk_relevant_rate": sum(any(rec.decision.p >= 0.5 for rec in r.units) for r in ok) / len(ok),
    }


def analyze_repeats(runs: list[list[dict[str, Any]]], threshold: float) -> dict[str, Any]:
    keyed = [{(r["sample_id"], r["index"]): r["p"] for r in run} for run in runs]
    keys = sorted(set.intersection(*(set(k) for k in keyed)))
    deltas, flips, identical = [], 0, 0
    for key in keys:
        values = [k[key] for k in keyed]
        deltas.append(max(values) - min(values))
        flips += len({v >= threshold for v in values}) > 1
        identical += len(set(values)) == 1
    return {
        "runs": len(runs),
        "units": len(keys),
        "identical_rate": identical / len(keys),
        "flip_rate": flips / len(keys),
        "max_delta": max(deltas),
        "mean_delta": math.fsum(deltas) / len(deltas),
    }


def usage_summary(results: list[SampleResult], timed: TimedJudge) -> dict[str, Any]:
    tokens = sum(r.usage.get("jev_input_tokens", 0) for r in results)
    lat = timed.latencies
    return {
        "requests": sum(r.usage.get("jev_requests", 0) for r in results),
        "input_tokens": tokens,
        "latency_p50": percentile(lat, 0.5) if lat else None,
        "latency_p95": percentile(lat, 0.95) if lat else None,
        "errors": sum(r.status == "error" for r in results),
        "error_examples": [r.error for r in results if r.error][:3],
    }


def raw_path(name: str, tag: str) -> Path:
    suffix = "" if CHUNK_QUESTION == "chunk_relevance.v1" else f".{CHUNK_QUESTION}"
    return RAW_DIR / f"{name}{suffix}.{tag}.jsonl"


def load_raw(name: str, tag: str) -> list[SampleResult]:
    return [SampleResult.model_validate_json(line) for line in raw_path(name, tag).open(encoding="utf-8")]


def save_raw(name: str, tag: str, results: list[SampleResult]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with raw_path(name, tag).open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    threshold = settings.ragas_jev_decision_threshold
    report: dict[str, Any] = {
        "date": date.today().isoformat(),
        "threshold": threshold,
        "chunk_question": CHUNK_QUESTION,
        "datasets": {},
    }
    for name in args.datasets:
        samples, labels = load(name)
        entry: dict[str, Any] = {"samples": len(samples), "units": sum(len(v) for v in labels.values())}
        print(f"== {name}: {entry['samples']} samples, {entry['units']} judged passages")

        runs, first_results, first_timed = [], None, None
        for i in range(args.repeats):
            if args.reuse_jev:
                results, timed = load_raw(name, f"jev.run{i}"), TimedJudge(None)
            else:
                jev = JevJudge.from_settings(settings)
                try:
                    results, timed = await run_judge(jev, samples, args.concurrency)
                finally:
                    await jev.aclose()
                save_raw(name, f"jev.run{i}", results)
            runs.append(unit_table(results, labels))
            if first_results is None:
                first_results, first_timed = results, timed
                entry["jev_model"] = next((u.decision.jev_model for r in results for u in r.units), None)
            print(f"   jev run {i + 1}/{args.repeats} {'loaded' if args.reuse_jev else 'done'}")
        entry["jev"] = {
            "units": analyze_units(runs[0], threshold),
            "samples": analyze_samples(first_results, labels),
            "usage": usage_summary(first_results, first_timed),
        }
        entry["jev"]["usage"]["cost_usd"] = entry["jev"]["usage"]["input_tokens"] * JEV_PRICE_PER_MTOK / 1e6
        if len(runs) > 1:
            entry["jev"]["repeatability"] = analyze_repeats(runs, threshold)

        if args.llm_judge:
            model = args.llm_model or settings.ragas_jev_baseline_judge_model
            llm = LlmJudge.from_settings(settings, model=model, max_concurrency=args.concurrency)
            try:
                results, timed = await run_judge(llm, samples, args.concurrency)
            finally:
                await llm.aclose()
            save_raw(name, f"llm.{llm.model_id}", results)
            llm_rows = unit_table(results, labels)
            entry["llm"] = {
                "model": llm.model_id,
                "units": analyze_units(llm_rows, threshold),
                "samples": analyze_samples(results, labels),
                "usage": usage_summary(results, timed),
            }
            jev_map = {(r["sample_id"], r["index"]): r["p"] for r in runs[0]}
            both = [(jev_map[(r["sample_id"], r["index"])], r["p"]) for r in llm_rows if (r["sample_id"], r["index"]) in jev_map]
            entry["jev_vs_llm"] = {} if not both else {
                "units": len(both),
                "kappa": cohen_kappa([int(a >= threshold) for a, _ in both], [int(b >= threshold) for _, b in both]),
                "pearson_p": pearson([a for a, _ in both], [b for _, b in both]),
                "agreement": sum((a >= threshold) == (b >= threshold) for a, b in both) / len(both),
            }
            print("   llm judge done")
        report["datasets"][name] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--llm-model", default=None, help="Default: RAGAS_JEV_BASELINE_JUDGE_MODEL")
    parser.add_argument("--chunk-question", default="chunk_relevance.v1")
    parser.add_argument("--reuse-jev", action="store_true", help="Load saved JEV runs instead of calling the API")
    parser.add_argument("--summary", type=Path, default=RAW_DIR / "summary.json")
    args = parser.parse_args()
    global CHUNK_QUESTION
    CHUNK_QUESTION = args.chunk_question
    report = asyncio.run(main_async(args))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary → {args.summary}")


if __name__ == "__main__":
    main()
