"""Phase 3 validation: Context Recall under controlled context removal.

Data comes from prepare_recall.py: each query has one reference and three
context variants (full / partial / none). Checks:
  - recall by variant: full should be high, none low, partial in between
  - separation: AUROC of the recall score for full vs none and full vs partial
  - monotonicity per query: recall(full) >= recall(partial) >= recall(none)
  - claim level: a reference claim is expected to be supported iff the passage
    its sentence came from is kept (AUROC / P / R / F1 of p)

Config syntax is the same as run_phase2.py: <judge>/<question>/<mode>.
Usage:
  uv run python benchmark/run_phase3.py --datasets recall_miracl_ko recall_miracl_en
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase1 import TimedJudge  # noqa: E402
from run_phase2 import Config  # noqa: E402
from stats import auroc, binary_report, ece, percentile  # noqa: E402

from ragas_jev.audit.llm_judge import LlmJudge  # noqa: E402
from ragas_jev.cache import AnswerCache  # noqa: E402
from ragas_jev.config import get_settings  # noqa: E402
from ragas_jev.judge.jev_client import JevJudge  # noqa: E402
from ragas_jev.pipeline import Evaluator  # noqa: E402
from ragas_jev.preprocess.llm_extractor import LlmExtractor  # noqa: E402
from ragas_jev.schemas import RagSample, SampleResult  # noqa: E402

RAW_DIR = Path(".cache/phase3")
JEV_PRICE_PER_MTOK = 0.042
VARIANTS = ("full", "partial", "none")
DEFAULT_CONFIGS = [
    "jev/claim_support.v1/joint",
    "llm:gpt-6-sol/claim_support.v1/joint",
    "llm:gpt-6-luna/claim_support.v1/joint",
]


def load(name: str) -> tuple[list[RagSample], dict[str, dict]]:
    root = Path("benchmark/datasets") / name
    samples = [RagSample.model_validate_json(l) for l in (root / "samples.jsonl").open(encoding="utf-8")]
    labels = {d["sample_id"]: d for d in map(json.loads, (root / "labels.jsonl").open(encoding="utf-8"))}
    return samples, labels


def query_id(sample_id: str) -> str:
    return sample_id.rsplit("-", 1)[0]


def expected_support(reference: str, span: str | None, label: dict) -> bool | None:
    """True if the claim's sentence came from a kept passage; None if the span is not found."""
    start = reference.find(span) if span else -1
    if start < 0:
        return None
    end = start + len(span)
    passages = {s["passage"] for s in label["sentences"] if start < s["end"] and s["start"] < end}
    return bool(passages & set(label["kept_passages"])) if passages else None


def analyze(results: list[SampleResult], samples: dict[str, RagSample], labels: dict[str, dict], t: float) -> dict:
    ok = [r for r in results if r.status != "error"]
    out: dict[str, Any] = {"samples": len(ok), "errors": len(results) - len(ok)}

    scores: dict[str, dict[str, float]] = defaultdict(dict)  # query -> variant -> recall
    for r in ok:
        value = r.retrieval.get("context_recall")
        if value is not None:
            scores[query_id(r.sample_id)][labels[r.sample_id]["variant"]] = value
    out["recall_by_variant"] = {
        v: (math.fsum(q[v] for q in scores.values() if v in q) / max(1, sum(v in q for q in scores.values())))
        for v in VARIANTS
    }
    complete = [q for q in scores.values() if all(v in q for v in VARIANTS)]
    out["queries_complete"] = len(complete)
    out["auroc_full_vs_none"] = auroc([q["full"] for q in complete] + [q["none"] for q in complete], [1] * len(complete) + [0] * len(complete))
    out["auroc_full_vs_partial"] = auroc([q["full"] for q in complete] + [q["partial"] for q in complete], [1] * len(complete) + [0] * len(complete))
    out["monotonic_rate"] = sum(q["full"] >= q["partial"] >= q["none"] for q in complete) / len(complete) if complete else None
    out["partial_drop_rate"] = sum(q["partial"] < q["full"] for q in complete) / len(complete) if complete else None

    ys, ps, per_variant = [], [], defaultdict(lambda: [[], []])
    unmatched = 0
    for r in ok:
        label, reference = labels[r.sample_id], samples[r.sample_id].reference or ""
        for rec in r.units:
            expected = expected_support(reference, rec.unit.source_span, label)
            if expected is None:
                unmatched += 1
                continue
            ys.append(int(expected))
            ps.append(rec.decision.p)
            per_variant[label["variant"]][0].append(int(expected))
            per_variant[label["variant"]][1].append(rec.decision.p)
    out["claims"] = {
        "n": len(ys) + unmatched,
        "unmatched": unmatched,
        "auroc": auroc(ps, ys),
        "at_threshold": binary_report(ys, [int(p >= t) for p in ps]),
        "ece": ece(ps, ys) if ys else None,
        # full: every claim should be supported; none: none should be
        "unsupported_rate_full": sum(p < t for p in per_variant["full"][1]) / max(1, len(per_variant["full"][1])),
        "supported_rate_none": sum(p >= t for p in per_variant["none"][1]) / max(1, len(per_variant["none"][1])),
        "partial_auroc": auroc(per_variant["partial"][1], per_variant["partial"][0]),
    }
    return out


async def extract_all(extractor: LlmExtractor, samples: list[RagSample], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    unique = {(s.question, s.reference): s for s in samples if s.reference}
    latencies: list[float] = []
    counts: list[int] = []

    async def one(s: RagSample) -> None:
        async with sem:
            started = perf_counter()
            units = await extractor.ref_claims(s.question, s.reference or "")
            latencies.append(perf_counter() - started)
            counts.append(len(units))

    before = extractor.calls
    await asyncio.gather(*(one(s) for s in unique.values()))
    fresh = extractor.calls - before
    return {
        "references": len(unique),
        "llm_calls": fresh,
        "claims_per_reference": sum(counts) / len(counts) if counts else None,
        "latency_p50": percentile(latencies, 0.5) if fresh else None,
    }


async def run_config(cfg: Config, samples: list[RagSample], extractor: LlmExtractor, concurrency: int):
    settings = get_settings()
    inner: Any = (
        JevJudge.from_settings(settings)
        if cfg.judge == "jev"
        else LlmJudge.from_settings(settings, model=cfg.judge.split(":", 1)[1], max_concurrency=concurrency)
    )
    timed = TimedJudge(inner)
    evaluator = Evaluator.from_settings(settings, timed, extractor)
    evaluator.claim_support_version = cfg.question
    evaluator.support_mode = cfg.mode  # type: ignore[assignment]
    try:
        results = await evaluator.evaluate(samples, ("context_recall",), concurrency=concurrency)
    finally:
        await inner.aclose()
    return results, timed


async def main_async(args: argparse.Namespace) -> dict:
    settings = get_settings()
    configs = [Config.parse(c) for c in args.configs]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = AnswerCache(RAW_DIR / "extractions.sqlite")
    extractor = LlmExtractor.from_settings(settings, cache)
    report: dict[str, Any] = {"date": date.today().isoformat(), "extractor": extractor.version, "datasets": {}}
    try:
        for name in args.datasets:
            samples, labels = load(name)
            by_id = {s.sample_id: s for s in samples}
            print(f"== {name}: {len(samples)} samples")
            entry: dict[str, Any] = {"samples": len(samples), "extraction": await extract_all(extractor, samples, args.concurrency)}
            print(f"   extraction done ({entry['extraction']['llm_calls']} new LLM calls)")
            for cfg in configs:
                results, timed = await run_config(cfg, samples, extractor, args.concurrency)
                with (RAW_DIR / f"{name}.{cfg.tag}.jsonl").open("w", encoding="utf-8") as f:
                    for r in results:
                        f.write(r.model_dump_json() + "\n")
                analysis = analyze(results, by_id, labels, settings.ragas_jev_decision_threshold)
                tokens = sum(r.usage.get("jev_input_tokens", 0) for r in results)
                analysis["usage"] = {
                    "input_tokens": tokens,
                    "latency_p50": percentile(timed.latencies, 0.5) if timed.latencies else None,
                    "cost_usd": tokens * JEV_PRICE_PER_MTOK / 1e6 if cfg.judge == "jev" else None,
                    "error_examples": [r.error for r in results if r.error][:3],
                }
                entry[cfg.tag] = analysis
                print(f"   {cfg.tag} done (errors {analysis['errors']})")
            report["datasets"][name] = entry
    finally:
        await extractor.aclose()
        cache.close()
    return report


def render(report: dict) -> str:
    f = lambda x, d=3: "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
    out = []
    for name, entry in report["datasets"].items():
        ex = entry["extraction"]
        tags = [k for k in entry if k not in ("samples", "extraction")]
        out.append(f"### {name} ({entry['samples']} samples, reference {ex['references']}개)\n")
        out.append(f"reference claim {f(ex['claims_per_reference'], 2)}개/reference\n")
        out.append("| 구성 | recall full | recall partial | recall none | AUROC full vs none | AUROC full vs partial | 단조 감소 비율 | partial에서 하락 비율 | p50 지연(s) | 비용(USD) | 오류 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for tag in tags:
            a = entry[tag]
            rv = a["recall_by_variant"]
            out.append(
                f"| {tag} | {f(rv['full'])} | {f(rv['partial'])} | {f(rv['none'])} | {f(a['auroc_full_vs_none'])} "
                f"| {f(a['auroc_full_vs_partial'])} | {f(a['monotonic_rate'])} | {f(a['partial_drop_rate'])} "
                f"| {f(a['usage']['latency_p50'], 2)} | {f(a['usage']['cost_usd'], 4)} | {a['errors']} |"
            )
        out.append("\n| 구성 | claim AUROC | claim P / R / F1 (지원됨 예측) | ECE | full에서 미지원 판정 | none에서 지원 판정 | partial claim AUROC | 구간 불일치 claim |")
        out.append("|---|---|---|---|---|---|---|---|")
        for tag in tags:
            c = entry[tag]["claims"]
            b = c["at_threshold"]
            out.append(
                f"| {tag} | {f(c['auroc'])} | {f(b['precision'])} / {f(b['recall'])} / {f(b['f1'])} | {f(c['ece'])} "
                f"| {c['unsupported_rate_full'] * 100:.1f}% | {c['supported_rate_none'] * 100:.1f}% | {f(c['partial_auroc'])} | {c['unmatched']} / {c['n']} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["recall_miracl_ko", "recall_miracl_en"])
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--summary", type=Path, default=RAW_DIR / "summary.json")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not args.render_only:
        report = asyncio.run(main_async(args))
        args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary → {args.summary}")
    print(render(json.loads(args.summary.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
