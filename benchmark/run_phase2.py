"""Phase 2 validation: Faithfulness against labeled hallucination spans.

Every configuration scores the *same* claims: extraction runs once per sample
through a shared cache, then each judge configuration re-judges those claims.

Config syntax: <judge>/<question>/<mode>
  judge    jev | llm:<model>
  question claim_support.v1 | claim_support.v2
  mode     joint | per_chunk

Usage:
  uv run python benchmark/run_phase2.py --datasets ragtruth_qa_test ko_hallu_miracl
Raw results: .cache/phase2/, summary: .cache/phase2/summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase1 import TimedJudge  # noqa: E402
from stats import auroc, binary_report, ece, percentile  # noqa: E402

from ragas_jev.audit.llm_judge import LlmJudge  # noqa: E402
from ragas_jev.cache import AnswerCache  # noqa: E402
from ragas_jev.config import get_settings  # noqa: E402
from ragas_jev.judge.jev_client import JevJudge  # noqa: E402
from ragas_jev.pipeline import Evaluator  # noqa: E402
from ragas_jev.preprocess.llm_extractor import LlmExtractor  # noqa: E402
from ragas_jev.schemas import RagSample, SampleResult  # noqa: E402

RAW_DIR = Path(".cache/phase2")
JEV_PRICE_PER_MTOK = 0.042
DEFAULT_CONFIGS = [
    "jev/claim_support.v1/joint",
    "jev/claim_support.v2/joint",
    "jev/claim_support.v2/per_chunk",
    "llm:gpt-6-sol/claim_support.v2/joint",
    "llm:gpt-6-luna/claim_support.v2/joint",
]


@dataclass(frozen=True)
class Config:
    judge: str
    question: str
    mode: str

    @classmethod
    def parse(cls, spec: str) -> Config:
        judge, question, mode = spec.split("/")
        return cls(judge, question, mode)

    @property
    def tag(self) -> str:
        return f"{self.judge.replace(':', '-')}.{self.question}.{self.mode}"


def load(name: str) -> tuple[list[RagSample], dict[str, dict]]:
    root = Path("benchmark/datasets") / name
    samples = [RagSample.model_validate_json(l) for l in (root / "samples.jsonl").open(encoding="utf-8")]
    labels = {d["sample_id"]: d for d in map(json.loads, (root / "labels.jsonl").open(encoding="utf-8"))}
    return samples, labels


# --- analysis -------------------------------------------------------------


def _overlaps(start: int, end: int, spans: list[dict]) -> bool:
    return any(start < s["end"] and s["start"] < end for s in spans)


def claim_rows(results: list[SampleResult], samples: dict[str, RagSample], labels: dict[str, dict]) -> list[dict]:
    rows = []
    for r in results:
        if r.status == "error":
            continue
        answer, spans = samples[r.sample_id].answer, labels[r.sample_id]["spans"]
        for rec in r.units:
            span = rec.unit.source_span or ""
            start = answer.find(span) if span else -1
            rows.append(
                {
                    "sample_id": r.sample_id,
                    "matched": start >= 0,
                    "hallucinated": start >= 0 and _overlaps(start, start + len(span), spans),
                    "p": rec.decision.p,
                    "confidence": rec.decision.confidence,
                    "numeric_mismatch": bool(rec.checks.get("numeric", {}).get("missing")),
                    "span": (start, start + len(span)) if start >= 0 else None,
                }
            )
    return rows


def analyze(results: list[SampleResult], samples: dict[str, RagSample], labels: dict[str, dict], t: float) -> dict:
    ok = [r for r in results if r.status != "error"]
    rows = claim_rows(ok, samples, labels)
    matched = [x for x in rows if x["matched"]]
    out: dict[str, Any] = {"responses": len(ok), "errors": len(results) - len(ok)}

    # Claim level: does 1 - p flag claims that overlap a labeled hallucination span?
    ys = [int(x["hallucinated"]) for x in matched]
    ps = [x["p"] for x in matched]
    out["claims"] = {
        "n": len(rows),
        "matched_rate": len(matched) / len(rows) if rows else None,
        "hallucinated": sum(ys),
        "auroc": auroc([1 - p for p in ps], ys),
        "at_threshold": binary_report(ys, [int(p < t) for p in ps]),
        "ece_support": ece(ps, [1 - y for y in ys]) if matched else None,
        "numeric_mismatch_claims": sum(x["numeric_mismatch"] for x in matched),
        "numeric_mismatch_hallucinated": sum(x["numeric_mismatch"] and x["hallucinated"] for x in matched),
    }

    # Response level: does the faithfulness score separate hallucinated answers?
    by_sample: dict[str, list[dict]] = {}
    for x in rows:
        by_sample.setdefault(x["sample_id"], []).append(x)
    y_resp, s_mean, s_min, s_guard, flag, flag_num = [], [], [], [], [], []
    for r in ok:
        claims = by_sample.get(r.sample_id, [])
        y_resp.append(int(labels[r.sample_id]["hallucinated"]))
        s_mean.append(1 - r.generation["faithfulness"] if r.generation.get("faithfulness") is not None else 0.0)
        s_min.append(1 - min((c["p"] for c in claims), default=1.0))
        guarded = r.generation.get("faithfulness_numeric_guarded")
        s_guard.append(1 - guarded if guarded is not None else 0.0)
        flag.append(int(any(c["p"] < t for c in claims)))
        flag_num.append(int(any(c["p"] < t or c["numeric_mismatch"] for c in claims)))
    out["responses_level"] = {
        "auroc_mean": auroc(s_mean, y_resp),
        "auroc_min": auroc(s_min, y_resp),
        "auroc_numeric_guarded": auroc(s_guard, y_resp),
        "any_unsupported": binary_report(y_resp, flag),
        "any_unsupported_or_numeric": binary_report(y_resp, flag_num),
        "zero_claim_responses": sum(1 for r in ok if not by_sample.get(r.sample_id)),
    }

    # Per injected kind (synthetic Korean set): recall of the any-unsupported flag.
    kinds: dict[str, list[int]] = {}
    for r, f in zip(ok, flag):
        kind = labels[r.sample_id].get("kind")
        if kind:
            kinds.setdefault(kind, []).append(f)
    if kinds:
        out["recall_by_kind"] = {k: sum(v) / len(v) for k, v in sorted(kinds.items())}

    # Extraction coverage: labeled spans touched by at least one claim span.
    total_spans = covered = 0
    for r in ok:
        spans = labels[r.sample_id]["spans"]
        claim_spans = [c["span"] for c in by_sample.get(r.sample_id, []) if c["span"]]
        for s in spans:
            total_spans += 1
            covered += any(a < s["end"] and s["start"] < b for a, b in claim_spans)
    out["span_coverage"] = covered / total_spans if total_spans else None
    out["claims_per_response"] = len(rows) / len(ok) if ok else None
    return out


# --- runs -----------------------------------------------------------------


async def extract_all(extractor: LlmExtractor, samples: list[RagSample], concurrency: int) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async def one(s: RagSample) -> None:
        async with sem:
            started = perf_counter()
            await extractor.claims(s.question, s.answer)
            latencies.append(perf_counter() - started)

    before = extractor.calls
    await asyncio.gather(*(one(s) for s in samples))
    fresh = extractor.calls - before
    return {"llm_calls": fresh, "latency_p50": percentile(latencies, 0.5) if fresh else None}


async def run_config(cfg: Config, samples: list[RagSample], extractor: LlmExtractor, concurrency: int):
    settings = get_settings()
    if cfg.judge == "jev":
        inner: Any = JevJudge.from_settings(settings)
    else:
        inner = LlmJudge.from_settings(settings, model=cfg.judge.split(":", 1)[1], max_concurrency=concurrency)
    timed = TimedJudge(inner)
    evaluator = Evaluator.from_settings(settings, timed, extractor)
    evaluator.claim_support_version = cfg.question
    evaluator.support_mode = cfg.mode  # type: ignore[assignment]
    try:
        results = await evaluator.evaluate(samples, ("faithfulness",), concurrency=concurrency)
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
            print(f"== {name}: {len(samples)} responses")
            entry: dict[str, Any] = {"responses": len(samples), "extraction": await extract_all(extractor, samples, args.concurrency)}
            print(f"   extraction done ({entry['extraction']['llm_calls']} new LLM calls)")
            for cfg in configs:
                results, timed = await run_config(cfg, samples, extractor, args.concurrency)
                with (RAW_DIR / f"{name}.{cfg.tag}.jsonl").open("w", encoding="utf-8") as f:
                    for r in results:
                        f.write(r.model_dump_json() + "\n")
                analysis = analyze(results, by_id, labels, settings.ragas_jev_decision_threshold)
                tokens = sum(r.usage.get("jev_input_tokens", 0) for r in results)
                analysis["usage"] = {
                    "requests": sum(r.usage.get("jev_requests", 0) for r in results),
                    "input_tokens": tokens,
                    "latency_p50": percentile(timed.latencies, 0.5) if timed.latencies else None,
                    "latency_p95": percentile(timed.latencies, 0.95) if timed.latencies else None,
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
        tags = [k for k in entry if k not in ("responses", "extraction")]
        out.append(f"### {name} ({entry['responses']} responses)\n")
        first = entry[tags[0]]
        out.append(
            f"추출: claim {f(first['claims_per_response'], 2)}개/응답, 원문 구간 일치 {f(first['claims']['matched_rate'])}, "
            f"라벨 구간 커버리지 {f(first['span_coverage'])}, claim 0개 응답 {first['responses_level']['zero_claim_responses']}개\n"
        )
        out.append("| 구성 | 응답 AUROC (mean) | 응답 AUROC (min) | 응답 P / R / F1 (claim p<0.5 존재) | +수치검사 F1 | claim AUROC | claim P / R / F1 | p50 지연(s) | 비용(USD) | 오류 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for tag in tags:
            a = entry[tag]
            rl, cl = a["responses_level"], a["claims"]["at_threshold"]
            au = rl["any_unsupported"]
            out.append(
                f"| {tag} | {f(rl['auroc_mean'])} | {f(rl['auroc_min'])} | {f(au['precision'])} / {f(au['recall'])} / {f(au['f1'])} "
                f"| {f(rl['any_unsupported_or_numeric']['f1'])} | {f(a['claims']['auroc'])} | {f(cl['precision'])} / {f(cl['recall'])} / {f(cl['f1'])} "
                f"| {f(a['usage']['latency_p50'], 2)} | {f(a['usage']['cost_usd'], 4)} | {a['errors']} |"
            )
        kinds = {tag: entry[tag].get("recall_by_kind") for tag in tags if entry[tag].get("recall_by_kind")}
        if kinds:
            names = sorted(next(iter(kinds.values())))
            out.append("\n| 구성 | " + " | ".join(f"탐지율: {k}" for k in names) + " |")
            out.append("|---|" + "---|" * len(names))
            for tag, rec in kinds.items():
                out.append(f"| {tag} | " + " | ".join(f"{rec[k] * 100:.1f}%" for k in names) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ragtruth_qa_test", "ko_hallu_miracl"])
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
