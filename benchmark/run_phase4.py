"""Phase 4 validation: Answer Relevancy.

Datasets come from prepare_relevancy.py. Checks:
  wikieval  good should outscore poor (pairwise accuracy); wrong should score
            about the same as good (relevance must not depend on correctness)
  miracl    base vs offtopic / nonanswer separation (AUROC, pairwise accuracy),
            detection of the injected off-topic statement, false flags on base
            statements (strictness), and base vs wrong invariance

Every configuration scores the same statements (shared extraction cache).
Config syntax: <judge>/<statement question>/<scale question or ->
Usage:
  uv run python benchmark/run_phase4.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase1 import TimedJudge  # noqa: E402
from stats import auroc, percentile  # noqa: E402

from ragas_jev.audit.llm_judge import LlmJudge  # noqa: E402
from ragas_jev.cache import AnswerCache  # noqa: E402
from ragas_jev.config import get_settings  # noqa: E402
from ragas_jev.judge.jev_client import JevJudge  # noqa: E402
from ragas_jev.pipeline import Evaluator  # noqa: E402
from ragas_jev.preprocess.llm_extractor import LlmExtractor  # noqa: E402
from ragas_jev.schemas import RagSample, SampleResult  # noqa: E402

RAW_DIR = Path(".cache/phase4")
JEV_PRICE_PER_MTOK = 0.042
DEFAULT_DATASETS = ["relevancy_wikieval", "relevancy_miracl_ko", "relevancy_miracl_en"]
DEFAULT_CONFIGS = [
    "jev/statement_relevance.v1/answer_relevance_scale.v1",
    "jev/statement_relevance.v2/answer_relevance_scale.v2",
    "llm:gpt-6-sol/statement_relevance.v1/-",
    "llm:gpt-6-sol/statement_relevance.v2/-",
    "llm:gpt-6-luna/statement_relevance.v2/-",
]
SCORES = ("statement", "scale")


@dataclass(frozen=True)
class Config:
    judge: str
    statement: str
    scale: str | None

    @classmethod
    def parse(cls, spec: str) -> Config:
        judge, statement, scale = spec.split("/")
        return cls(judge, statement, None if scale == "-" else scale)

    @property
    def tag(self) -> str:
        return f"{self.judge.replace(':', '-')}.{self.statement}.{self.scale or 'noscale'}"


def load(name: str) -> tuple[list[RagSample], dict[str, dict]]:
    root = Path("benchmark/datasets") / name
    samples = [RagSample.model_validate_json(l) for l in (root / "samples.jsonl").open(encoding="utf-8")]
    labels = {d["sample_id"]: d for d in map(json.loads, (root / "labels.jsonl").open(encoding="utf-8"))}
    return samples, labels


def _pairwise(higher: list[float], lower: list[float]) -> float | None:
    if not higher:
        return None
    return sum(1.0 if h > l else 0.5 if h == l else 0.0 for h, l in zip(higher, lower)) / len(higher)


def analyze(results: list[SampleResult], samples: dict[str, RagSample], labels: dict[str, dict], t: float) -> dict:
    ok = [r for r in results if r.status != "error"]
    out: dict[str, Any] = {"samples": len(ok), "errors": len(results) - len(ok)}
    scores: dict[str, dict[str, dict[str, float]]] = {k: defaultdict(dict) for k in SCORES}
    for r in ok:
        label = labels[r.sample_id]
        value = {"statement": r.generation.get("answer_relevancy"), "scale": r.generation.get("answer_relevancy_scale")}
        for k in SCORES:
            if value[k] is not None:
                scores[k][label["group"]][label["variant"]] = value[k]
    variants = sorted({labels[r.sample_id]["variant"] for r in ok})
    reference = "good" if "good" in variants else "base"

    for k in SCORES:
        groups = scores[k]
        if not groups:
            continue
        entry: dict[str, Any] = {
            "mean": {v: _mean([g[v] for g in groups.values() if v in g]) for v in variants},
        }
        for other in variants:
            if other == reference:
                continue
            pairs = [(g[reference], g[other]) for g in groups.values() if reference in g and other in g]
            hi, lo = [a for a, _ in pairs], [b for _, b in pairs]
            entry[f"{reference}_vs_{other}"] = {
                "pairs": len(pairs),
                "pairwise_acc": _pairwise(hi, lo),
                "auroc": auroc(hi + lo, [1] * len(hi) + [0] * len(lo)),
                "mean_diff": _mean([a - b for a, b in pairs]),
                "within_0.1": sum(abs(a - b) <= 0.1 for a, b in pairs) / len(pairs) if pairs else None,
            }
        out[k] = entry

    # Statement level (synthetic sets): injected off-topic sentence vs base statements.
    injected, base_flags, statements = [], [], 0
    for r in ok:
        label, answer = labels[r.sample_id], samples[r.sample_id].answer
        for rec in r.units:
            if rec.unit.kind != "statement":
                continue
            statements += 1
            if label["variant"] == "base":
                base_flags.append(rec.decision.p < t)
            elif label["variant"] == "offtopic":
                span = rec.unit.source_span or ""
                start = answer.find(span) if span else -1
                s = label["spans"][0]
                if start >= 0 and start < s["end"] and s["start"] < start + len(span):
                    injected.append(rec.decision.p < t)
    offtopic_groups = sum(1 for r in ok if labels[r.sample_id]["variant"] == "offtopic")
    out["statements_per_answer"] = statements / len(ok) if ok else None
    if base_flags:
        out["base_statement_flag_rate"] = sum(base_flags) / len(base_flags)
    if offtopic_groups:
        out["offtopic_statement_detect_rate"] = sum(injected) / len(injected) if injected else None
        out["offtopic_statement_coverage"] = len(injected) / offtopic_groups
    return out


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


async def extract_all(extractor: LlmExtractor, samples: list[RagSample], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async def one(s: RagSample) -> None:
        async with sem:
            started = perf_counter()
            await extractor.statements(s.question, s.answer)
            latencies.append(perf_counter() - started)

    before = extractor.calls
    await asyncio.gather(*(one(s) for s in samples))
    fresh = extractor.calls - before
    return {"llm_calls": fresh, "latency_p50": percentile(latencies, 0.5) if fresh else None}


async def run_config(cfg: Config, samples: list[RagSample], extractor: LlmExtractor, concurrency: int):
    settings = get_settings()
    inner: Any = (
        JevJudge.from_settings(settings)
        if cfg.judge == "jev"
        else LlmJudge.from_settings(settings, model=cfg.judge.split(":", 1)[1], max_concurrency=concurrency)
    )
    timed = TimedJudge(inner)
    evaluator = Evaluator.from_settings(settings, timed, extractor)
    evaluator.statement_relevance_version = cfg.statement
    evaluator.answer_scale_version = cfg.scale
    try:
        results = await evaluator.evaluate(samples, ("answer_relevancy",), concurrency=concurrency)
    finally:
        await inner.aclose()
    return results, timed


async def main_async(args: argparse.Namespace) -> dict:
    settings = get_settings()
    configs = [Config.parse(c) for c in args.configs]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = AnswerCache(RAW_DIR / "extractions.sqlite")
    extractor = LlmExtractor.from_settings(settings, cache)
    report: dict[str, Any] = {"date": date.today().isoformat(), "datasets": {}}
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
    pct = lambda x: "–" if x is None else f"{x * 100:.1f}%"
    out = []
    for name, entry in report["datasets"].items():
        tags = [k for k in entry if k not in ("samples", "extraction")]
        first = entry[tags[0]]
        out.append(f"### {name} ({entry['samples']} samples, statement {f(first['statements_per_answer'], 2)}개/답변)\n")
        comparisons = sorted({c for tag in tags for k in SCORES if k in entry[tag] for c in entry[tag][k] if "_vs_" in c})
        head = "| 구성 | 점수 | 평균 (" + " / ".join(entry[tags[0]]["statement"]["mean"]) + ") | "
        head += " | ".join(f"{c}: 쌍 정확도 / 평균 차 / 차이 0.1 이내" for c in comparisons) + " |"
        out.append(head)
        out.append("|---|---|---|" + "---|" * len(comparisons))
        for tag in tags:
            for k in SCORES:
                if k not in entry[tag]:
                    continue
                e = entry[tag][k]
                means = " / ".join(f(v) for v in e["mean"].values())
                cells = []
                for c in comparisons:
                    x = e.get(c)
                    cells.append("–" if not x else f"{f(x['pairwise_acc'])} / {f(x['mean_diff'])} / {pct(x['within_0.1'])}")
                out.append(f"| {tag} | {k} | {means} | " + " | ".join(cells) + " |")
        if "offtopic_statement_detect_rate" in first:
            out.append("\n| 구성 | 삽입된 무관 문장 탐지율 | 원본 statement 오탐률 | p50 지연(s) | 비용(USD) | 오류 |")
            out.append("|---|---|---|---|---|---|")
            for tag in tags:
                a = entry[tag]
                out.append(
                    f"| {tag} | {pct(a.get('offtopic_statement_detect_rate'))} | {pct(a.get('base_statement_flag_rate'))} "
                    f"| {f(a['usage']['latency_p50'], 2)} | {f(a['usage']['cost_usd'], 4)} | {a['errors']} |"
                )
        else:
            out.append("\n| 구성 | p50 지연(s) | 비용(USD) | 오류 |")
            out.append("|---|---|---|---|")
            for tag in tags:
                a = entry[tag]
                out.append(f"| {tag} | {f(a['usage']['latency_p50'], 2)} | {f(a['usage']['cost_usd'], 4)} | {a['errors']} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
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
