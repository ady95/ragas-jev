"""Phase 5: calibration and routing simulation from saved Phase 1-3 runs.

No API calls. For each metric it joins, per unit, the human/expected label with
the saved JEV p and the auditor (gpt-6-luna) / strong judge (gpt-6-sol) /
baseline (gpt-6-sol) p, then:

  1. calibration: grouped 5-fold cross-validation of per-language isotonic maps
     (a question never appears in both train and test folds); ECE / F1 raw vs
     calibrated
  2. routing: for a grid of (accept, audit) thresholds on the calibrated
     confidence, the final decision is JEV / auditor / strong judge by band;
     report quality against the escalation rate and LLM calls per request
  3. writes the calibration fitted on all data to
     src/ragas_jev/data/calibration_<jev model>.json

Usage:
  uv run python benchmark/run_phase5.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase3 import expected_support  # noqa: E402
from stats import auroc, binary_report, ece  # noqa: E402

from ragas_jev.scoring.calibration import Calibrator, fit_isotonic  # noqa: E402

OUT = Path(".cache/phase5")
MODELS = {"auditor": "gpt-6-luna", "strong": "gpt-6-sol", "baseline": "gpt-6-sol"}
# accept 0.0 = never escalate (JEV only); accept 1.01 = escalate everything.
# audit == accept means no auditor band: only confidence < audit goes to the strong judge.
ACCEPTS = [0.0, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
AUDITS = [0.0, 0.55, 0.6, 0.65, 0.7]


@dataclass
class Unit:
    key: tuple[str, str]
    group: str
    lang: str
    y: int  # 1 = yes (relevant / supported)
    p: dict[str, float] = field(default_factory=dict)  # "jev", "auditor", "strong", "baseline"


@dataclass
class MetricData:
    name: str
    question_id: str
    positive: int  # class whose F1 is reported (matches each phase report)
    units: list[Unit]


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def _collect(units: dict, source: str, rows: list[dict], key_fn) -> None:
    for r in rows:
        for u in r["units"]:
            key = key_fn(r, u)
            if key in units:
                units[key].p[source] = u["decision"]["p"]


def load_context_precision() -> MetricData:
    units: dict = {}
    for name, lang in (("miracl_ko_dev_relevant", "ko"), ("miracl_en_dev_relevant", "en"), ("miracl_ko_dev_non_relevant", "ko")):
        labels = {d["sample_id"]: d["labels"] for d in map(json.loads, open(f"benchmark/datasets/{name}/labels.jsonl", encoding="utf-8"))}
        base = Path(".cache/phase1")
        jev = _read(base / f"{name}.chunk_relevance.v2.jev.run0.jsonl")
        key_fn = lambda r, u: (r["sample_id"], u["unit"]["unit_id"])
        for r in jev:
            for u in r["units"]:
                units[key_fn(r, u)] = Unit(key_fn(r, u), r["sample_id"], lang, labels[r["sample_id"]][u["unit"]["index"]], {"jev": u["decision"]["p"]})
        for role, model in MODELS.items():
            _collect(units, role, _read(base / f"{name}.chunk_relevance.v2.llm.{model}.jsonl"), key_fn)
    return MetricData("context_precision", "chunk_relevance.v2", 1, list(units.values()))


def load_faithfulness() -> MetricData:
    units: dict = {}
    for name, lang in (("ragtruth_qa_test", "en"), ("ko_hallu_miracl", "ko")):
        labels = {d["sample_id"]: d for d in map(json.loads, open(f"benchmark/datasets/{name}/labels.jsonl", encoding="utf-8"))}
        answers = {d["sample_id"]: d["answer"] for d in map(json.loads, open(f"benchmark/datasets/{name}/samples.jsonl", encoding="utf-8"))}
        base = Path(".cache/phase2")
        key_fn = lambda r, u: (r["sample_id"], u["unit"]["unit_id"])
        for r in _read(base / f"{name}.jev.claim_support.v1.joint.jsonl"):
            answer, spans = answers[r["sample_id"]], labels[r["sample_id"]]["spans"]
            for u in r["units"]:
                span = u["unit"]["source_span"] or ""
                start = answer.find(span) if span else -1
                if start < 0:
                    continue
                hallucinated = any(start < s["end"] and s["start"] < start + len(span) for s in spans)
                units[key_fn(r, u)] = Unit(key_fn(r, u), r["sample_id"], lang, int(not hallucinated), {"jev": u["decision"]["p"]})
        _collect(units, "auditor", _read(base / f"{name}.llm-gpt-6-luna.claim_support.v1.joint.jsonl"), key_fn)
        _collect(units, "strong", _read(base / f"{name}.llm-gpt-6-sol.claim_support.v1.joint.jsonl"), key_fn)
        _collect(units, "baseline", _read(base / f"{name}.llm-gpt-6-sol.claim_support.v2.joint.jsonl"), key_fn)
    return MetricData("faithfulness", "claim_support.v1", 0, list(units.values()))


def load_context_recall() -> MetricData:
    units: dict = {}
    for name, lang in (("recall_miracl_ko", "ko"), ("recall_miracl_en", "en")):
        labels = {d["sample_id"]: d for d in map(json.loads, open(f"benchmark/datasets/{name}/labels.jsonl", encoding="utf-8"))}
        refs = {d["sample_id"]: d["reference"] for d in map(json.loads, open(f"benchmark/datasets/{name}/samples.jsonl", encoding="utf-8"))}
        base = Path(".cache/phase3")
        key_fn = lambda r, u: (r["sample_id"], u["unit"]["unit_id"])
        for r in _read(base / f"{name}.jev.claim_support.v1.joint.jsonl"):
            for u in r["units"]:
                expected = expected_support(refs[r["sample_id"]], u["unit"]["source_span"], labels[r["sample_id"]])
                if expected is None:
                    continue
                group = r["sample_id"].rsplit("-", 1)[0]  # all variants of a query share a fold
                units[key_fn(r, u)] = Unit(key_fn(r, u), group, lang, int(expected), {"jev": u["decision"]["p"]})
        for role, model in MODELS.items():
            _collect(units, role, _read(base / f"{name}.llm-{model}.claim_support.v1.joint.jsonl"), key_fn)
    return MetricData("context_recall", "ref_claim_coverage.v1", 1, list(units.values()))


# --- calibration ----------------------------------------------------------


def cross_calibrate(units: list[Unit], folds: int = 5, seed: int = 7) -> dict[tuple[str, str], float]:
    groups = sorted({u.group for u in units})
    random.Random(seed).shuffle(groups)
    fold_of = {g: i % folds for i, g in enumerate(groups)}
    out = {}
    for k in range(folds):
        for lang in sorted({u.lang for u in units}):
            train = [u for u in units if fold_of[u.group] != k and u.lang == lang]
            test = [u for u in units if fold_of[u.group] == k and u.lang == lang]
            if not train or not test:
                continue
            mapping = fit_isotonic([u.p["jev"] for u in train], [u.y for u in train])
            for u in test:
                out[u.key] = mapping(u.p["jev"])
    return out


def quality(ys: list[int], ps: list[float], positive: int, t: float = 0.5) -> dict[str, float]:
    preds = [int(p >= t) for p in ps]
    if positive == 0:  # report F1 of the "no" class (e.g. hallucination detection)
        ys, preds = [1 - y for y in ys], [1 - p for p in preds]
    b = binary_report(ys, preds)
    return {"f1": b["f1"], "kappa": b["kappa"], "accuracy": b["accuracy"]}


# --- routing --------------------------------------------------------------


def simulate(units: list[Unit], cal: dict, accept: float, audit: float, positive: int, calibrated: bool) -> dict:
    ys, ps = [], []
    audited = strong = flagged = 0
    requests: dict[str, set[str]] = defaultdict(set)  # sample -> bands that needed an LLM call
    for u in units:
        p = cal[u.key] if calibrated else u.p["jev"]
        conf = max(p, 1 - p)
        if conf >= accept:
            final = p
        elif conf >= audit:
            final, audited = u.p["auditor"], audited + 1
            requests[u.key[0]].add("audit")
            flagged += (p >= 0.5) != (final >= 0.5)
        else:
            final, strong = u.p["strong"], strong + 1
            requests[u.key[0]].add("strong")
            flagged += (p >= 0.5) != (final >= 0.5)
        ys.append(u.y)
        ps.append(final)
    samples = {u.key[0] for u in units}
    n = len(units)
    return {
        "accept": accept, "audit": audit, **quality(ys, ps, positive),
        "audited_rate": audited / n, "strong_rate": strong / n, "human_review_rate": flagged / n,
        "llm_calls_per_sample": sum(len(v) for v in requests.values()) / len(samples),
    }


def analyze(md: MetricData) -> dict:
    units = [u for u in md.units if all(r in u.p for r in ("jev", "auditor", "strong", "baseline"))]
    if not units:
        return {"units": 0, "dropped": len(md.units), "by_lang": {}, "grid": []}
    cal = cross_calibrate(units)
    units = [u for u in units if u.key in cal]
    ys = [u.y for u in units]
    out: dict[str, Any] = {"units": len(units), "dropped": len(md.units) - len(units), "by_lang": {}}
    for lang in sorted({u.lang for u in units}) + ["all"]:
        sel = [u for u in units if lang in ("all", u.lang)]
        y = [u.y for u in sel]
        raw = [u.p["jev"] for u in sel]
        calp = [cal[u.key] for u in sel]
        out["by_lang"][lang] = {
            "n": len(sel),
            "ece_raw": ece(raw, y), "ece_cal": ece(calp, y),
            "auroc_raw": auroc(raw, y), "auroc_cal": auroc(calp, y),
            "jev_raw": quality(y, raw, md.positive), "jev_cal": quality(y, calp, md.positive),
            **{role: quality(y, [u.p[role] for u in sel], md.positive) for role in ("auditor", "strong", "baseline")},
            "escalated_at_085_raw": sum(max(p, 1 - p) < 0.85 for p in raw) / len(sel),
            "escalated_at_085_cal": sum(max(p, 1 - p) < 0.85 for p in calp) / len(sel),
        }
    grid = []
    for calibrated in (False, True):
        for accept in ACCEPTS:
            for audit in AUDITS:
                if audit <= accept and (accept > 0 or audit == 0):
                    grid.append({"calibrated": calibrated, **simulate(units, cal, accept, audit, md.positive, calibrated)})
    out["grid"] = grid
    return out


def fit_final(metrics: list[MetricData], model: str) -> Calibrator:
    cal = Calibrator(source=f"phase5 {date.today().isoformat()} (MIRACL, RAGTruth, synthetic ko/en; {model})")
    for md in metrics:
        units = [u for u in md.units if "jev" in u.p]
        for lang in sorted({u.lang for u in units}):
            sel = [u for u in units if u.lang == lang]
            cal.maps[f"{md.question_id}:{lang}"] = fit_isotonic([u.p["jev"] for u in sel], [u.y for u in sel])
        cal.maps[f"{md.question_id}:*"] = fit_isotonic([u.p["jev"] for u in units], [u.y for u in units])
    return cal


def _row(grid: list[dict], calibrated: bool, accept: float, audit: float) -> dict | None:
    return next((r for r in grid if r["calibrated"] == calibrated and r["accept"] == accept and r["audit"] == audit), None)


def render(report: dict) -> str:
    out = []
    for name, m in report["metrics"].items():
        if not m["units"]:
            continue
        grid, overall = m["grid"], m["by_lang"]["all"]
        cal_grid = [r for r in grid if r["calibrated"]]
        best = max(cal_grid, key=lambda r: (round(r["kappa"], 4), -r["llm_calls_per_sample"]))
        policies = [
            ("JEV만 (보정 전)", _row(grid, False, 0.0, 0.0)),
            ("JEV만 (보정 후)", _row(grid, True, 0.0, 0.0)),
            ("보정 + Strong Judge만 (conf < 0.60)", _row(grid, True, 0.6, 0.6)),
            ("보정 + 계획서 기본값 (0.85 / 0.60)", _row(grid, True, 0.85, 0.6)),
            ("보정 없이 계획서 기본값 (0.85 / 0.60)", _row(grid, False, 0.85, 0.6)),
            (f"보정 + κ 최대 ({best['accept']:.2f} / {best['audit']:.2f})", best),
        ]
        out.append(f"#### {name} ({m['units']} units)\n")
        top = max(r["kappa"] for r in cal_grid)
        chosen = min((r for r in cal_grid if r["kappa"] >= top - 0.01), key=lambda r: (r["llm_calls_per_sample"], -r["kappa"]))
        policies.append((f"**권장: 보정 + ({chosen['accept']:.2f} / {chosen['audit']:.2f})**", chosen))
        out.append("| 정책 (accept / audit) | F1 | κ | Auditor 재판정 | Strong 재판정 | 사람 검토 | 샘플당 LLM 호출 |")
        out.append("|---|---|---|---|---|---|---|")
        for label, r in policies:
            if r:
                out.append(
                    f"| {label} | {r['f1']:.3f} | {r['kappa']:.3f} | {r['audited_rate'] * 100:.1f}% | {r['strong_rate'] * 100:.1f}% "
                    f"| {r['human_review_rate'] * 100:.1f}% | {r['llm_calls_per_sample']:.2f} |"
                )
        for role, label in (("auditor", "gpt-6-luna 단독"), ("strong", "gpt-6-sol 단독")):
            q = overall[role]
            out.append(f"| {label} | {q['f1']:.3f} | {q['kappa']:.3f} | 100% | – | – | 1.00 |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument("--no-write-calibration", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = [load_context_precision(), load_faithfulness(), load_context_recall()]
    report = {"date": date.today().isoformat(), "metrics": {md.name: analyze(md) for md in metrics}}
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_write_calibration:
        path = Path(f"src/ragas_jev/data/calibration_{args.jev_model}.json")
        fit_final(metrics, args.jev_model).save(path)
        print(f"calibration → {path}")
    print(render(report))
    for name, m in report["metrics"].items():
        print(f"== {name}: {m['units']} units (dropped {m['dropped']})")
        for lang, v in m["by_lang"].items():
            print(f"  {lang}: ECE {v['ece_raw']:.3f}->{v['ece_cal']:.3f}  F1 raw {v['jev_raw']['f1']:.3f} cal {v['jev_cal']['f1']:.3f} "
                  f"| auditor {v['auditor']['f1']:.3f} strong {v['strong']['f1']:.3f} baseline {v['baseline']['f1']:.3f} "
                  f"| escalated@0.85 raw {v['escalated_at_085_raw']:.2f} cal {v['escalated_at_085_cal']:.2f}")


if __name__ == "__main__":
    main()
