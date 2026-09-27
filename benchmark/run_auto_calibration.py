"""Can calibration data be produced automatically from an evaluation set? (no API calls)

Compares, on saved Phase 1-3 judgments with human / expected labels used only for scoring:

  raw     no calibration (0.2.0 default)
  base    isotonic map fitted on a *source* set with labels (the packaged-calibration situation)
  shift   C: base map adjusted to the target set's class prior without labels. The prior is
          estimated by EM on the target's unlabeled JEV probabilities (Saerens et al., 2002)
  silver  B: isotonic map fitted on the target's JEV p with LLM verdicts as labels
          (gpt-6-sol p >= 0.5); no human label is used
  human   isotonic map fitted on the target's human labels, 2-fold by query (the upper bound,
          i.e. domain re-calibration)

Scenarios:
  cp_shift       source MIRACL Phase 1 (18% relevant), target the reference CP set (44% relevant),
                 chunk_relevance.v2. This is where the packaged calibration failed in Phase 6.
  cp_v3          target the reference CP set with chunk_relevance.v3: no source map exists, so only
                 silver / human apply
  recall_kept    source: half of the recall queries, all variants; target: the other half without
                 the `none` variant (support rate goes up, like bench40 where retrieval mostly worked).
                 `full` alone is not used: every expected label is 1 there, which makes it trivial
  recall_same    same split, target keeps all variants (no shift: C should change nothing)
  faith_halluc   source: half of the responses; target: hallucinated responses of the other half
                 (support rate goes down)
  faith_same     same split, target keeps all responses

Usage:
  uv run python benchmark/run_auto_calibration.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cp_reference import average_precision, calibration_map, load_units  # noqa: E402
from run_phase5 import load_context_recall, load_faithfulness  # noqa: E402
from stats import binary_report, kendall_tau, pearson  # noqa: E402

from ragas_jev.scoring.calibration import fit_isotonic  # noqa: E402
from ragas_jev.scoring.recalibration import _ece  # noqa: E402

OUT = Path(".cache/auto_calibration")
METHODS = ("raw", "base", "shift", "silver", "human")


def prior_shift(qs: list[float], source_prior: float, iters: int = 200, tol: float = 1e-6):
    """EM estimate of the target prior; returns (prior, adjust(q)) for source-calibrated posteriors q."""
    pi_s = min(max(source_prior, 1e-3), 1 - 1e-3)
    pi_t = pi_s

    def adjust_with(pi: float):
        w1, w0 = pi / pi_s, (1 - pi) / (1 - pi_s)
        return lambda q: w1 * q / (w1 * q + w0 * (1 - q))

    for _ in range(iters):
        adj = adjust_with(pi_t)
        new = sum(adj(q) for q in qs) / len(qs)
        if abs(new - pi_t) < tol:
            pi_t = new
            break
        pi_t = new
    return pi_t, adjust_with(pi_t)


def two_fold_human(units: list[dict]) -> dict[int, float]:
    """Out-of-fold human-label map value per unit index (2-fold by group)."""
    groups = sorted({u["group"] for u in units})
    random.Random(7).shuffle(groups)
    fold = {g: i % 2 for i, g in enumerate(groups)}
    out = {}
    for k in (0, 1):
        train = [u for u in units if fold[u["group"]] != k]
        m = fit_isotonic([u["jev"] for u in train], [u["y"] for u in train])
        for i, u in enumerate(units):
            if fold[u["group"]] == k:
                out[i] = m(u["jev"])
    return out


def calibrate(target: list[dict], base_map, source_prior: float | None) -> dict[str, list[float]]:
    """Calibrated p per method for each target unit (same order)."""
    jev = [u["jev"] for u in target]
    out: dict[str, list[float]] = {"raw": jev}
    if base_map is not None:
        out["base"] = [base_map(p) for p in jev]
        est, adjust = prior_shift(out["base"], source_prior)
        out["shift"] = [adjust(q) for q in out["base"]]
        out["_prior_est"] = [est]
    silver = fit_isotonic(jev, [int(u["llm"] >= 0.5) for u in target])
    out["silver"] = [silver(p) for p in jev]
    oof = two_fold_human(target)
    out["human"] = [oof[i] for i in range(len(target))]
    return out


def unit_scores(target: list[dict], ps: list[float]) -> dict:
    ys = [u["y"] for u in target]
    b = binary_report(ys, [int(p >= 0.5) for p in ps])
    return {"ece": _ece(ps, ys), "kappa": b["kappa"], "f1": b["f1"], "mean_p": sum(ps) / len(ps)}


def sample_scores(target: list[dict], ps: list[float], kind: str) -> dict:
    by_sample: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for u, p in zip(target, ps):
        by_sample[u["sample"]].append((u["order"], p, u["y"]))
    ids = sorted(by_sample)
    if kind == "ap":
        pred, truth = [], []
        for sid in ids:
            rows = sorted(by_sample[sid])
            pred.append(average_precision([int(p >= 0.5) for _, p, _ in rows]))
            truth.append(average_precision([y for _, _, y in rows]))
    else:  # mean of p vs share of supported units
        pred = [sum(p for _, p, _ in by_sample[s]) / len(by_sample[s]) for s in ids]
        truth = [sum(y for _, _, y in by_sample[s]) / len(by_sample[s]) for s in ids]
    return {
        "n": len(ids),
        "pearson": pearson(pred, truth),
        "kendall": kendall_tau(pred, truth),
        "mae": sum(abs(a - b) for a, b in zip(pred, truth)) / len(ids),
    }


# --- scenarios --------------------------------------------------------------


def cp_units(lang: str, question: str) -> list[dict]:
    name = f"cp_ref_miracl_{lang}"
    labels = {d["sample_id"]: d["labels"] for d in map(json.loads, open(f"benchmark/datasets/{name}/labels.jsonl", encoding="utf-8"))}
    jev = load_units(Path(f".cache/phase1/{name}.{question}.jev.run0.jsonl"))
    llm = load_units(Path(f".cache/phase1/{name}.{question}.llm.gpt-6-sol.jsonl"))
    return [
        {"sample": sid, "group": sid.rsplit("-", 1)[0], "order": i, "y": labels[sid][i], "jev": p, "llm": llm[sid][i]}
        for sid, ps in jev.items()
        for i, p in enumerate(ps)
    ]


def split_units(md, target_filter, lang: str, group_of) -> tuple[list[dict], list[dict]]:
    units = [u for u in md.units if u.lang == lang and "strong" in u.p]
    groups = sorted({group_of(u.key[0]) for u in units})
    random.Random(11).shuffle(groups)
    half = {g: i % 2 for i, g in enumerate(groups)}
    as_dict = lambda u: {"sample": u.key[0], "group": group_of(u.key[0]), "order": 0, "y": u.y, "jev": u.p["jev"], "llm": u.p["strong"]}
    source = [as_dict(u) for u in units if half[group_of(u.key[0])] == 0]
    target = [as_dict(u) for u in units if half[group_of(u.key[0])] == 1 and target_filter(u.key[0])]
    return source, target


def scenarios():
    for lang in ("ko", "en"):
        target = cp_units(lang, "chunk_relevance.v2")
        base_map, _ = calibration_map(lang, {u["group"] for u in target})
        # source prior of the base map: MIRACL Phase 1 units of the other queries
        src_ys = []
        names = ["miracl_ko_dev_relevant", "miracl_ko_dev_non_relevant"] if lang == "ko" else ["miracl_en_dev_relevant"]
        exclude = {u["group"] for u in target}
        for name in names:
            for d in map(json.loads, open(f"benchmark/datasets/{name}/labels.jsonl", encoding="utf-8")):
                if d["sample_id"] not in exclude:
                    src_ys += d["labels"]
        yield f"cp_shift/{lang}", target, base_map, sum(src_ys) / len(src_ys), "ap"
        yield f"cp_v3/{lang}", cp_units(lang, "chunk_relevance.v3"), None, None, "ap"

    recall = load_context_recall()
    q_of = lambda sid: sid.rsplit("-", 1)[0]
    faith = load_faithfulness()
    labels = {}
    for ds in ("ragtruth_qa_test", "ko_hallu_miracl"):
        for d in map(json.loads, open(f"benchmark/datasets/{ds}/labels.jsonl", encoding="utf-8")):
            labels[d["sample_id"]] = d["hallucinated"]
    f_group = lambda sid: sid if sid.startswith("ragtruth") else sid.rsplit("-", 1)[0]
    for lang in ("ko", "en"):
        for name, keep in (("recall_kept", lambda sid: not sid.endswith("-none")), ("recall_same", lambda sid: True)):
            source, target = split_units(recall, keep, lang, q_of)
            yield f"{name}/{lang}", target, fit_isotonic([u["jev"] for u in source], [u["y"] for u in source]), sum(u["y"] for u in source) / len(source), "mean"
        for name, keep in (("faith_halluc", lambda sid: labels.get(sid, False)), ("faith_same", lambda sid: True)):
            source, target = split_units(faith, keep, lang, f_group)
            yield f"{name}/{lang}", target, fit_isotonic([u["jev"] for u in source], [u["y"] for u in source]), sum(u["y"] for u in source) / len(source), "mean"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, target, base_map, source_prior, kind in scenarios():
        cal = calibrate(target, base_map, source_prior)
        entry = {
            "units": len(target),
            "target_rate": sum(u["y"] for u in target) / len(target),
            "source_rate": source_prior,
            "estimated_rate": cal.get("_prior_est", [None])[0],
            "llm_label_rate": sum(u["llm"] >= 0.5 for u in target) / len(target),
            "llm_label_agreement": sum((u["llm"] >= 0.5) == bool(u["y"]) for u in target) / len(target),
            "methods": {},
        }
        for m in METHODS:
            if m in cal:
                entry["methods"][m] = {"unit": unit_scores(target, cal[m]), "sample": sample_scores(target, cal[m], kind)}
        report[name] = entry
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render(report))


def render(report: dict) -> str:
    f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"
    names = {"raw": "보정 없음", "base": "기본 곡선", "shift": "C: 비율 조정 (라벨 없음)", "silver": "B: LLM 라벨", "human": "사람 라벨 (상한)"}
    out = []
    for name, e in report.items():
        kind = "AP" if name.startswith("cp") else "샘플 평균"
        out.append(f"#### {name} (unit {e['units']}, 실제 비율 {f(e['target_rate'], 2)}, 기본 곡선 학습 비율 {f(e['source_rate'], 2)}, "
                   f"C 추정 비율 {f(e['estimated_rate'], 2)}, LLM 라벨 비율 {f(e['llm_label_rate'], 2)}, LLM 라벨 일치 {f(e['llm_label_agreement'], 2)})\n")
        out.append(f"| 방법 | unit ECE | unit κ | {kind} Pearson | {kind} Kendall | {kind} MAE |")
        out.append("|---|---|---|---|---|---|")
        for m, r in e["methods"].items():
            u, s = r["unit"], r["sample"]
            out.append(f"| {names[m]} | {f(u['ece'])} | {f(u['kappa'])} | {f(s['pearson'])} | {f(s['kendall'])} | {f(s['mae'])} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
