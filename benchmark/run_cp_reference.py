"""Context Precision with a reference answer: ragas vs LLM Judge vs JEV vs Hybrid.

Data: prepare_cp_reference.py (Phase 3 references, MIRACL native-speaker labels
per context, fixed context order). Ground truth per sample is the ragas formula
applied to the human labels: average precision over the ranked contexts.

  ragas   ContextPrecision (with reference): one gpt-6-sol verdict per context -> AP
  llm     gpt-6-sol answering chunk_relevance.v2 (question + contexts, no reference)
  jev     JEV answering chunk_relevance.v2
  hybrid  JEV calibrated with a map fitted on Phase 1 MIRACL units of *other*
          queries (no leakage), then conf < 0.60 -> gpt-6-sol (Phase 5 policy)
  hybrid_recal  same routing, but the map is re-fitted on this set's own labels with
          ragas_jev.scoring.recalibration (2-fold by query: each half is scored with
          a map fitted on the other half), i.e. the domain re-calibration workflow

For llm / jev / hybrid the AP uses verdict = p >= 0.5, like ragas; the soft
precision (mean p) is reported as well.

Usage:
  uv run python benchmark/run_cp_reference.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats import binary_report, kendall_tau, pearson, percentile, spearman  # noqa: E402

from ragas_jev.audit.router import DEFAULT_METRIC_POLICIES  # noqa: E402
from ragas_jev.scoring.calibration import fit_isotonic  # noqa: E402
from ragas_jev.scoring.recalibration import LabeledUnit, fit_calibration  # noqa: E402

LANGS = ("ko", "en")


def average_precision(verdicts: list[int]) -> float:
    """ragas' formula (ContextPrecision._calculate_average_precision)."""
    hits = 0
    total = 0.0
    for i, v in enumerate(verdicts, start=1):
        if v:
            hits += 1
            total += hits / i
    return total / (sum(verdicts) + 1e-10)


def load_units(path: Path) -> dict[str, list[float]]:
    out = {}
    for r in map(json.loads, path.open(encoding="utf-8")):
        units = sorted(r["units"], key=lambda u: u["unit"]["index"])
        out[r["sample_id"]] = [u["decision"]["p"] for u in units]
    return out


def calibration_map(lang: str, exclude_queries: set[str]):
    """Isotonic map from Phase 1 MIRACL units of queries not in this benchmark."""
    ps, ys = [], []
    names = ["miracl_ko_dev_relevant", "miracl_ko_dev_non_relevant"] if lang == "ko" else ["miracl_en_dev_relevant"]
    for name in names:
        labels = {d["sample_id"]: d["labels"] for d in map(json.loads, open(f"benchmark/datasets/{name}/labels.jsonl", encoding="utf-8"))}
        for r in map(json.loads, open(f".cache/phase1/{name}.chunk_relevance.v2.jev.run0.jsonl", encoding="utf-8")):
            if r["sample_id"] in exclude_queries:
                continue
            for u in r["units"]:
                ps.append(u["decision"]["p"])
                ys.append(labels[r["sample_id"]][u["unit"]["index"]])
    return fit_isotonic(ps, ys), len(ps)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    policy = DEFAULT_METRIC_POLICIES["context_precision"]
    f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"
    report = {}
    for lang in LANGS:
        name = f"cp_ref_miracl_{lang}"
        labels = {d["sample_id"]: d["labels"] for d in map(json.loads, open(f"benchmark/datasets/{name}/labels.jsonl", encoding="utf-8"))}
        truth_ap = {sid: average_precision(y) for sid, y in labels.items()}
        truth_prec = {sid: sum(y) / len(y) for sid, y in labels.items()}
        jev = load_units(Path(f".cache/phase1/{name}.chunk_relevance.v2.jev.run0.jsonl"))
        llm = load_units(Path(f".cache/phase1/{name}.chunk_relevance.v2.llm.gpt-6-sol.jsonl"))
        queries = {sid.rsplit("-", 1)[0] for sid in labels}
        mapping, n_cal = calibration_map(lang, queries)

        # 2-fold re-calibration on this set's labels (by query), via the recalibration module
        fold_of = {q: i % 2 for i, q in enumerate(sorted(queries))}
        recal_maps = {}
        for k in (0, 1):
            train = [
                LabeledUnit("chunk_relevance.v2", lang, p, labels[sid][i], sid)
                for sid, ps in jev.items()
                if fold_of[sid.rsplit("-", 1)[0]] != k
                for i, p in enumerate(ps)
            ]
            cal, _ = fit_calibration(train, None, min_labels=100)
            recal_maps[k] = cal.lookup("chunk_relevance.v2", lang)

        systems: dict[str, dict[str, list[float]]] = {"llm": llm, "jev": jev, "hybrid": {}, "hybrid_recal": {}}
        stats = {"hybrid": [0, 0, 0], "hybrid_recal": [0, 0, 0]}  # escalated units, total units, samples with a call
        for sid, ps in jev.items():
            for system, m in (("hybrid", mapping), ("hybrid_recal", recal_maps[fold_of[sid.rsplit("-", 1)[0]]])):
                final, called = [], False
                for i, p in enumerate(ps):
                    c = m(p)
                    stats[system][1] += 1
                    if policy.band(max(c, 1 - c)) != "accept":
                        c, called = llm[sid][i], True
                        stats[system][0] += 1
                    final.append(c)
                stats[system][2] += called
                systems[system][sid] = final

        ragas_rows = [json.loads(l) for l in open(f".cache/phase6/ragas.{name}.context_precision.jsonl", encoding="utf-8")]
        ragas = {r["sample_id"]: r["score"] for r in ragas_rows if r["score"] is not None}
        ragas_lat = [r["latency"] for r in ragas_rows if r["error"] is None]

        rows = {}
        ids_all = sorted(labels)
        for system in ("ragas", "llm", "jev", "hybrid", "hybrid_recal"):
            if system == "ragas":
                ap = ragas
                soft = None
                unit = None
            else:
                ps = systems[system]
                ap = {sid: average_precision([int(p >= 0.5) for p in v]) for sid, v in ps.items()}
                soft = {sid: sum(v) / len(v) for sid, v in ps.items()}
                ys = [y for sid in ids_all if sid in ps for y in labels[sid]]
                pred = [int(p >= 0.5) for sid in ids_all if sid in ps for p in ps[sid]]
                unit = binary_report(ys, pred)
            ids = [sid for sid in ids_all if sid in ap]
            x, y = [ap[s] for s in ids], [truth_ap[s] for s in ids]
            rows[system] = {
                "n": len(ids),
                "ap_pearson": pearson(x, y), "ap_spearman": spearman(x, y), "ap_kendall": kendall_tau(x, y),
                "ap_mae": sum(abs(a - b) for a, b in zip(x, y)) / len(ids) if ids else None,
                "soft_pearson_precision": pearson([soft[s] for s in ids], [truth_prec[s] for s in ids]) if soft else None,
                "soft_mae_precision": (sum(abs(soft[s] - truth_prec[s]) for s in ids) / len(ids)) if soft else None,
                "unit": unit,
            }
        report[lang] = {
            "rows": rows,
            "hybrid": {"escalated_unit_rate": stats["hybrid"][0] / stats["hybrid"][1], "llm_calls_per_sample": stats["hybrid"][2] / len(jev), "calibration_units": n_cal},
            "hybrid_recal": {"escalated_unit_rate": stats["hybrid_recal"][0] / stats["hybrid_recal"][1], "llm_calls_per_sample": stats["hybrid_recal"][2] / len(jev)},
            "ragas_latency_p50": percentile(ragas_lat, 0.5) if ragas_lat else None,
            "ragas_errors": sum(1 for r in ragas_rows if r["error"]),
        }

    Path(".cache/phase6/cp_reference_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    names = {
        "ragas": "RAGAS ContextPrecision (reference 사용)", "llm": "LLM Judge (gpt-6-sol)", "jev": "JEV만",
        "hybrid": "Hybrid (기본 보정 + routing)", "hybrid_recal": "Hybrid (이 도메인 라벨로 재보정 + routing)",
    }
    for lang in LANGS:
        rep = report[lang]
        print(f"#### {lang} ({rep['rows']['jev']['n']} samples)\n")
        print("| 시스템 | AP Pearson / Spearman / Kendall | AP MAE | soft precision Pearson / MAE | chunk F1 / κ |")
        print("|---|---|---|---|---|")
        for s, r in rep["rows"].items():
            u = r["unit"]
            print(
                f"| {names[s]} | {f(r['ap_pearson'])} / {f(r['ap_spearman'])} / {f(r['ap_kendall'])} | {f(r['ap_mae'])} "
                f"| {f(r['soft_pearson_precision'])} / {f(r['soft_mae_precision'])} | {f(u['f1']) + ' / ' + f(u['kappa']) if u else '–'} |"
            )
        h, hr = rep["hybrid"], rep["hybrid_recal"]
        print(
            f"Hybrid 재보정: 재판정 unit {hr['escalated_unit_rate'] * 100:.1f}%, 샘플당 LLM 호출 {hr['llm_calls_per_sample']:.2f}"
        )
        print(
            f"\nHybrid: 재판정 unit {h['escalated_unit_rate'] * 100:.1f}%, 샘플당 LLM 호출 {h['llm_calls_per_sample']:.2f} "
            f"(보정 학습 unit {h['calibration_units']}개, 이 세트의 질문 제외). RAGAS: 샘플당 LLM 호출 5 (context마다 1), "
            f"p50 지연 {f(rep['ragas_latency_p50'], 1)}초, 오류 {rep['ragas_errors']}\n"
        )


if __name__ == "__main__":
    main()
