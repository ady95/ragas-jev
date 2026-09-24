"""Phase 6 benchmark: RAGAS vs LLM Judge vs JEV-only vs Hybrid (no API calls).

Systems, all scored per sample on the same data:
  ragas     official ragas 0.4 metrics judged by gpt-6-sol (run_ragas_baseline.py)
  llm       this pipeline with gpt-6-sol as the judge (same units, same question versions as JEV)
  jev       this pipeline, JEV only (raw probabilities, no routing)
  hybrid    JEV + calibration + per-metric routing (DEFAULT_METRIC_POLICIES). Calibration
            values are the Phase 5 out-of-fold ones, so no sample is scored with a map fitted on it.

Sample-level ground truth:
  context_precision  MIRACL human labels: fraction of judged passages that are relevant
  faithfulness       RAGTruth expert labels / synthetic: does the response contain a hallucination
  context_recall     share of reference sentences whose source passage is still in the contexts
  answer_relevancy   variant pairs (WikiEval good/poor/wrong, synthetic base/offtopic/nonanswer/wrong)

Usage:
  uv run python benchmark/run_phase6.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_phase4  # noqa: E402
from run_phase5 import cross_calibrate, load_context_precision, load_context_recall, load_faithfulness  # noqa: E402
from stats import auroc, binary_report, kendall_tau, pearson, percentile, spearman  # noqa: E402

from ragas_jev.audit.router import DEFAULT_METRIC_POLICIES  # noqa: E402
from ragas_jev.schemas import SampleResult  # noqa: E402

OUT = Path(".cache/phase6")
SYSTEMS = ("ragas", "llm", "jev", "hybrid")
RAGAS_METRIC = {"context_precision": "context_relevance", "faithfulness": "faithfulness", "context_recall": "context_recall"}


def _mean(xs: list[float]) -> float | None:
    return math.fsum(xs) / len(xs) if xs else None


def load_ragas(metric: str, datasets: list[str]) -> dict[str, dict]:
    out = {}
    for ds in datasets:
        path = OUT / f"ragas.{ds}.{RAGAS_METRIC[metric]}.jsonl"
        if path.exists():
            for row in map(json.loads, path.open(encoding="utf-8")):
                out[row["sample_id"]] = row
    return out


def unit_systems(metric: str, md) -> tuple[dict[str, dict[str, float]], dict[str, float], dict]:
    """Per-sample scores for llm / jev / hybrid from the unit table, plus hybrid routing stats."""
    units = [u for u in md.units if all(r in u.p for r in ("jev", "auditor", "strong"))]
    cal = cross_calibrate(units)
    policy = DEFAULT_METRIC_POLICIES[metric]
    finals: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in ("llm", "jev", "hybrid")}
    called: dict[str, set[str]] = defaultdict(set)
    escalated = 0
    for u in units:
        sid = u.key[0]
        finals["llm"][sid].append(u.p["strong"])  # gpt-6-sol with the same question version as JEV
        finals["jev"][sid].append(u.p["jev"])
        p = cal.get(u.key, u.p["jev"])
        band = policy.band(max(p, 1 - p))
        if band == "audit":
            p, escalated = u.p["auditor"], escalated + 1
            called[sid].add("audit")
        elif band == "strong":
            p, escalated = u.p["strong"], escalated + 1
            called[sid].add("strong")
        finals["hybrid"][sid].append(p)
    scores = {s: {sid: _mean(ps) for sid, ps in f.items()} for s, f in finals.items()}
    flags = {s: {sid: any(p < 0.5 for p in ps) for sid, ps in f.items()} for s, f in finals.items()}
    samples = {u.key[0] for u in units}
    routing = {
        "escalated_unit_rate": escalated / len(units) if units else 0.0,
        "llm_calls_per_sample": sum(len(v) for v in called.values()) / len(samples) if samples else 0.0,
    }
    return scores, flags, routing


def correlation_block(pred: dict[str, float], truth: dict[str, float]) -> dict:
    ids = sorted(set(pred) & set(truth))
    x, y = [pred[i] for i in ids], [truth[i] for i in ids]
    return {
        "n": len(ids),
        "pearson": pearson(x, y),
        "spearman": spearman(x, y),
        "kendall": kendall_tau(x, y),
        "mae": _mean([abs(a - b) for a, b in zip(x, y)]),
    }


# --- metrics --------------------------------------------------------------


def bench_context_precision() -> dict:
    md = load_context_precision()
    truth_units: dict[str, list[int]] = defaultdict(list)
    lang_of = {}
    for u in md.units:
        truth_units[u.key[0]].append(u.y)
        lang_of[u.key[0]] = u.lang
    truth = {sid: sum(v) / len(v) for sid, v in truth_units.items()}
    scores, _, routing = unit_systems("context_precision", md)
    ragas = load_ragas("context_precision", ["miracl_ko_dev_relevant", "miracl_en_dev_relevant", "miracl_ko_dev_non_relevant"])
    scores["ragas"] = {sid: r["score"] for sid, r in ragas.items() if r["score"] is not None}
    subsets = {
        "ko_relevant": lambda sid: lang_of.get(sid) == "ko" and "#" not in sid,
        "en_relevant": lambda sid: lang_of.get(sid) == "en",
        "ko_non_relevant": lambda sid: "#" in sid,
    }
    out: dict[str, Any] = {"routing": routing, "systems": {}}
    for system in SYSTEMS:
        entry = {}
        for name, keep in subsets.items():
            pred = {sid: v for sid, v in scores.get(system, {}).items() if keep(sid) and v is not None}
            gt = {sid: v for sid, v in truth.items() if keep(sid)}
            entry[name] = {"mean": _mean(list(pred.values())), **correlation_block(pred, gt)}
        out["systems"][system] = entry
    out["errors"] = {"ragas": sum(1 for r in ragas.values() if r["error"])}
    out["scores"] = scores
    return out


def bench_faithfulness() -> dict:
    md = load_faithfulness()
    labels = {}
    for ds in ("ragtruth_qa_test", "ko_hallu_miracl"):
        for d in map(json.loads, open(f"benchmark/datasets/{ds}/labels.jsonl", encoding="utf-8")):
            labels[d["sample_id"]] = (ds, int(d["hallucinated"]))
    scores, flags, routing = unit_systems("faithfulness", md)
    ragas = load_ragas("faithfulness", ["ragtruth_qa_test", "ko_hallu_miracl"])
    scores["ragas"] = {sid: r["score"] for sid, r in ragas.items() if r["score"] is not None}
    flags["ragas"] = {sid: s < 1.0 - 1e-9 for sid, s in scores["ragas"].items()}
    out: dict[str, Any] = {"routing": routing, "systems": {}}
    for system in SYSTEMS:
        entry = {}
        for ds in ("ragtruth_qa_test", "ko_hallu_miracl"):
            ids = sorted(sid for sid in scores.get(system, {}) if labels.get(sid, ("",))[0] == ds and scores[system][sid] is not None)
            y = [labels[sid][1] for sid in ids]
            entry[ds] = {
                "n": len(ids),
                "auroc": auroc([1 - scores[system][sid] for sid in ids], y),
                "flag": binary_report(y, [int(flags[system][sid]) for sid in ids]),
            }
        out["systems"][system] = entry
    out["errors"] = {"ragas": sum(1 for r in ragas.values() if r["error"])}
    out["scores"] = scores
    return out


def bench_context_recall() -> dict:
    md = load_context_recall()
    truth, variant = {}, {}
    for ds in ("recall_miracl_ko", "recall_miracl_en"):
        for d in map(json.loads, open(f"benchmark/datasets/{ds}/labels.jsonl", encoding="utf-8")):
            sents = d["sentences"]
            truth[d["sample_id"]] = sum(s["passage"] in d["kept_passages"] for s in sents) / len(sents)
            variant[d["sample_id"]] = d["variant"]
    scores, _, routing = unit_systems("context_recall", md)
    ragas = load_ragas("context_recall", ["recall_miracl_ko", "recall_miracl_en"])
    scores["ragas"] = {sid: r["score"] for sid, r in ragas.items() if r["score"] is not None}
    out: dict[str, Any] = {"routing": routing, "systems": {}}
    for system in SYSTEMS:
        entry = {}
        for lang in ("ko", "en"):
            pred = {sid: v for sid, v in scores.get(system, {}).items() if f"miracl-{lang}-" in sid and v is not None}
            gt = {sid: v for sid, v in truth.items() if sid in pred}
            by_q: dict[str, dict[str, float]] = defaultdict(dict)
            for sid, v in pred.items():
                by_q[sid.rsplit("-", 1)[0]][variant[sid]] = v
            complete = [q for q in by_q.values() if len(q) == 3]
            entry[lang] = {
                **correlation_block(pred, gt),
                "auroc_full_vs_none": auroc([q["full"] for q in complete] + [q["none"] for q in complete], [1] * len(complete) + [0] * len(complete)),
                "monotonic_rate": sum(q["full"] >= q["partial"] >= q["none"] for q in complete) / len(complete) if complete else None,
            }
        out["systems"][system] = entry
    out["errors"] = {"ragas": sum(1 for r in ragas.values() if r["error"])}
    out["scores"] = scores
    return out


AR_DATASETS = ("relevancy_wikieval", "relevancy_miracl_ko", "relevancy_miracl_en")
AR_PAIRS = {
    "relevancy_wikieval": [("good", "poor"), ("good", "wrong")],
    "relevancy_miracl_ko": [("base", "offtopic"), ("base", "nonanswer"), ("base", "wrong")],
    "relevancy_miracl_en": [("base", "offtopic"), ("base", "nonanswer"), ("base", "wrong")],
}


def bench_answer_relevancy() -> dict:
    """Pairwise comparisons of variant answers. Hybrid = JEV (no routing for this metric).

    ragas AnswerRelevancy uses a local multilingual SentenceTransformer (the proxy has no embeddings).
    """
    tags = {"jev": "jev.statement_relevance.v2.answer_relevance_scale.v2", "llm": "llm-gpt-6-sol.statement_relevance.v2.noscale"}
    scores: dict[str, dict[str, float]] = {s: {} for s in ("ragas", "llm", "jev")}
    labels: dict[str, dict] = {}
    for ds in AR_DATASETS:
        _, ds_labels = run_phase4.load(ds)
        labels.update({sid: {**lab, "dataset": ds} for sid, lab in ds_labels.items()})
        for system, tag in tags.items():
            for r in map(json.loads, (Path(".cache/phase4") / f"{ds}.{tag}.jsonl").open(encoding="utf-8")):
                value = r["generation"].get("answer_relevancy")
                if value is not None:
                    scores[system][r["sample_id"]] = value
        path = OUT / f"ragas.{ds}.answer_relevancy.jsonl"
        if path.exists():
            for r in map(json.loads, path.open(encoding="utf-8")):
                if r["score"] is not None:
                    scores["ragas"][r["sample_id"]] = r["score"]
    out: dict[str, Any] = {"systems": {}}
    for system, sc in scores.items():
        entry: dict[str, dict] = {}
        for ds, pairs in AR_PAIRS.items():
            groups: dict[str, dict[str, float]] = defaultdict(dict)
            for sid, v in sc.items():
                lab = labels.get(sid)
                if lab and lab["dataset"] == ds:
                    groups[lab["group"]][lab["variant"]] = v
            entry[ds] = {}
            for hi, lo in pairs:
                pairs_ = [(g[hi], g[lo]) for g in groups.values() if hi in g and lo in g]
                entry[ds][f"{hi}_vs_{lo}"] = {
                    "pairs": len(pairs_),
                    "pairwise_acc": sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs_) / len(pairs_) if pairs_ else None,
                    "mean_diff": _mean([a - b for a, b in pairs_]),
                }
        out["systems"][system] = entry
    out["systems"]["hybrid"] = out["systems"]["jev"]
    out["errors"] = {"ragas": sum(1 for ds in AR_DATASETS for path in [OUT / f"ragas.{ds}.answer_relevancy.jsonl"] if path.exists() for r in map(json.loads, path.open(encoding="utf-8")) if r["error"])}
    return out


def agreement_between_systems(scores: dict[str, dict[str, float]]) -> dict[str, float | None]:
    out = {}
    names = [s for s in SYSTEMS if scores.get(s)]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ids = sorted(sid for sid in set(scores[a]) & set(scores[b]) if scores[a][sid] is not None and scores[b][sid] is not None)
            out[f"{a}~{b}"] = kendall_tau([scores[a][s] for s in ids], [scores[b][s] for s in ids]) if ids else None
    return out


def repeatability() -> dict[str, dict]:
    """Two runs of the same 80 RAGTruth responses (same claims for llm / jev)."""
    base = Path(".cache/phase2")
    out: dict[str, dict] = {}
    for system, tag in (("jev", "jev.claim_support.v1.joint"), ("llm", "llm-gpt-6-sol.claim_support.v1.joint")):
        rep_path = base / f"ragtruth_qa_test.{tag}.rep1.jsonl"
        if not rep_path.exists():
            continue
        rep = {r["sample_id"]: r for r in map(json.loads, rep_path.open(encoding="utf-8"))}
        orig = {
            r["sample_id"]: r
            for r in map(json.loads, (base / f"ragtruth_qa_test.{tag}.jsonl").open(encoding="utf-8"))
            if r["sample_id"] in rep
        }
        deltas, flips, sample_d, sample_flip = [], 0, [], 0
        for sid, r in orig.items():
            a_units = {u["unit"]["unit_id"]: u["decision"]["p"] for u in r["units"]}
            b_units = {u["unit"]["unit_id"]: u["decision"]["p"] for u in rep[sid]["units"]}
            for uid in a_units.keys() & b_units.keys():
                deltas.append(abs(a_units[uid] - b_units[uid]))
                flips += (a_units[uid] >= 0.5) != (b_units[uid] >= 0.5)
            fa, fb = r["generation"].get("faithfulness"), rep[sid]["generation"].get("faithfulness")
            if fa is not None and fb is not None:
                sample_d.append(abs(fa - fb))
            sample_flip += any(p < 0.5 for p in a_units.values()) != any(p < 0.5 for p in b_units.values())
        out[system] = {
            "samples": len(orig),
            "units": len(deltas),
            "unit_identical_rate": sum(d == 0 for d in deltas) / len(deltas),
            "unit_flip_rate": flips / len(deltas),
            "unit_mean_delta": _mean(deltas),
            "unit_max_delta": max(deltas),
            "sample_mean_delta": _mean(sample_d),
            "sample_max_delta": max(sample_d),
            "sample_flag_flip_rate": sample_flip / len(orig),
        }
    rep_path = OUT / "ragas.ragtruth_qa_test.faithfulness.rep1.jsonl"
    if rep_path.exists():
        rep = {r["sample_id"]: r["score"] for r in map(json.loads, rep_path.open(encoding="utf-8")) if r["score"] is not None}
        orig = {
            r["sample_id"]: r["score"]
            for r in map(json.loads, (OUT / "ragas.ragtruth_qa_test.faithfulness.jsonl").open(encoding="utf-8"))
            if r["score"] is not None and r["sample_id"] in rep
        }
        d = [abs(orig[k] - rep[k]) for k in orig]
        out["ragas"] = {
            "samples": len(d),
            "units": None,
            "unit_identical_rate": None,
            "unit_flip_rate": None,
            "unit_mean_delta": None,
            "unit_max_delta": None,
            "sample_mean_delta": _mean(d),
            "sample_max_delta": max(d) if d else None,
            "sample_flag_flip_rate": sum((orig[k] < 1 - 1e-9) != (rep[k] < 1 - 1e-9) for k in orig) / len(orig) if orig else None,
        }
    return out


def ragas_efficiency() -> dict[str, dict]:
    out = {}
    for metric, rmetric in RAGAS_METRIC.items():
        lat = []
        for path in OUT.glob(f"ragas.*.{rmetric}.jsonl"):
            lat += [r["latency"] for r in map(json.loads, path.open(encoding="utf-8")) if r["error"] is None]
        out[metric] = {"latency_p50": percentile(lat, 0.5) if lat else None, "latency_p95": percentile(lat, 0.95) if lat else None}
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"date": date.today().isoformat(), "metrics": {}}
    for name, fn in (
        ("context_precision", bench_context_precision),
        ("faithfulness", bench_faithfulness),
        ("context_recall", bench_context_recall),
        ("answer_relevancy", bench_answer_relevancy),
    ):
        result = fn()
        if "scores" in result:
            result["system_agreement_kendall"] = agreement_between_systems(result.pop("scores"))
        report["metrics"][name] = result
    report["ragas_efficiency"] = ragas_efficiency()
    report["repeatability"] = repeatability()
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render(report))


def render(report: dict) -> str:
    f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"
    names = {"ragas": "RAGAS (ragas 0.4, gpt-6-sol)", "llm": "LLM Judge (gpt-6-sol)", "jev": "JEV만", "hybrid": "Hybrid (보정 + routing)"}
    m = report["metrics"]
    out = []

    cp = m["context_precision"]
    out.append("#### Context Precision (사람 라벨 관련 비율과의 상관)\n")
    out.append("| 시스템 | 한국어 Spearman / Kendall / MAE | 영어 Spearman / Kendall / MAE | 전부 무관 세트 평균 점수 (낮을수록 좋음) |")
    out.append("|---|---|---|---|")
    for s in SYSTEMS:
        e = cp["systems"][s]
        ko, en, nr = e["ko_relevant"], e["en_relevant"], e["ko_non_relevant"]
        out.append(f"| {names[s]} | {f(ko['spearman'])} / {f(ko['kendall'])} / {f(ko['mae'])} | {f(en['spearman'])} / {f(en['kendall'])} / {f(en['mae'])} | {f(nr['mean'])} |")

    fa = m["faithfulness"]
    out.append("\n#### Faithfulness (응답 단위 hallucination 탐지)\n")
    out.append("| 시스템 | RAGTruth AUROC | RAGTruth P / R / F1 | 한국어 AUROC | 한국어 P / R / F1 |")
    out.append("|---|---|---|---|---|")
    for s in SYSTEMS:
        e = fa["systems"][s]
        rt, ko = e["ragtruth_qa_test"], e["ko_hallu_miracl"]
        out.append(
            f"| {names[s]} | {f(rt['auroc'])} | {f(rt['flag']['precision'])} / {f(rt['flag']['recall'])} / {f(rt['flag']['f1'])} "
            f"| {f(ko['auroc'])} | {f(ko['flag']['precision'])} / {f(ko['flag']['recall'])} / {f(ko['flag']['f1'])} |"
        )

    cr = m["context_recall"]
    out.append("\n#### Context Recall (기대 recall과의 상관, context 제거 반응)\n")
    out.append("| 시스템 | 한국어 Pearson / MAE / full vs none AUROC / 단조 | 영어 Pearson / MAE / full vs none AUROC / 단조 |")
    out.append("|---|---|---|")
    for s in SYSTEMS:
        e = cr["systems"][s]
        cells = [f"{f(e[l]['pearson'])} / {f(e[l]['mae'])} / {f(e[l]['auroc_full_vs_none'])} / {f(e[l]['monotonic_rate'])}" for l in ("ko", "en")]
        out.append(f"| {names[s]} | " + " | ".join(cells) + " |")

    ar = m["answer_relevancy"]
    out.append("\n#### Answer Relevancy (쌍 정확도; Hybrid = JEV)\n")
    out.append("RAGAS AnswerRelevancy는 로컬 다국어 임베딩(paraphrase-multilingual-MiniLM-L12-v2)으로 실행했다. "
               "wrong 비교는 쌍 정확도가 0.5 근처, 평균 차가 0에 가까워야 좋다 (관련성은 정답 여부와 무관해야 한다).\n")
    out.append("| 시스템 | WikiEval good > poor | 한국어 base > offtopic | 한국어 base > nonanswer | 영어 base > offtopic | 영어 base > nonanswer "
               "| WikiEval good vs wrong 평균 차 | 한국어 base vs wrong 평균 차 | 영어 base vs wrong 평균 차 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for s in ("ragas", "llm", "jev"):
        e = ar["systems"].get(s)
        if not e or not any(v.get("pairs") for ds in e.values() for v in ds.values()):
            continue
        g = lambda ds, c, k="pairwise_acc": f(e[ds].get(c, {}).get(k))
        out.append(
            f"| {names[s]} | {g('relevancy_wikieval', 'good_vs_poor')} | {g('relevancy_miracl_ko', 'base_vs_offtopic')} | {g('relevancy_miracl_ko', 'base_vs_nonanswer')} "
            f"| {g('relevancy_miracl_en', 'base_vs_offtopic')} | {g('relevancy_miracl_en', 'base_vs_nonanswer')} "
            f"| {g('relevancy_wikieval', 'good_vs_wrong', 'mean_diff')} | {g('relevancy_miracl_ko', 'base_vs_wrong', 'mean_diff')} | {g('relevancy_miracl_en', 'base_vs_wrong', 'mean_diff')} |"
        )

    out.append("\n#### 시스템 간 점수 순위 일치 (Kendall τ, 샘플 단위)\n")
    out.append("| metric | " + " | ".join(k for k in m["context_precision"]["system_agreement_kendall"]) + " |")
    out.append("|---|" + "---|" * len(m["context_precision"]["system_agreement_kendall"]))
    for name in ("context_precision", "faithfulness", "context_recall"):
        agr = m[name]["system_agreement_kendall"]
        out.append(f"| {name} | " + " | ".join(f(v) for v in agr.values()) + " |")

    rep = report.get("repeatability", {})
    if rep:
        out.append("\n#### 반복 실행 안정성 (RAGTruth 80개 응답, 같은 입력 2회)\n")
        out.append("| 시스템 | unit 값 동일 | unit 판정 뒤집힘 | unit 평균 / 최대 Δp | 샘플 점수 평균 / 최대 Δ | 응답 판정(hallucination 여부) 뒤집힘 |")
        out.append("|---|---|---|---|---|---|")
        pct = lambda x: "–" if x is None else f"{x * 100:.1f}%"
        for s_ in ("ragas", "llm", "jev"):
            r = rep.get(s_)
            if r:
                out.append(
                    f"| {names[s_]} | {pct(r['unit_identical_rate'])} | {pct(r['unit_flip_rate'])} "
                    f"| {f(r['unit_mean_delta'])} / {f(r['unit_max_delta'])} "
                    f"| {f(r['sample_mean_delta'])} / {f(r['sample_max_delta'])} | {pct(r['sample_flag_flip_rate'])} |"
                )

    out.append("\n#### Hybrid routing 비용과 RAGAS 지연\n")
    out.append("| metric | Hybrid 재판정 unit 비율 | Hybrid 샘플당 LLM 호출 | RAGAS 샘플당 p50 / p95 지연(s) | RAGAS 오류 |")
    out.append("|---|---|---|---|---|")
    for name in ("context_precision", "faithfulness", "context_recall"):
        r, eff = m[name]["routing"], report["ragas_efficiency"][name]
        out.append(f"| {name} | {r['escalated_unit_rate'] * 100:.1f}% | {r['llm_calls_per_sample']:.2f} | {f(eff['latency_p50'], 1)} / {f(eff['latency_p95'], 1)} | {m[name]['errors']['ragas']} |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
