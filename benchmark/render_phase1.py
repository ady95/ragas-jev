"""Render the Phase 1 summary JSON as Markdown tables (printed to stdout).

Usage: uv run python benchmark/render_phase1.py [.cache/phase1/summary.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def f(x, digits: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def pct(x) -> str:
    return "–" if x is None else f"{x * 100:.1f}%"


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else ".cache/phase1/summary.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    ds = report["datasets"]
    out = []

    out.append("### Unit 단위 판정 (threshold = %s)\n" % report["threshold"])
    out.append("| 데이터셋 | Judge | n | Accuracy | Precision | Recall | F1 | κ | AUROC | ECE | 최적 threshold (F1) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, e in ds.items():
        for judge in ("jev", "llm"):
            if judge not in e:
                continue
            u = e[judge]["units"]
            if "at_threshold" not in u:
                out.append(f"| {name} | {judge.upper()} | 실패: {e[judge]['usage']['error_examples'][:1]} |" + " |" * 8)
                continue
            t = u["at_threshold"]
            best = f"{u['best_f1_threshold']:.2f} (F1 {u['best_f1']:.3f})" if "best_f1" in u else "–"
            label = "JEV" if judge == "jev" else f"LLM ({e['llm']['model']})"
            out.append(
                f"| {name} | {label} | {t['n']} | {f(t['accuracy'])} | {f(t['precision'])} | {f(t['recall'])} "
                f"| {f(t['f1'])} | {f(t['kappa'])} | {f(u['auroc'])} | {f(u['ece'])} | {best} |"
            )

    out.append("\n### Confidence 구간별 JEV 정확도\n")
    out.append("| 데이터셋 | 구간 | 비율 | n | Accuracy |")
    out.append("|---|---|---|---|---|")
    for name, e in ds.items():
        for b in e["jev"]["units"]["bands"]:
            out.append(f"| {name} | {b['band']} | {pct(b['share'])} | {b['n']} | {pct(b['accuracy'])} |")

    out.append("\n### Uncertainty 품질 (uncertainty로 오판을 예측하는 AUROC)\n")
    out.append("| 데이터셋 | JEV | LLM |")
    out.append("|---|---|---|")
    for name, e in ds.items():
        llm = e.get("llm", {}).get("units", {}).get("uncertainty_auroc")
        out.append(f"| {name} | {f(e['jev']['units']['uncertainty_auroc'])} | {f(llm)} |")

    out.append("\n### Sample 단위 Context Precision\n")
    out.append("| 데이터셋 | Judge | 사람 precision 평균 | soft 평균 | binary 평균 | Pearson(soft) | Spearman(soft) | MAE(soft) | 관련 chunk ≥1개 판정 비율 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for name, e in ds.items():
        for judge in ("jev", "llm"):
            if judge not in e or not e[judge]["samples"].get("n"):
                continue
            s = e[judge]["samples"]
            out.append(
                f"| {name} | {judge.upper()} | {f(s['human_mean'])} | {f(s['soft_mean'])} | {f(s['binary_mean'])} "
                f"| {f(s['pearson_soft'])} | {f(s['spearman_soft'])} | {f(s['mae_soft'])} | {pct(s['any_chunk_relevant_rate'])} |"
            )

    out.append("\n### 반복 실행 안정성 (JEV, 캐시 없음)\n")
    out.append("| 데이터셋 | 실행 수 | unit 수 | 모든 실행에서 값 동일 | 0.5 기준 판정 뒤집힘 | 평균 Δp | 최대 Δp |")
    out.append("|---|---|---|---|---|---|---|")
    for name, e in ds.items():
        r = e["jev"].get("repeatability")
        if r:
            out.append(
                f"| {name} | {r['runs']} | {r['units']} | {pct(r['identical_rate'])} | {pct(r['flip_rate'])} "
                f"| {f(r['mean_delta'])} | {f(r['max_delta'])} |"
            )

    out.append("\n### 지연 시간과 비용 (1회 실행 기준)\n")
    out.append("| 데이터셋 | Judge | 요청 수 | 입력 토큰 | p50 지연(s) | p95 지연(s) | JEV 비용(USD) | 오류 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for name, e in ds.items():
        for judge in ("jev", "llm"):
            if judge not in e:
                continue
            u = e[judge]["usage"]
            cost = f(u.get("cost_usd"), 4) if judge == "jev" else "–"
            out.append(
                f"| {name} | {judge.upper()} | {u['requests']} | {u['input_tokens']:,} | {f(u['latency_p50'], 2)} "
                f"| {f(u['latency_p95'], 2)} | {cost} | {u['errors']} |"
            )

    if any("jev_vs_llm" in e for e in ds.values()):
        out.append("\n### JEV vs LLM Judge\n")
        out.append("| 데이터셋 | unit 수 | 판정 일치율 | κ | Pearson(p) |")
        out.append("|---|---|---|---|---|")
        for name, e in ds.items():
            j = e.get("jev_vs_llm")
            if j:
                out.append(f"| {name} | {j['units']} | {pct(j['agreement'])} | {f(j['kappa'])} | {f(j['pearson_p'])} |")

    out.append("\n### Reliability (JEV p 구간별 실제 관련 비율)\n")
    for name, e in ds.items():
        rows = e["jev"]["units"]["reliability"]
        if not any(r["frac_relevant"] for r in rows):
            continue
        out.append(f"**{name}**\n")
        out.append("| p 구간 | n | 평균 p | 실제 관련 비율 |")
        out.append("|---|---|---|---|")
        for r in rows:
            out.append(f"| {r['bin']} | {r['n']} | {f(r['mean_p'])} | {f(r['frac_relevant'])} |")
        out.append("")

    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
