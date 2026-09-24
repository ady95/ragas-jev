"""Agreement, calibration, and correlation statistics for judge validation.

Pure Python so the benchmark adds no numeric dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def binary_report(labels: Sequence[int], preds: Sequence[int]) -> dict[str, float]:
    tp = sum(1 for y, p in zip(labels, preds) if y and p)
    fp = sum(1 for y, p in zip(labels, preds) if not y and p)
    fn = sum(1 for y, p in zip(labels, preds) if y and not p)
    tn = sum(1 for y, p in zip(labels, preds) if not y and not p)
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": cohen_kappa(labels, preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(1 for x, y in zip(a, b) if bool(x) == bool(y)) / n
    pa, pb = sum(map(bool, a)) / n, sum(map(bool, b)) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Probability that a random positive outranks a random negative (ties count half)."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        rank_sum += avg_rank * sum(1 for k in range(i, j) if pairs[k][1])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ece(probs: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    """Expected calibration error of P(label = 1)."""
    n = len(probs)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = math.fsum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return total


def reliability_table(probs: Sequence[float], labels: Sequence[int], bins: int = 10) -> list[dict[str, float]]:
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            rows.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": len(idx),
                    "mean_p": math.fsum(probs[i] for i in idx) / len(idx),
                    "frac_relevant": sum(labels[i] for i in idx) / len(idx),
                }
            )
    return rows


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = math.fsum(x) / n, math.fsum(y) / n
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else None


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            ranks[order[k]] = (i + j + 1) / 2
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    return pearson(_ranks(x), _ranks(y))


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall tau-b (ties handled); O(n^2), fine for a few thousand points."""
    n = len(x)
    if n < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denom if denom else None
