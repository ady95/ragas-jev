import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))

from stats import auroc, binary_report, cohen_kappa, ece, pearson, percentile, spearman  # noqa: E402


def test_binary_report_counts():
    r = binary_report([1, 1, 0, 0], [1, 0, 1, 0])
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1)
    assert r["precision"] == r["recall"] == r["f1"] == r["accuracy"] == 0.5
    assert r["kappa"] == pytest.approx(0.0)


def test_kappa_perfect_and_chance():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)


def test_auroc():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    assert auroc([0.3, 0.4], [1, 1]) is None


def test_ece_perfectly_calibrated_is_zero():
    probs = [0.25] * 4 + [0.75] * 4
    labels = [1, 0, 0, 0] + [1, 1, 1, 0]
    assert ece(probs, labels) == pytest.approx(0.0)


def test_correlations_and_percentile():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [1, 10, 100]) == pytest.approx(1.0)
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)


def test_kendall_tau():
    from stats import kendall_tau

    assert kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert kendall_tau([1, 1, 1], [1, 2, 3]) is None
