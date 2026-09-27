"""Prior-shift adaptation of calibration maps (no labels, no API calls)."""

import random

from ragas_jev.schemas import Decision, EvalUnit, SampleResult, UnitRecord
from ragas_jev.scoring.adaptation import adapt_calibration, adapted_map, estimate_prior, prior_adjuster
from ragas_jev.scoring.calibration import Calibrator, IsotonicMap, fit_isotonic
from ragas_jev.scoring.recalibration import LabeledUnit, fit_calibration


def _draw(n: int, rate: float, rng: random.Random) -> tuple[list[float], list[int]]:
    """JEV-like scores with fixed class-conditional distributions: only the label rate varies."""
    ps, ys = [], []
    for _ in range(n):
        y = int(rng.random() < rate)
        ps.append(0.45 + 0.55 * rng.random() ** 0.5 if y else 0.9 * rng.random())
        ys.append(y)
    return ps, ys


def _results(ps: list[float], question_id: str = "claim_support.v1", lang: str = "ko") -> list[SampleResult]:
    out = []
    for i, p in enumerate(ps):
        d = Decision(unit_id=f"claim_{i}", primitive="noul", question_id=question_id, distribution={"true": p, "false": 1 - p},
                     p=p, confidence=max(p, 1 - p), uncertainty=1 - max(p, 1 - p), entropy=0.0, jev_model="mock", jev_p=p)
        unit = EvalUnit(unit_id=f"claim_{i}", kind="claim", index=i, text="t")
        out.append(SampleResult(sample_id=f"s{i}", evaluation_model={"lang": lang}, units=[UnitRecord(metric="faithfulness", unit=unit, decision=d)]))
    return out


def test_estimate_prior_recovers_a_shifted_rate():
    rng = random.Random(3)
    src_p, src_y = _draw(4000, 0.3, rng)
    base = fit_isotonic(src_p, src_y)
    tgt_p, _ = _draw(4000, 0.7, rng)
    est = estimate_prior([base(p) for p in tgt_p], sum(src_y) / len(src_y))
    assert abs(est - 0.7) < 0.05


def test_no_shift_leaves_the_map_nearly_unchanged():
    rng = random.Random(4)
    src_p, src_y = _draw(4000, 0.5, rng)
    base = fit_isotonic(src_p, src_y)
    prior = sum(src_y) / len(src_y)
    tgt_p, _ = _draw(4000, 0.5, rng)
    est = estimate_prior([base(p) for p in tgt_p], prior)
    adapted = adapted_map(base, prior, est)
    assert abs(est - prior) < 0.05
    assert max(abs(adapted(x / 100) - base(x / 100)) for x in range(101)) < 0.06


def test_adapted_map_is_monotone_and_clipped():
    base = IsotonicMap((0.2, 0.5, 0.8, 0.95), (0.01, 0.2, 0.6, 0.99))
    for target in (0.05, 0.5, 0.95):
        m = adapted_map(base, 0.3, target)
        assert all(a <= b for a, b in zip(m.ys, m.ys[1:]))
        assert all(0.01 <= y <= 0.99 for y in m.ys)
    assert prior_adjuster(0.3, 0.3)(0.42) == 0.42


def test_adapt_calibration_writes_language_keys_and_is_idempotent():
    rng = random.Random(5)
    src_p, src_y = _draw(3000, 0.3, rng)
    base = Calibrator({"claim_support.v1:*": fit_isotonic(src_p, src_y)}, "test", {"claim_support.v1:*": sum(src_y) / len(src_y)})
    tgt_p, _ = _draw(1500, 0.75, rng)
    adapted, report = adapt_calibration(_results(tgt_p), base)
    (entry,) = report.keys
    assert entry.adapted and entry.key == "claim_support.v1:ko" and entry.base_key == "claim_support.v1:*"
    assert abs(entry.estimated_prior - 0.75) < 0.06
    assert "claim_support.v1:ko" in adapted.maps and "claim_support.v1:*" in adapted.maps  # fallback kept
    assert report.large_shifts(0.15) == [entry]
    _, again = adapt_calibration(_results(tgt_p), adapted)
    assert abs(again.keys[0].estimated_prior - entry.estimated_prior) < 0.01


def test_adapt_calibration_skips_keys_it_cannot_adapt():
    base = Calibrator({"claim_support.v1:ko": IsotonicMap((0.1, 0.9), (0.1, 0.9))}, "old file without priors")
    _, report = adapt_calibration(_results([0.8] * 60) + _results([0.7] * 60, question_id="chunk_relevance.v3"), base)
    notes = {k.key: k.note for k in report.keys}
    assert "no stored prior" in notes["claim_support.v1:ko"]
    assert "no base map" in notes["chunk_relevance.v3:ko"]
    base.priors["claim_support.v1:ko"] = 0.5
    _, report = adapt_calibration(_results([0.8] * 10), base, min_units=50)
    assert not report.keys[0].adapted and "fewer than 50" in report.keys[0].note


def test_priors_round_trip_and_come_from_fit(tmp_path):
    units = [LabeledUnit("claim_support.v1", "ko", i / 200, int(i % 4 != 0), f"s{i // 5}") for i in range(200)]
    cal, _ = fit_calibration(units, None, min_labels=100)
    assert cal.priors == {"claim_support.v1:ko": 0.75}
    path = tmp_path / "cal.json"
    cal.save(path)
    assert Calibrator.load(path).priors == {"claim_support.v1:ko": 0.75}


def test_packaged_calibration_has_priors_for_every_map():
    from ragas_jev.config import Settings

    cal = Calibrator.load(Settings().calibration_path)
    assert set(cal.priors) == set(cal.maps)
    assert 0.1 < cal.priors["chunk_relevance.v2:ko"] < 0.2  # MIRACL-ko: few relevant passages
