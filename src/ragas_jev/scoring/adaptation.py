"""Prior-shift adaptation of calibration maps, without labels.

An isotonic map learns the "yes" rate of the data it was fitted on (Phase 5
report, 5.3). When the evaluation data has a different rate, the packaged map
can be worse than no calibration at all (Phase 6, section 9). Assuming only the
label rate changed (label shift), the map's outputs can be re-weighted to the
evaluation data's rate, which is estimated from its unlabeled JEV
probabilities by EM (Saerens, Latinne & Decaestecker, 2002):

    q' = w1 q / (w1 q + w0 (1 - q)),  w1 = pi_t / pi_s,  w0 = (1 - pi_t) / (1 - pi_s)

Validated on saved benchmark judgments in docs/reports/auto_calibration.md.
The adapted map is written as an ordinary isotonic map, so evaluation uses it
unchanged. It does not fix a map whose shape is wrong; that needs labels
(ragas_jev.scoring.recalibration).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ragas_jev.schemas import SampleResult
from ragas_jev.scoring.calibration import Calibrator, IsotonicMap

_CLIP = 0.01  # same bounds as fit_isotonic
_GRID = 200


def estimate_prior(qs: list[float], source_prior: float, iters: int = 500, tol: float = 1e-7) -> float:
    """EM estimate of the target "yes" rate from source-calibrated posteriors `qs`."""
    if not qs:
        raise ValueError("no probabilities to estimate a prior from")
    pi_s = _bounded(source_prior)
    pi_t = pi_s
    for _ in range(iters):
        adjust = prior_adjuster(pi_s, pi_t)
        new = sum(adjust(q) for q in qs) / len(qs)
        if abs(new - pi_t) < tol:
            return new
        pi_t = new
    return pi_t


def prior_adjuster(source_prior: float, target_prior: float) -> Callable[[float], float]:
    pi_s, pi_t = _bounded(source_prior), _bounded(target_prior)
    w1, w0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
    return lambda q: w1 * q / (w1 * q + w0 * (1 - q))


def adapted_map(base: IsotonicMap, source_prior: float, target_prior: float) -> IsotonicMap:
    """The base map followed by the prior adjustment, sampled on a grid over the base's range.

    Both steps are non-decreasing, so the result is a valid isotonic map.
    """
    adjust = prior_adjuster(source_prior, target_prior)
    lo, hi = base.xs[0], base.xs[-1]
    xs = sorted({*base.xs, *(lo + (hi - lo) * i / _GRID for i in range(_GRID + 1))})
    ys = [min(1 - _CLIP, max(_CLIP, adjust(base(x)))) for x in xs]
    for i in range(1, len(ys)):  # guard against float noise
        ys[i] = max(ys[i], ys[i - 1])
    return IsotonicMap(tuple(xs), tuple(ys))


def _bounded(p: float) -> float:
    return min(1 - 1e-3, max(1e-3, p))


@dataclass
class KeyAdaptation:
    key: str
    units: int
    base_key: str | None = None
    source_prior: float | None = None
    estimated_prior: float | None = None
    adapted: bool = False
    note: str = ""

    @property
    def shift(self) -> float | None:
        if self.source_prior is None or self.estimated_prior is None:
            return None
        return self.estimated_prior - self.source_prior


@dataclass
class AdaptationReport:
    keys: list[KeyAdaptation] = field(default_factory=list)

    def large_shifts(self, threshold: float) -> list[KeyAdaptation]:
        return [k for k in self.keys if k.shift is not None and abs(k.shift) >= threshold]


def raw_probabilities(results: Iterable[SampleResult]) -> dict[tuple[str, str], list[float]]:
    """Raw JEV p of every Noul decision, keyed by (question_id, lang)."""
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in results:
        lang = r.evaluation_model.get("lang")
        if lang is None:
            continue
        for rec in r.units:
            d = rec.decision
            if d is not None and d.primitive == "noul" and d.jev_p is not None:
                out[(d.question_id, lang)].append(d.jev_p)
    return out


def adapt_calibration(
    results: Iterable[SampleResult],
    base: Calibrator,
    *,
    min_units: int = 50,
    source: str = "",
) -> tuple[Calibrator, AdaptationReport]:
    """Re-weight each base map to the label rate estimated from `results`.

    Keys whose base map has no stored prior, or with fewer than `min_units` units,
    keep the base map. The adapted key is written language-specific even when it
    came from a "*" fallback map, and its prior becomes the estimated one, so
    adapting the output again changes nothing.
    """
    maps = dict(base.maps)
    priors = dict(base.priors)
    report = AdaptationReport()
    for (question_id, lang), ps in sorted(raw_probabilities(results).items()):
        key = f"{question_id}:{lang}"
        entry = KeyAdaptation(key=key, units=len(ps))
        base_key = base.resolve(question_id, lang)
        entry.base_key = base_key
        if base_key is None:
            entry.note = "no base map: fit one from labels (calibration sample / fit)"
        elif base_key not in base.priors:
            entry.note = "base map has no stored prior (calibration file from before 0.3)"
        else:
            entry.source_prior = base.priors[base_key]
            mapping = base.maps[base_key]
            entry.estimated_prior = estimate_prior([mapping(p) for p in ps], entry.source_prior)
            if len(ps) < min_units:
                entry.note = f"fewer than {min_units} units: kept the base map"
            else:
                maps[key] = adapted_map(mapping, entry.source_prior, entry.estimated_prior)
                priors[key] = entry.estimated_prior
                entry.adapted = True
        report.keys.append(entry)
    return Calibrator(maps, source=source or f"{base.source} (prior-adapted)", priors=priors), report
