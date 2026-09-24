"""Post-hoc calibration of JEV probabilities (plan Phase 5).

Phases 1-3 showed JEV's Noul p is skewed toward "yes" (best F1 thresholds
0.75-0.85, ECE up to 0.25 in Korean). Because Noul confidence is
max(p, 1 - p), a skewed p also puts the routing thresholds in the wrong place.
The fix is to map p to a calibrated probability first, per question and
language, and derive confidence from the calibrated value.

Maps are isotonic (pool-adjacent-violators) fits stored as monotone
breakpoints and applied with linear interpolation. The raw JEV value stays in
`Decision.jev_p`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ragas_jev.schemas import Decision
from ragas_jev.scoring.uncertainty import binary_confidence, normalized_entropy


@dataclass(frozen=True)
class IsotonicMap:
    xs: tuple[float, ...]
    ys: tuple[float, ...]

    def __call__(self, p: float) -> float:
        xs, ys = self.xs, self.ys
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= p:
                lo = mid
            else:
                hi = mid
        span = xs[hi] - xs[lo]
        w = (p - xs[lo]) / span if span else 0.0
        return ys[lo] + w * (ys[hi] - ys[lo])

    def to_dict(self) -> dict:
        return {"xs": list(self.xs), "ys": list(self.ys)}

    @classmethod
    def from_dict(cls, data: dict) -> IsotonicMap:
        return cls(tuple(data["xs"]), tuple(data["ys"]))


def fit_isotonic(ps: list[float], ys: list[int], *, clip: float = 0.01) -> IsotonicMap:
    """Pool-adjacent-violators fit of P(y=1 | p); outputs are clipped to [clip, 1-clip]."""
    if not ps:
        raise ValueError("cannot fit calibration without data")
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    # Each block: [sum_p, sum_y, count]
    blocks: list[list[float]] = []
    for i in order:
        blocks.append([ps[i], float(ys[i]), 1.0])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]:
            last = blocks.pop()
            blocks[-1] = [a + b for a, b in zip(blocks[-1], last)]
    xs = [b[0] / b[2] for b in blocks]
    vals = [min(1 - clip, max(clip, b[1] / b[2])) for b in blocks]
    # Merge equal x (ties in p) and keep strictly increasing breakpoints.
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(xs, vals):
        if out_x and math.isclose(x, out_x[-1]):
            out_y[-1] = max(out_y[-1], y)
        else:
            out_x.append(x)
            out_y.append(y)
    return IsotonicMap(tuple(out_x), tuple(out_y))


@dataclass
class Calibrator:
    """Maps keyed by "<question_id>:<lang>"; "<question_id>:*" is the language fallback."""

    maps: dict[str, IsotonicMap] = field(default_factory=dict)
    source: str = ""

    def lookup(self, question_id: str, lang: str) -> IsotonicMap | None:
        return self.maps.get(f"{question_id}:{lang}") or self.maps.get(f"{question_id}:*")

    def apply(self, decision: Decision, lang: str) -> Decision:
        """Calibrate a Noul decision in place of its p; other primitives pass through."""
        if decision.primitive != "noul":
            return decision
        mapping = self.lookup(decision.question_id, lang)
        if mapping is None:
            return decision
        p = mapping(decision.jev_p)
        confidence = binary_confidence(p)
        return decision.model_copy(
            update={
                "p": p,
                "distribution": {"true": p, "false": 1.0 - p},
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
                "entropy": normalized_entropy([p, 1.0 - p]),
                "calibrated": True,
            }
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"source": self.source, "maps": {k: m.to_dict() for k, m in sorted(self.maps.items())}}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls({k: IsotonicMap.from_dict(v) for k, v in data["maps"].items()}, data.get("source", ""))
