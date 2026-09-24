"""Domain re-calibration from human labels (Phase 6 follow-up).

The packaged calibration was fitted on benchmark data, and isotonic maps learn
the label base rate of that data (Phase 5 report, 5.3). Before trusting
calibrated scores on a service domain, re-fit the maps on labels from that
domain:

  1. sample_units   pick units from evaluation results, spread evenly over
                    JEV p within each question × language, and write a label
                    sheet that includes the (PII-masked) text a reviewer needs
  2. a person fills the `label` column with 1 (yes) or 0 (no)
  3. fit_calibration   fit new maps; keys with too few labels keep the base map

Sampling is stratified on JEV p. That does not bias the fit, because the maps
estimate P(label | p), not the overall label rate. Units from the human review
queue should not be the only source: they are concentrated on hard cases.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ragas_jev.privacy.pii_guard import mask_sample
from ragas_jev.schemas import RagSample, SampleResult
from ragas_jev.scoring.calibration import Calibrator, IsotonicMap, fit_isotonic

SHEET_COLUMNS = [
    "sample_id", "metric", "unit_id", "question_id", "lang", "jev_p",
    "question", "unit_text", "evidence", "label",
]
_QUESTION_HINT = {
    "chunk_relevance": "Does the passage (evidence) contain information that answers the question?",
    "claim_support": "Is the claim (unit_text) fully supported by the contexts (evidence)?",
    "ref_claim_coverage": "Is the reference claim (unit_text) fully supported by the contexts (evidence)?",
    "statement_relevance": "Is the statement (unit_text) relevant to answering the question?",
}


def _evidence(metric: str, index: int, sample: RagSample) -> str:
    if metric == "context_precision":
        return sample.contexts[index] if index < len(sample.contexts) else ""
    if metric in ("faithfulness", "context_recall"):
        return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(sample.contexts))
    return sample.answer


def sample_units(
    results: list[SampleResult],
    samples: dict[str, RagSample],
    *,
    per_key: int = 300,
    bins: int = 10,
    seed: int = 13,
    pii_masking: bool = True,
) -> list[dict]:
    """Rows for a label sheet: up to `per_key` Noul units per question × language."""
    masked = {sid: mask_sample(s)[0] if pii_masking else s for sid, s in samples.items()}
    pools: dict[tuple[str, str], list[list[tuple]]] = defaultdict(lambda: [[] for _ in range(bins)])
    for r in results:
        sample = masked.get(r.sample_id)
        lang = r.evaluation_model.get("lang")
        if sample is None or lang is None:
            continue
        for rec in r.units:
            d = rec.decision
            if d is None or d.primitive != "noul":
                continue
            b = min(bins - 1, int(d.jev_p * bins))
            pools[(d.question_id, lang)][b].append((r.sample_id, rec, sample))
    rng = random.Random(seed)
    rows = []
    for (question_id, lang), buckets in sorted(pools.items()):
        for bucket in buckets:
            rng.shuffle(bucket)
        # Round-robin over p bins so every part of the p range is covered.
        picked, i = [], 0
        while len(picked) < per_key and any(i < len(b) for b in buckets):
            for bucket in buckets:
                if i < len(bucket) and len(picked) < per_key:
                    picked.append(bucket[i])
            i += 1
        for sample_id, rec, sample in picked:
            rows.append(
                {
                    "sample_id": sample_id,
                    "metric": rec.metric,
                    "unit_id": rec.unit.unit_id,
                    "question_id": question_id,
                    "lang": lang,
                    "jev_p": f"{rec.decision.jev_p:.4f}",
                    "question": sample.question,
                    "unit_text": rec.unit.text,
                    "evidence": _evidence(rec.metric, rec.unit.index, sample),
                    "label": "",
                }
            )
    return rows


def write_sheet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def question_hint(question_id: str) -> str:
    return _QUESTION_HINT.get(question_id.split(".")[0], "")


@dataclass
class LabeledUnit:
    question_id: str
    lang: str
    jev_p: float
    label: int
    group: str  # sample_id: all units of a sample stay in one fold


def read_sheets(paths: list[Path]) -> list[LabeledUnit]:
    units = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                value = (row.get("label") or "").strip()
                if value == "":
                    continue
                if value not in ("0", "1"):
                    raise ValueError(f"{path}: label must be 0 or 1, got {value!r} ({row.get('sample_id')}/{row.get('unit_id')})")
                if not row.get("lang"):
                    raise ValueError(f"{path}: missing `lang` column (use `ragas-jev calibration sample` sheets)")
                units.append(LabeledUnit(row["question_id"], row["lang"], float(row["jev_p"]), int(value), row["sample_id"]))
    return units


def _ece(ps: list[float], ys: list[int], bins: int = 10) -> float:
    total = 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(ps) if b / bins <= p < (b + 1) / bins or (b == bins - 1 and p == 1.0)]
        if idx:
            total += len(idx) / len(ps) * abs(sum(ys[i] for i in idx) / len(idx) - math.fsum(ps[i] for i in idx) / len(idx))
    return total


@dataclass
class KeyReport:
    key: str
    labels: int
    positive_rate: float
    fitted: bool
    ece_raw: float | None = None
    ece_base: float | None = None
    ece_new_cv: float | None = None
    note: str = ""


@dataclass
class FitReport:
    keys: list[KeyReport] = field(default_factory=list)


def _cross_validated(units: list[LabeledUnit], folds: int, seed: int) -> list[float]:
    groups = sorted({u.group for u in units})
    random.Random(seed).shuffle(groups)
    fold_of = {g: i % folds for i, g in enumerate(groups)}
    out = [0.0] * len(units)
    for k in range(folds):
        train = [u for u in units if fold_of[u.group] != k]
        if not train:
            continue
        mapping = fit_isotonic([u.jev_p for u in train], [u.label for u in train])
        for i, u in enumerate(units):
            if fold_of[u.group] == k:
                out[i] = mapping(u.jev_p)
    return out


def fit_calibration(
    units: list[LabeledUnit],
    base: Calibrator | None = None,
    *,
    min_labels: int = 100,
    folds: int = 5,
    seed: int = 7,
    source: str = "domain re-calibration",
) -> tuple[Calibrator, FitReport]:
    """New maps per question × language; keys under `min_labels` keep the base map."""
    maps: dict[str, IsotonicMap] = dict(base.maps) if base else {}
    report = FitReport()
    by_key: dict[tuple[str, str], list[LabeledUnit]] = defaultdict(list)
    for u in units:
        by_key[(u.question_id, u.lang)].append(u)
    for (question_id, lang), group in sorted(by_key.items()):
        key = f"{question_id}:{lang}"
        ys = [u.label for u in group]
        raw = [u.jev_p for u in group]
        base_map = base.lookup(question_id, lang) if base else None
        entry = KeyReport(
            key=key,
            labels=len(group),
            positive_rate=sum(ys) / len(ys),
            fitted=False,
            ece_raw=_ece(raw, ys),
            ece_base=_ece([base_map(p) for p in raw], ys) if base_map else None,
        )
        if len(group) < min_labels:
            entry.note = f"fewer than {min_labels} labels: kept the base map" if base_map else f"fewer than {min_labels} labels: no map"
        elif len(set(ys)) < 2:
            entry.note = "all labels are the same: kept the base map"
        else:
            entry.ece_new_cv = _ece(_cross_validated(group, folds, seed), ys)
            maps[key] = fit_isotonic(raw, ys)
            entry.fitted = True
        report.keys.append(entry)
    return Calibrator(maps, source=source), report
