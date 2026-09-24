"""Human review queue: export flagged units to CSV, import labels, re-score.

The CSV holds only what the results already hold, i.e. PII-masked text.
Reviewers fill the `label` column with 1 (yes: supported / relevant) or
0 (no); rows left empty are ignored on import.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from ragas_jev.schemas import SampleResult

COLUMNS = [
    "sample_id", "metric", "unit_id", "unit_text", "question_id",
    "jev_p", "final_p", "decision_source", "auditor_model", "label",
]


def export_review(results: Iterable[SampleResult], path: Path) -> int:
    rows = []
    for result in results:
        for record in result.units:
            d = record.decision
            if d is None or not d.needs_human_review:
                continue
            rows.append(
                {
                    "sample_id": result.sample_id,
                    "metric": record.metric,
                    "unit_id": record.unit.unit_id,
                    "unit_text": record.unit.text,
                    "question_id": d.question_id,
                    "jev_p": f"{d.jev_p:.4f}",
                    "final_p": f"{d.p:.4f}",
                    "decision_source": d.decision_source,
                    "auditor_model": d.auditor_model or "",
                    "label": "",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so the file opens with Korean text intact in Excel
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_labels(path: Path) -> dict[tuple[str, str, str], int]:
    labels = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            value = (row.get("label") or "").strip()
            if value == "":
                continue
            if value not in ("0", "1"):
                raise ValueError(f"label must be 0 or 1, got {value!r} for {row['sample_id']}/{row['unit_id']}")
            labels[(row["sample_id"], row["metric"], row["unit_id"])] = int(value)
    return labels


def apply_labels(result: SampleResult, labels: dict[tuple[str, str, str], int]) -> tuple[SampleResult, int]:
    """Replace labeled decisions with the human verdict; returns the updated copy and count."""
    updated = result.model_copy(deep=True)
    applied = 0
    for record in updated.units:
        key = (result.sample_id, record.metric, record.unit.unit_id)
        if key not in labels or record.decision is None:
            continue
        p = float(labels[key])
        record.decision = record.decision.model_copy(
            update={
                "p": p,
                "distribution": {"true": p, "false": 1.0 - p},
                "confidence": 1.0,
                "uncertainty": 0.0,
                "entropy": 0.0,
                "decision_source": "human",
                "needs_human_review": False,
            }
        )
        applied += 1
    return updated, applied
