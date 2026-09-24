"""Domain re-calibration: label sheet sampling, fitting, and the CLI round trip (no API calls)."""

import csv
import json
import random

import pytest
from typer.testing import CliRunner

from ragas_jev.cli import app
from ragas_jev.judge.mock_jev import MockJudge
from ragas_jev.pipeline import Evaluator
from ragas_jev.preprocess.base import SentenceSplitExtractor
from ragas_jev.schemas import RagSample
from ragas_jev.scoring.calibration import Calibrator, IsotonicMap
from ragas_jev.scoring.recalibration import LabeledUnit, fit_calibration, read_sheets, sample_units, write_sheet


def _samples(n: int) -> list[RagSample]:
    return [
        RagSample(
            sample_id=f"s{i}",
            question=f"질문 {i}번은 무엇인가?",
            answer=f"답변 {i}입니다. 담당자 연락처는 010-1234-56{i % 10}{i % 10} 입니다.",  # synthetic number
            contexts=[f"문서 {i}의 첫 번째 내용.", f"문서 {i}의 두 번째 내용."],
        )
        for i in range(n)
    ]


async def test_sample_units_masks_pii_and_spreads_over_p():
    samples = _samples(30)
    evaluator = Evaluator(MockJudge(), SentenceSplitExtractor())
    results = [await evaluator.evaluate_sample(s, ("context_precision", "faithfulness")) for s in samples]
    rows = sample_units(results, {s.sample_id: s for s in samples}, per_key=20)
    by_key = {}
    for r in rows:
        by_key.setdefault((r["question_id"], r["lang"]), []).append(r)
    assert set(by_key) == {("chunk_relevance.v2", "ko"), ("claim_support.v1", "ko")}
    assert all(len(v) == 20 for v in by_key.values())
    assert all("010-1234" not in r["evidence"] + r["unit_text"] for r in rows)
    ps = sorted(float(r["jev_p"]) for r in by_key[("chunk_relevance.v2", "ko")])
    assert ps[0] < 0.3 and ps[-1] > 0.7  # both ends of the p range are sampled
    chunk_row = by_key[("chunk_relevance.v2", "ko")][0]
    assert chunk_row["evidence"].startswith("문서")  # the judged passage itself


def _skewed_units(n: int, seed: int = 1) -> list[LabeledUnit]:
    """JEV p that overstates 'yes': true P(yes | p) = p ** 3."""
    rng = random.Random(seed)
    units = []
    for i in range(n):
        p = rng.random()
        units.append(LabeledUnit("claim_support.v1", "ko", p, int(rng.random() < p**3), f"s{i // 5}"))
    return units


def test_fit_reduces_ece_on_skewed_labels():
    cal, report = fit_calibration(_skewed_units(1500), None, min_labels=100)
    key = report.keys[0]
    assert key.fitted and key.labels == 1500
    assert key.ece_new_cv < key.ece_raw / 2
    assert cal.lookup("claim_support.v1", "ko")(0.8) == pytest.approx(0.8**3, abs=0.12)


def test_keys_below_min_labels_keep_the_base_map():
    base = Calibrator({"claim_support.v1:ko": IsotonicMap((0.0, 1.0), (0.3, 0.7))})
    cal, report = fit_calibration(_skewed_units(50), base, min_labels=100)
    assert not report.keys[0].fitted and "kept the base map" in report.keys[0].note
    assert cal.lookup("claim_support.v1", "ko")(0.5) == pytest.approx(0.5)


def test_single_class_labels_are_not_fitted():
    units = [LabeledUnit("chunk_relevance.v2", "en", i / 200, 0, f"s{i}") for i in range(200)]
    _, report = fit_calibration(units, None, min_labels=100)
    assert not report.keys[0].fitted


def test_sheet_roundtrip_skips_unlabeled_rows(tmp_path):
    path = tmp_path / "sheet.csv"
    rows = [
        {"sample_id": "a", "metric": "faithfulness", "unit_id": "claim_0", "question_id": "claim_support.v1",
         "lang": "ko", "jev_p": "0.9000", "question": "q", "unit_text": "c", "evidence": "e", "label": label}
        for label in ("1", "", "0")
    ]
    write_sheet(rows, path)
    units = read_sheets([path])
    assert [u.label for u in units] == [1, 0]
    rows[0]["label"] = "yes"
    write_sheet(rows, path)
    with pytest.raises(ValueError):
        read_sheets([path])


def test_cli_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text("\n".join(s.model_dump_json() for s in _samples(60)), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, [
        "evaluate", "-i", str(samples_path), "-o", "results.jsonl", "--judge", "mock", "--extractor", "sentence",
        "--no-audit", "--no-cache", "--metrics", "faithfulness",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])["evaluation_model"]["lang"] == "ko"

    result = runner.invoke(app, ["calibration", "sample", "-r", "results.jsonl", "-s", str(samples_path), "-o", "sheet.csv", "--per-key", "150"])
    assert result.exit_code == 0, result.output
    with (tmp_path / "sheet.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(3)
    for row in rows:  # a reviewer who says yes with probability p ** 2
        row["label"] = str(int(rng.random() < float(row["jev_p"]) ** 2))
    with (tmp_path / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = runner.invoke(app, ["calibration", "fit", "-l", "sheet.csv", "-o", "domain_cal.json", "--min-labels", "50"])
    assert result.exit_code == 0, result.output
    assert "claim_support.v1:ko" in result.output and "fitted" in result.output
    cal = Calibrator.load(tmp_path / "domain_cal.json")
    assert cal.lookup("claim_support.v1", "ko") is not None
    assert cal.lookup("chunk_relevance.v2", "en") is not None  # kept from the packaged base
