"""Command-line entry point: `ragas-jev healthcheck | evaluate | review export | review import`."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Optional

import typer

from ragas_jev.audit.llm_judge import LlmJudge
from ragas_jev.audit.review_queue import apply_labels, export_review, read_labels
from ragas_jev.audit.router import DEFAULT_METRIC_POLICIES, Router, RoutingPolicy
from ragas_jev.cache import AnswerCache
from ragas_jev.config import get_settings
from ragas_jev.judge.base import JudgeQuestion
from ragas_jev.judge.cached import CachedJudge
from ragas_jev.judge.jev_client import JevJudge
from ragas_jev.judge.mock_jev import MockJudge
from ragas_jev.pipeline import Evaluator
from ragas_jev.preprocess.base import SentenceSplitExtractor
from ragas_jev.preprocess.llm_extractor import LlmExtractor
from ragas_jev.schemas import METRICS, RagSample, SampleResult
from ragas_jev.scoring.calibration import Calibrator
from ragas_jev.scoring.recalibration import fit_calibration, question_hint, read_sheets, sample_units, write_sheet

app = typer.Typer(add_completion=False, no_args_is_help=True)
review_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Human review queue")
app.add_typer(review_app, name="review")
calibration_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Domain re-calibration from human labels")
app.add_typer(calibration_app, name="calibration")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@app.command()
def healthcheck() -> None:
    """Check the LLM proxy and the JEV API with synthetic, non-sensitive inputs."""
    ok = asyncio.run(_healthcheck())
    raise typer.Exit(code=0 if ok else 1)


async def _healthcheck() -> bool:
    settings = get_settings()
    ok = True

    typer.echo(f"[proxy] {settings.openai_base_url}")
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key.get_secret_value())
        served = sorted(m.id for m in (await client.models.list()).data)
        typer.echo(f"  OK  models served: {', '.join(served)}")
        missing = [m for m in settings.available_model_list if m not in served]
        extra = [m for m in served if m not in settings.available_model_list]
        if missing:
            typer.echo(f"  WARN AVAILABLE_MODELS not served: {', '.join(missing)}")
        if extra:
            typer.echo(f"  INFO served but not in AVAILABLE_MODELS: {', '.join(extra)}")
        for role, model in [
            ("preprocessor", settings.preprocessor_model),
            ("auditor", settings.ragas_jev_auditor_model),
            ("strong_judge", settings.ragas_jev_strong_judge_model),
            ("baseline_judge", settings.ragas_jev_baseline_judge_model),
        ]:
            mark = "OK " if model in served else "ERR"
            ok &= model in served
            typer.echo(f"  {mark} {role} model: {model}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        typer.echo(f"  ERR {type(exc).__name__}: {exc}")

    typer.echo(f"[jev] model={settings.typesafe_default_model}")
    try:
        judge = JevJudge.from_settings(settings)
        try:
            typer.echo(f"  OK  models listed: {', '.join(await judge.list_models())}")
            probe = JudgeQuestion(
                unit_id="probe",
                question_id="healthcheck",
                primitive="noul",
                instructions="Does `text` mention a color?",
            )
            result = await judge.evaluate({"text": "The sky is blue."}, [probe])
            answer = result.answers["probe"]
            typer.echo(f"  OK  probe noul={answer.noul} (served by {answer.model})")
        finally:
            await judge.aclose()
    except Exception as exc:  # noqa: BLE001
        ok = False
        typer.echo(f"  ERR {type(exc).__name__}: {exc}")
    return ok


@app.command()
def evaluate(
    input: Path = typer.Option(..., "--input", "-i", help="JSONL file of RagSample records"),
    output: Path = typer.Option(..., "--output", "-o", help="JSONL file for SampleResult records"),
    metrics: str = typer.Option("all", help="Comma-separated metrics or 'all'"),
    judge: str = typer.Option("jev", help="jev | mock"),
    extractor: str = typer.Option("llm", help="llm (OPENAI_MODEL) | sentence (offline, no LLM calls)"),
    concurrency: int = typer.Option(4, help="Samples evaluated in parallel"),
    cache: bool = typer.Option(True, help="Reuse cached judge answers"),
    resume: bool = typer.Option(True, help="Skip sample_ids already in --output"),
    limit: Optional[int] = typer.Option(None, help="Evaluate at most N samples"),
    audit: Optional[bool] = typer.Option(None, "--audit/--no-audit", help="Escalate low-confidence JEV decisions to LLM judges (default: RAGAS_JEV_AUDIT_ENABLED)"),
    calibration: Optional[Path] = typer.Option(None, help="Calibration JSON (default: RAGAS_JEV_CALIBRATION_FILE, else the packaged file)"),
    no_calibration: bool = typer.Option(False, "--no-calibration", help="Use raw JEV probabilities"),
) -> None:
    """Evaluate RAG samples and append one result per line to --output."""
    selected = METRICS if metrics == "all" else tuple(m.strip() for m in metrics.split(",") if m.strip())
    samples = [RagSample.model_validate_json(line) for line in input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if resume and output.exists():
        done = {json.loads(line)["sample_id"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
        samples = [s for s in samples if s.sample_id not in done]
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        typer.echo("nothing to evaluate")
        return
    settings = get_settings()
    use_audit = settings.ragas_jev_audit_enabled if audit is None else audit
    calibrator = None if no_calibration else _load_calibrator(calibration or settings.calibration_path)
    results = asyncio.run(_evaluate(samples, selected, judge, extractor, concurrency, cache, output, use_audit, calibrator))
    _print_summary(results)


async def _evaluate(
    samples: list[RagSample],
    metrics: tuple[str, ...],
    judge_name: str,
    extractor_name: str,
    concurrency: int,
    use_cache: bool,
    output: Path,
    use_audit: bool = False,
    calibrator: Calibrator | None = None,
) -> list[SampleResult]:
    settings = get_settings()
    inner = MockJudge() if judge_name == "mock" else JevJudge.from_settings(settings)
    answer_cache = AnswerCache(settings.ragas_jev_cache_dir / "judge_answers.sqlite") if use_cache else None
    judge = CachedJudge(inner, answer_cache) if answer_cache else inner
    # Extractions are always cached: LLM output is not deterministic, and reusing
    # the same units is what keeps re-runs reproducible.
    extraction_cache = AnswerCache(settings.ragas_jev_cache_dir / "extractions.sqlite")
    units = LlmExtractor.from_settings(settings, extraction_cache) if extractor_name == "llm" else SentenceSplitExtractor()
    router = _build_router(settings) if use_audit and judge_name != "mock" else None
    evaluator = Evaluator.from_settings(settings, judge, units)
    evaluator.calibrator = calibrator
    evaluator.router = router
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("a", encoding="utf-8") as sink:

        def write(result: SampleResult) -> None:
            sink.write(result.model_dump_json() + "\n")
            sink.flush()

        try:
            return await evaluator.evaluate(samples, metrics, concurrency=concurrency, on_result=write)
        finally:
            close = getattr(judge, "aclose", None)
            if close is not None:
                await close()
            if answer_cache:
                answer_cache.close()
            if isinstance(units, LlmExtractor):
                await units.aclose()
            if router is not None:
                await router.auditor.aclose()
                if router.strong is not None:
                    await router.strong.aclose()
            extraction_cache.close()


def _load_calibrator(path: Path | None) -> Calibrator | None:
    if path is None or not path.exists():
        return None
    return Calibrator.load(path)


def _build_router(settings) -> Router:
    """Per-metric Phase 5 policies, unless RAGAS_JEV_CONF_ACCEPT/AUDIT are set explicitly."""
    explicit = {"ragas_jev_conf_accept", "ragas_jev_conf_audit"} & settings.model_fields_set
    policy = RoutingPolicy(settings.ragas_jev_conf_accept, settings.ragas_jev_conf_audit)
    return Router(
        auditor=LlmJudge.from_settings(settings, model=settings.ragas_jev_auditor_model),
        strong=LlmJudge.from_settings(settings, model=settings.ragas_jev_strong_judge_model),
        policy=policy,
        metric_policies={} if explicit else dict(DEFAULT_METRIC_POLICIES),
        decision_threshold=settings.ragas_jev_decision_threshold,
    )


def _read_results(path: Path) -> list[SampleResult]:
    return [SampleResult.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@review_app.command("export")
def review_export(
    input: Path = typer.Option(..., "--input", "-i", help="Results JSONL from `evaluate`"),
    out: Path = typer.Option(..., "--out", "-o", help="CSV to write"),
) -> None:
    """Write units flagged for human review to a CSV (PII-masked text only)."""
    n = export_review(_read_results(input), out)
    typer.echo(f"{n} units written to {out}; fill the `label` column with 1 (yes) or 0 (no)")


@review_app.command("import")
def review_import(
    input: Path = typer.Option(..., "--input", "-i", help="Results JSONL from `evaluate`"),
    labels: Path = typer.Option(..., "--labels", "-l", help="Reviewed CSV"),
    out: Path = typer.Option(..., "--out", "-o", help="Re-scored results JSONL"),
) -> None:
    """Apply human labels (decision_source=human) and re-score the affected samples."""
    settings = get_settings()
    evaluator = Evaluator.from_settings(settings, MockJudge(), SentenceSplitExtractor())
    label_map = read_labels(labels)
    total = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sink:
        for result in _read_results(input):
            updated, applied = apply_labels(result, label_map)
            total += applied
            sink.write((evaluator.rescore(updated) if applied else updated).model_dump_json() + "\n")
    typer.echo(f"applied {total} labels; re-scored results written to {out}")


def _print_summary(results: list[SampleResult]) -> None:
    statuses: dict[str, int] = {}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    typer.echo(f"samples: {len(results)}  status: {statuses}")
    keys = sorted({k for r in results for k in (*r.retrieval, *r.generation)})
    for key in keys:
        values = [v for r in results for v in [r.retrieval.get(key, r.generation.get(key))] if v is not None]
        if values:
            typer.echo(f"  {key:<28} mean={math.fsum(values) / len(values):.4f}  n={len(values)}")
    errors = [r for r in results if r.error]
    for r in errors[:5]:
        typer.echo(f"  [{r.status}] {r.sample_id}: {r.error}")


@calibration_app.command("sample")
def calibration_sample(
    results: Path = typer.Option(..., "--results", "-r", help="Results JSONL from `evaluate`"),
    samples: Path = typer.Option(..., "--samples", "-s", help="The RagSample JSONL that was evaluated"),
    out: Path = typer.Option(..., "--out", "-o", help="Label sheet CSV to write"),
    per_key: int = typer.Option(300, help="Units per question x language (spread over JEV p)"),
    seed: int = typer.Option(13),
) -> None:
    """Write a label sheet for re-calibration (PII-masked question, unit and evidence)."""
    settings = get_settings()
    sample_map = {
        s.sample_id: s
        for s in (RagSample.model_validate_json(l) for l in samples.read_text(encoding="utf-8").splitlines() if l.strip())
    }
    rows = sample_units(_read_results(results), sample_map, per_key=per_key, seed=seed, pii_masking=settings.ragas_jev_pii_masking)
    write_sheet(rows, out)
    counts: dict[str, int] = {}
    for row in rows:
        counts[f"{row['question_id']}:{row['lang']}"] = counts.get(f"{row['question_id']}:{row['lang']}", 0) + 1
    typer.echo(f"{len(rows)} units written to {out}")
    for key, n in sorted(counts.items()):
        typer.echo(f"  {key}: {n}  ({question_hint(key.split(':')[0])})")
    typer.echo("Fill `label` with 1 (yes) or 0 (no), then run `ragas-jev calibration fit`.")


@calibration_app.command("fit")
def calibration_fit(
    labels: list[Path] = typer.Option(..., "--labels", "-l", help="Labeled sheet(s) from `calibration sample`"),
    out: Path = typer.Option(..., "--out", "-o", help="Calibration JSON to write"),
    base: Optional[Path] = typer.Option(None, help="Maps kept for keys without enough labels (default: packaged calibration)"),
    min_labels: int = typer.Option(100, help="Minimum labels to fit a question x language key"),
) -> None:
    """Fit calibration maps on domain labels and report cross-validated ECE."""
    settings = get_settings()
    base_cal = _load_calibrator(base or settings.calibration_path)
    units = read_sheets(labels)
    if not units:
        typer.echo("no labeled rows found")
        raise typer.Exit(code=1)
    calibrator, report = fit_calibration(
        units, base_cal, min_labels=min_labels, source=f"domain re-calibration from {', '.join(p.name for p in labels)}"
    )
    calibrator.save(out)
    fmt = lambda x: "-" if x is None else f"{x:.3f}"
    typer.echo(f"{len(units)} labels -> {out}")
    typer.echo("  key | labels | yes rate | ECE raw | ECE base map | ECE new map (cross-validated) | result")
    for k in report.keys:
        status = "fitted" if k.fitted else k.note
        typer.echo(f"  {k.key} | {k.labels} | {k.positive_rate:.2f} | {fmt(k.ece_raw)} | {fmt(k.ece_base)} | {fmt(k.ece_new_cv)} | {status}")
    typer.echo(f"Use it with `ragas-jev evaluate --calibration {out}` or RAGAS_JEV_CALIBRATION_FILE={out}")


if __name__ == "__main__":
    app()
