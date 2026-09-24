"""Command-line entry point: `ragas-jev healthcheck | evaluate`."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Optional

import typer

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

app = typer.Typer(add_completion=False, no_args_is_help=True)

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
    results = asyncio.run(_evaluate(samples, selected, judge, extractor, concurrency, cache, output))
    _print_summary(results)


async def _evaluate(
    samples: list[RagSample],
    metrics: tuple[str, ...],
    judge_name: str,
    extractor_name: str,
    concurrency: int,
    use_cache: bool,
    output: Path,
) -> list[SampleResult]:
    settings = get_settings()
    inner = MockJudge() if judge_name == "mock" else JevJudge.from_settings(settings)
    answer_cache = AnswerCache(settings.ragas_jev_cache_dir / "judge_answers.sqlite") if use_cache else None
    judge = CachedJudge(inner, answer_cache) if answer_cache else inner
    # Extractions are always cached: LLM output is not deterministic, and reusing
    # the same units is what keeps re-runs reproducible.
    extraction_cache = AnswerCache(settings.ragas_jev_cache_dir / "extractions.sqlite")
    units = LlmExtractor.from_settings(settings, extraction_cache) if extractor_name == "llm" else SentenceSplitExtractor()
    evaluator = Evaluator.from_settings(settings, judge, units)
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
            extraction_cache.close()


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


if __name__ == "__main__":
    app()
