"""Phase 6 baseline: the official `ragas` library on the Phase 1-3 datasets.

ragas 0.4 metrics with its own decomposition and prompts, judged by the
benchmark baseline model (RAGAS_JEV_BASELINE_JUDGE_MODEL, gpt-6-sol):

  ContextRelevance  MIRACL (question + contexts; MIRACL has no reference answers,
                    so ragas' reference-based ContextPrecision cannot be used)
  Faithfulness      RAGTruth QA, Korean synthetic hallucination set
  ContextRecall     Phase 3 recall sets (full / partial / none)

  ContextPrecision  Phase 3 references + MIRACL labels (prepare_cp_reference.py)
  AnswerRelevancy   Phase 4 sets. ragas scores it with embeddings; the proxy has no
                    embeddings endpoint, so a local multilingual SentenceTransformer
                    is used (--embedding-model, `embeddings` dependency group)

ragas sends temperature=1.0, which GPT-5+ models reject through the proxy, so
the client drops `temperature` before every call. Only public / synthetic data
is used; ragas calls do not go through the PII guard.

Usage:
  uv run --group benchmark python benchmark/run_ragas_baseline.py
Results: .cache/phase6/ragas.<dataset>.<metric>.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

OUT = Path(".cache/phase6")
JOBS = [
    ("context_relevance", "miracl_ko_dev_relevant"),
    ("context_relevance", "miracl_en_dev_relevant"),
    ("context_relevance", "miracl_ko_dev_non_relevant"),
    ("faithfulness", "ragtruth_qa_test"),
    ("faithfulness", "ko_hallu_miracl"),
    ("context_recall", "recall_miracl_ko"),
    ("context_recall", "recall_miracl_en"),
    ("context_precision", "cp_ref_miracl_ko"),
    ("context_precision", "cp_ref_miracl_en"),
    ("answer_relevancy", "relevancy_wikieval"),
    ("answer_relevancy", "relevancy_miracl_ko"),
    ("answer_relevancy", "relevancy_miracl_en"),
]
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def make_client(settings):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key.get_secret_value())
    create = client.chat.completions.create

    async def create_without_temperature(**kwargs):
        kwargs.pop("temperature", None)  # GPT-5+ models reject it
        return await create(**kwargs)

    client.chat.completions.create = create_without_temperature
    return client


def build_metric(name: str, llm, embedding_model: str = EMBEDDING_MODEL):
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, ContextRelevance, Faithfulness

    if name == "answer_relevancy":
        from ragas.embeddings import HuggingFaceEmbeddings

        return AnswerRelevancy(llm=llm, embeddings=HuggingFaceEmbeddings(model=embedding_model))

    metrics = {
        "context_relevance": ContextRelevance,
        "context_precision": ContextPrecision,  # with reference; rank-aware average precision
        "faithfulness": Faithfulness,
        "context_recall": ContextRecall,
    }
    return metrics[name](llm=llm)


def kwargs_for(name: str, sample: dict) -> dict:
    if name == "context_relevance":
        return {"user_input": sample["question"], "retrieved_contexts": sample["contexts"]}
    if name == "context_precision":
        return {"user_input": sample["question"], "reference": sample["reference"], "retrieved_contexts": sample["contexts"]}
    if name == "answer_relevancy":
        return {"user_input": sample["question"], "response": sample["answer"]}
    if name == "faithfulness":
        return {"user_input": sample["question"], "response": sample["answer"], "retrieved_contexts": sample["contexts"]}
    return {"user_input": sample["question"], "retrieved_contexts": sample["contexts"], "reference": sample["reference"]}


async def run_job(
    metric_name: str, dataset: str, llm, concurrency: int, limit: int | None, tag: str = "", embedding_model: str = EMBEDDING_MODEL
) -> None:
    path = OUT / f"ragas.{dataset}.{metric_name}{'.' + tag if tag else ''}.jsonl"
    done = {json.loads(l)["sample_id"] for l in path.open(encoding="utf-8")} if path.exists() else set()
    samples = [json.loads(l) for l in open(f"benchmark/datasets/{dataset}/samples.jsonl", encoding="utf-8")]
    samples = [s for s in samples if s["sample_id"] not in done][:limit]
    metric = build_metric(metric_name, llm, embedding_model)
    sem = asyncio.Semaphore(concurrency)

    async def one(s: dict) -> dict:
        async with sem:
            started = perf_counter()
            error = None
            score = None
            for attempt in range(2):
                try:
                    score = (await metric.ascore(**kwargs_for(metric_name, s))).value
                    error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {str(exc)[:200]}"
            return {"sample_id": s["sample_id"], "score": score, "latency": perf_counter() - started, "error": error}

    with path.open("a", encoding="utf-8") as sink:
        for coro in asyncio.as_completed([one(s) for s in samples]):
            row = await coro
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
    print(f"   {dataset} / {metric_name}: {len(samples)} new samples")


async def main_async(args: argparse.Namespace) -> None:
    from ragas.llms import llm_factory

    from ragas_jev.config import get_settings

    settings = get_settings()
    model = args.model or settings.ragas_jev_baseline_judge_model
    llm = llm_factory(model, client=make_client(settings))
    OUT.mkdir(parents=True, exist_ok=True)
    for metric_name, dataset in JOBS:
        if args.only and metric_name not in args.only:
            continue
        print(f"== ragas {metric_name} on {dataset} ({model})")
        if args.datasets and dataset not in args.datasets:
            continue
        await run_job(metric_name, dataset, llm, args.concurrency, args.limit, args.tag, args.embedding_model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="at most N new samples per job (smoke tests)")
    parser.add_argument("--only", nargs="*", default=None, help="subset of metrics")
    parser.add_argument("--datasets", nargs="*", default=None, help="subset of datasets")
    parser.add_argument("--tag", default="", help="write to ragas.<dataset>.<metric>.<tag>.jsonl (repeat runs)")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="local SentenceTransformer for AnswerRelevancy")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
