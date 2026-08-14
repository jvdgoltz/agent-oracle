#!/usr/bin/env python
"""Measure FastEmbed ONNX embedding quality and model latency on a portable dataset."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from agent_oracle.eval_dataset import validate_dataset

logger = logging.getLogger(__name__)

EVALS_DIR = Path.home() / ".agent-oracle" / "evals"
EMBED_CHUNK = 128
EMBED_BATCH = 32
ONNX_THREADS = 2
K_VALUES = (5, 10, 20)
MRR_K = 10


def load_dataset(path: Path) -> dict:
    """Load and validate a self-contained evaluation dataset."""
    dataset = json.loads(path.read_text())
    validate_dataset(dataset)
    return dataset


def embed_texts(model: TextEmbedding, texts: list[str]) -> tuple[np.ndarray, float]:
    """Embed texts and return vectors plus model-only inference elapsed milliseconds."""
    vectors: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(texts), EMBED_CHUNK):
        vectors.extend(model.embed(texts[start : start + EMBED_CHUNK], batch_size=EMBED_BATCH))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return np.asarray(vectors, dtype=np.float32), elapsed_ms


def embed_queries(model: TextEmbedding, questions: list[dict]) -> tuple[np.ndarray, float]:
    """Embed queries with the model's query encoder and time inference only."""
    query_texts = [question["question"] for question in questions]
    started = time.perf_counter()
    vectors = list(model.query_embed(query_texts))
    return np.asarray(vectors, dtype=np.float32), (time.perf_counter() - started) * 1000


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Return an L2-normalized copy of a vector matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def ranked_sessions(scores: np.ndarray, corpus: list[dict]) -> list[str]:
    """Deduplicate passages into a highest-scoring-first session ranking."""
    seen: set[str] = set()
    ranked: list[str] = []
    for index in np.argsort(-scores).tolist():
        session_id = corpus[index]["session_id"]
        if session_id not in seen:
            seen.add(session_id)
            ranked.append(session_id)
    return ranked


def evaluate_model(model_name: str, dataset: dict) -> dict:
    """Evaluate one ONNX model; similarity scoring is a non-model diagnostic."""
    corpus = dataset["corpus"]
    questions = dataset["questions"]
    model = TextEmbedding(model_name=model_name, threads=ONNX_THREADS)
    corpus_vectors, document_ms = embed_texts(model, [passage["content"] for passage in corpus])
    query_vectors, query_ms = embed_queries(model, questions)
    if corpus_vectors.ndim != 2 or query_vectors.ndim != 2:
        raise ValueError("Embedding model returned invalid vector shapes")
    if corpus_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("Document and query embedding dimensions differ")

    mrr_total = 0.0
    recall_totals = dict.fromkeys(K_VALUES, 0.0)
    scoring_ms = 0.0
    corpus_matrix = normalize(corpus_vectors)
    for question, query in zip(questions, normalize(query_vectors), strict=True):
        started = time.perf_counter()
        ranked = ranked_sessions(corpus_matrix @ query, corpus)
        scoring_ms += (time.perf_counter() - started) * 1000
        top = ranked[:MRR_K]
        if question["session_id"] in top:
            mrr_total += 1.0 / (top.index(question["session_id"]) + 1)
        for k in K_VALUES:
            recall_totals[k] += float(question["session_id"] in ranked[:k])

    count = max(1, len(questions))
    return {
        "model": model_name,
        "embedding_dimension": int(corpus_vectors.shape[1]),
        "mrr@10": mrr_total / count,
        "recall": {k: recall_totals[k] / count for k in K_VALUES},
        "document_embed_ms_total": document_ms,
        "document_embed_ms_per_passage": document_ms / max(1, len(corpus)),
        "query_embed_ms_per_query": query_ms / count,
        "in_memory_scoring_ms_per_query": scoring_ms / count,
    }


def format_report(dataset: dict, results: list[dict]) -> str:
    """Render only results from the current invocation as Markdown."""
    lines = [
        "# Embedding model evaluation",
        "",
        f"Dataset fingerprint: `{dataset['fingerprint']}`",
        "",
        "Similarity scoring is exhaustive in-memory diagnostics, not search-stack latency.",
        "Dimensions support adoption planning; schema migration and re-embedding are "
        "separate work.",
        "",
        "| Model | Dim | MRR@10 | R@5 | R@10 | R@20 | Document ms/passage | "
        "Query ms/query | Score ms/query |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['model']} | {result['embedding_dimension']} | {result['mrr@10']:.4f} | "
            f"{result['recall'][5]:.4f} | {result['recall'][10]:.4f} | "
            f"{result['recall'][20]:.4f} | {result['document_embed_ms_per_passage']:.2f} | "
            f"{result['query_embed_ms_per_query']:.2f} | "
            f"{result['in_memory_scoring_ms_per_query']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run the requested FastEmbed ONNX model comparison."""
    parser = argparse.ArgumentParser(description="Evaluate FastEmbed ONNX embedding models")
    parser.add_argument("--models", nargs="+", default=["BAAI/bge-small-en-v1.5"])
    parser.add_argument("--dataset", type=Path, default=EVALS_DIR / "dataset.json")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    results = [evaluate_model(model_name, dataset) for model_name in args.models]
    report = format_report(dataset, results)
    print("\n" + report)
    report_path = args.report or args.dataset.with_name("report_embeddings.md")
    report_path.write_text(report + "\n")
    logger.info("Current invocation report written to %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
