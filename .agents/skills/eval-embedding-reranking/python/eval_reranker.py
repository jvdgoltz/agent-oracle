#!/usr/bin/env python
"""Measure FastEmbed ONNX reranker quality and model latency on a portable dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
EVALS_DIR = Path.home() / ".agent-oracle" / "evals"
CANDIDATE_K = 50
ONNX_THREADS = 2
K_VALUES = (5, 10, 20)


def load_dataset(path: Path) -> dict:
    """Load and validate a self-contained evaluation dataset."""
    dataset = json.loads(path.read_text())
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: dict) -> None:
    """Require a schema-versioned corpus containing every gold session."""
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported eval dataset schema version")
    if not dataset.get("fingerprint"):
        raise ValueError("Eval dataset has no fingerprint")
    questions = dataset.get("questions")
    corpus = dataset.get("corpus")
    if not isinstance(questions, list) or not isinstance(corpus, list):
        raise ValueError("Eval dataset requires questions and corpus lists")
    if dataset["fingerprint"] != dataset_fingerprint(dataset):
        raise ValueError("Eval dataset fingerprint does not match its contents")
    sessions = {passage.get("session_id") for passage in corpus}
    for question in questions:
        if question.get("session_id") not in sessions:
            raise ValueError(f"Question {question.get('id')} has no gold session passage in corpus")


def dataset_fingerprint(dataset: dict) -> str:
    """Return the generator-compatible fingerprint for portable dataset contents."""
    payload = {
        "schema_version": dataset["schema_version"],
        "questions": dataset["questions"],
        "corpus": dataset["corpus"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Return an L2-normalized copy of a vector matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def dedup_sessions(indices: list[int], corpus: list[dict]) -> list[str]:
    """Map ordered passage indices to one ordered occurrence per session."""
    seen: set[str] = set()
    ranked: list[str] = []
    for index in indices:
        session_id = corpus[index]["session_id"]
        if session_id not in seen:
            seen.add(session_id)
            ranked.append(session_id)
    return ranked


def compute_candidates(dataset: dict, base_model: str) -> dict[str, list[int]]:
    """Compute fresh base-retriever candidates from this dataset for this invocation."""
    corpus = dataset["corpus"]
    questions = dataset["questions"]
    model = TextEmbedding(model_name=base_model, threads=ONNX_THREADS)
    corpus_vectors = np.asarray(
        list(model.embed([passage["content"] for passage in corpus])), dtype=np.float32
    )
    query_vectors = np.asarray(
        list(model.query_embed([question["question"] for question in questions])), dtype=np.float32
    )
    scores = normalize(query_vectors) @ normalize(corpus_vectors).T
    return {
        question["id"]: np.argsort(-row)[:CANDIDATE_K].tolist()
        for question, row in zip(questions, scores, strict=True)
    }


def metrics(ranked: dict[str, list[str]], questions: list[dict]) -> dict:
    """Compute MRR@10 and Recall@k for session-level rankings."""
    mrr_total = 0.0
    recall_totals = dict.fromkeys(K_VALUES, 0.0)
    for question in questions:
        ranked_sessions = ranked[question["id"]]
        top = ranked_sessions[:10]
        if question["session_id"] in top:
            mrr_total += 1.0 / (top.index(question["session_id"]) + 1)
        for k in K_VALUES:
            recall_totals[k] += float(question["session_id"] in ranked_sessions[:k])
    count = max(1, len(questions))
    return {"mrr@10": mrr_total / count, "recall": {k: recall_totals[k] / count for k in K_VALUES}}


def rerank_model(
    reranker_name: str, dataset: dict, candidates: dict[str, list[int]]
) -> tuple[dict, float]:
    """Run one cross-encoder over fresh current-run candidates and time it."""
    corpus = dataset["corpus"]
    questions = dataset["questions"]
    reranker = TextCrossEncoder(reranker_name, threads=ONNX_THREADS)
    ranked: dict[str, list[str]] = {}
    elapsed_ms = 0.0
    for question in questions:
        indices = candidates[question["id"]]
        documents = [corpus[index]["content"] for index in indices]
        started = time.perf_counter()
        scores = list(reranker.rerank(question["question"], documents))
        elapsed_ms += (time.perf_counter() - started) * 1000
        order = sorted(range(len(indices)), key=lambda index: -scores[index])
        ranked[question["id"]] = dedup_sessions([indices[index] for index in order], corpus)
    return metrics(ranked, questions), elapsed_ms / max(1, len(questions))


def format_report(
    dataset: dict, base_model: str, baseline: dict, rows: list[tuple[str, dict, float]]
) -> str:
    """Render only results created by this invocation as Markdown."""
    lines = [
        "# Reranker evaluation",
        "",
        f"Dataset fingerprint: `{dataset['fingerprint']}`",
        f"Base ONNX embedder: `{base_model}`. Candidates were recomputed for this run.",
        "Rerank latency measures only cross-encoder inference over the candidate passages.",
        "",
        f"Base MRR@10: {baseline['mrr@10']:.4f}",
        "",
        "| Reranker | MRR@10 | Recall@5 | Recall@10 | Recall@20 | Rerank ms/query |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, result, rerank_ms in rows:
        lines.append(
            f"| {name} | {result['mrr@10']:.4f} | {result['recall'][5]:.4f} | "
            f"{result['recall'][10]:.4f} | {result['recall'][20]:.4f} | {rerank_ms:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Evaluate requested FastEmbed ONNX cross-encoders on fresh candidates."""
    parser = argparse.ArgumentParser(description="Evaluate FastEmbed ONNX rerankers")
    parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--rerankers", nargs="+", default=["Xenova/ms-marco-MiniLM-L-6-v2"])
    parser.add_argument("--dataset", type=Path, default=EVALS_DIR / "dataset.json")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    candidates = compute_candidates(dataset, args.base_model)
    corpus = dataset["corpus"]
    baseline = metrics(
        {
            question_id: dedup_sessions(indices, corpus)
            for question_id, indices in candidates.items()
        },
        dataset["questions"],
    )
    rows = [
        (reranker_name, *rerank_model(reranker_name, dataset, candidates))
        for reranker_name in args.rerankers
    ]
    report = format_report(dataset, args.base_model, baseline, rows)
    print("\n" + report)
    report_path = args.report or args.dataset.with_name("report_rerankers.md")
    report_path.write_text(report + "\n")
    logger.info("Current invocation report written to %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
