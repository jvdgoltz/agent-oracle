---
name: eval-embedding-reranking
description: Build a portable synthetic dataset from archived coding-agent sessions and compare local FastEmbed ONNX embedding and reranking models for retrieval quality and model-only latency.
---

# Evaluate embedding and reranking models

Use this skill to compare embedding models or cross-encoder rerankers before
choosing a retrieval model. It evaluates model inference and in-memory ranking
on a fixed portable dataset; it does **not** benchmark Agent Oracle's database,
FTS, hybrid retrieval, API, or full search-stack latency.

## Inputs and outputs

- Build input: the Agent Oracle archive at `~/.agent-oracle/index.db`.
- Portable dataset: `~/.agent-oracle/evals/dataset.json`.
- Embedding report: `~/.agent-oracle/evals/report_embeddings.md`.
- Reranker report: `~/.agent-oracle/evals/report_rerankers.md`.

The constructed dataset is schema-versioned and fingerprinted. It contains both
the questions and the exact searchable corpus passages, including passage and
session IDs. Once built, either evaluator requires only this JSON file; neither
opens the database, makes a database snapshot, nor uses persistent embedding or
candidate caches.

## Workflow

1. Research candidates before proposing them.

   Check current primary benchmark sources for embedding models (for example,
   MTEB and BEIR) and cross-encoders. Shortlist FastEmbed-supported **ONNX**
   models that can run locally. This skill does not support PyTorch or MPS.
   Compare quality and model latency; faster/smaller variants are preferred when
   their measured quality is within the agreed tolerance.

2. Build or extend the portable dataset.

   ```bash
   uv run python .agents/skills/eval-embedding-reranking/python/generate_eval_dataset.py
   ```

   The builder uses `Store` only while constructing the dataset. It never sends
   every archived session to an LLM: it selects at most 50 new sessions by
   default (hard maximum 100). It clusters **existing stored session-summary
   embeddings** to choose representative sessions, then adds a bounded,
   deduplicated sample for each eligible entity value. It never re-indexes,
   re-embeds, or enriches sessions. Sessions already recorded as processed in
   the artifact are excluded before selection.

   Run `--dry-run` first to print the deterministic selection, the maximum LLM
   call estimate, and entity coverage that could not fit under the global cap.
   Use `--seed`, `--max-sessions`, `--cluster-samples`, and
   `--sessions-per-entity` to control the selection. The builder prints that
   preview before it creates an OpenAI client. For every selected session, it
   saves filtered searchable messages as corpus passages and asks an LLM to
   create 1–3 recall-style paraphrase questions. It incrementally handles new
   sessions and preserves the existing artifact's corpus and questions.

   Use `--reset` to deliberately start over. It removes that output file and
   any legacy matching checkpoint before generation, so regenerated questions
   cannot be appended to prior ones or duplicated. `--out` selects a separate
   dataset. The dataset records its processed session IDs, including sessions
   that produced no accepted questions.

   This is a **synthetic paraphrase** evaluation, not leakage-free independently
   authored ground truth: the generator sees the target transcript. Its
   contiguous-quote guard reduces obvious copying, and samples still need human
   review. Use independently written held-out queries when making a stronger
   generalization claim.

3. Review topic coverage with the user.

   Present topic counts and a small question sample. Do not ask the user to
   review every question. Explain that topics with no matching archived session
   cannot have a gold label and should be skipped. Fewer than 100 questions
   makes small model differences noisy; generate more before selecting a model.

4. Evaluate embedding models.

   ```bash
   uv run python .agents/skills/eval-embedding-reranking/python/eval_embeddings.py \
     --dataset ~/.agent-oracle/evals/dataset.json \
     --models "BAAI/bge-small-en-v1.5"
   ```

   The evaluator embeds the dataset's exact corpus passages and questions with
   every candidate. It reports MRR@10, Recall@5/10/20, document embedding
   latency per passage, query embedding latency per question, and vector
   dimension. It also shows exhaustive NumPy similarity time as a clearly
   labeled non-model diagnostic; do not treat it as production search latency.

   The report contains only the models requested in the current invocation.
   A dimension mismatch is an adoption concern outside this skill: production
   schema migration and re-embedding must be planned separately.

5. Evaluate rerankers.

   ```bash
   uv run python .agents/skills/eval-embedding-reranking/python/eval_reranker.py \
     --dataset ~/.agent-oracle/evals/dataset.json \
     --base-model "BAAI/bge-small-en-v1.5" \
     --rerankers "Xenova/ms-marco-MiniLM-L-6-v2"
   ```

   Each invocation freshly computes the base model's top-50 passage candidates
   from the portable corpus. It then measures each ONNX cross-encoder's rerank
   inference time only, along with session-level MRR@10 and Recall@5/10/20. The
   base embedding/candidate generation is not included in rerank latency.

6. Select a model.

   Choose the measured quality/latency trade-off appropriate for interactive
   search. Record the dataset fingerprint with the decision so results are
   comparable only against the same dataset artifact.

## Metrics

- **MRR@10**: reciprocal rank of the gold session in the top ten deduplicated
  session results, averaged over questions.
- **Recall@k**: fraction of questions whose gold session appears in the top-k
  deduplicated session results.
- **Document embed ms/passage**: model inference time to embed corpus passages.
- **Query embed ms/query**: model inference time to embed recall questions.
- **Rerank ms/query**: cross-encoder inference time for one query's 50 passage
  candidates.

With one gold session per question, binary nDCG and Precision do not add useful
information beyond MRR and Recall, so they are not computed.

## What not to do

- Do not claim this evaluates database, FTS, hybrid, API, or end-to-end search
  latency.
- Do not use live database state, stale reports, embedding caches, or candidate
  caches while evaluating a constructed dataset.
- Do not call the dataset leakage-free; it is LLM-generated from its gold
  transcript and requires spot checks.
- Do not send all archived sessions to the LLM. Use the bounded summary-vector
  clustering and entity coverage selection; do not re-index, re-embed, or
  enrich the archive to construct evaluation data.
- Do not commit session content; the eval directory is outside the repository.
