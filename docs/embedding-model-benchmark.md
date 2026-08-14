# Embedding model benchmark

Date: 2026-08-14

## Decision

Keep `BAAI/bge-small-en-v1.5`.

Do not repeat this benchmark with the same dataset and models. No candidate gave
better retrieval quality at the same or lower query latency.

## Scope

The benchmark measured embedding inference and in-memory ranking. It did not
measure the database, FTS, hybrid search, the API, or end-to-end search.

- Dataset fingerprint: `58119a5cd1c554c2c7c6237e06a12ef5ad827588294f256b348fdce79c9a22d3`
- Questions: 537
- Corpus passages: 18,322
- ONNX threads: 2

| Model | MRR@10 | Recall@10 | Document ms/passage | Query ms/query | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `BAAI/bge-small-en-v1.5` | 0.2108 | 0.3594 | 166.85 | 11.05 | Keep |
| `sentence-transformers/all-MiniLM-L6-v2` | 0.1883 | 0.3166 | 18.66 | 19.74 | Reject |
| `snowflake/snowflake-arctic-embed-xs` | 0.1176 | 0.1974 | 76.79 | 5.29 | Reject |
| `snowflake/snowflake-arctic-embed-s` | 0.1294 | 0.2048 | 155.44 | 11.55 | Reject |
| `jinaai/jina-embeddings-v2-small-en` | — | — | — | — | Invalid: exit 137 |

The Arctic runs used the required `query: ` prefix. The Jina run used too much
memory and the operating system stopped it. It did not produce a valid result.

## Repeat conditions

Run a new benchmark only if at least one condition is true:

- The dataset fingerprint changes.
- A new candidate model is available.
- The runtime, hardware, thread count, or model configuration changes.
- The retrieval-quality requirement changes.
