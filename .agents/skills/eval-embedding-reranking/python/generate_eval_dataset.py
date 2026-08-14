#!/usr/bin/env python
"""Build a portable synthetic retrieval evaluation dataset from archived sessions."""

import argparse
import hashlib
import json
import logging
import struct
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from selection import (  # deptry: ignore[DEP001]
    cluster_nearest_sessions,
    farthest_first_backfill,
)

from agent_oracle.eval_dataset import SCHEMA_VERSION, dataset_fingerprint
from agent_oracle.store import Store

logger = logging.getLogger(__name__)

EVALS_DIR = Path.home() / ".agent-oracle" / "evals"
DEFAULT_DB = Path.home() / ".agent-oracle" / "index.db"
DEFAULT_MODEL = "gpt-5.6-luna"
CONTEXT_LIMIT = 8000
MAX_QUOTED_RUN = 5
DEFAULT_CLUSTERS = 20
DEFAULT_SESSIONS_PER_CLUSTER = 5
DEFAULT_MIN_SESSIONS = 100
DEFAULT_MAX_SESSIONS = 200
HARD_MAX_SESSIONS = 200
DEFAULT_SESSIONS_PER_ENTITY = 1

PROMPT = """\
You are given a sample of a coding-agent session transcript (messages between a \
developer and an AI coding agent).

Generate 1-3 questions the developer might later ask to find this session again \
(e.g. "how did I set up hybrid search with sqlite-vec?"). Rules:
- Each question must be answerable by this session's content.
- Paraphrase: never copy phrases longer than 5 words from the transcript.
- Write like a real recall attempt, not a summary.
- Assign each question one short topic label.

Transcript:
"""


class GeneratedQuestion(BaseModel):
    """Define one synthetic retrieval question in the LLM output."""

    question: str
    topic: str


class GeneratedQuestionsOutput(BaseModel):
    """Define the structured output requested from the question-generation LLM."""

    questions: list[GeneratedQuestion]


class SessionSelection:
    """Bounded selected sessions and entity coverage diagnostics."""

    def __init__(
        self,
        session_ids: list[str],
        cluster_session_ids: list[str],
        backfill_session_ids: list[str],
        entity_session_ids: list[str],
        covered_entities: list[str],
        already_covered_entities: list[str],
        uncovered_entities: list[str],
        minimum_shortfall: int,
    ) -> None:
        self.session_ids = session_ids
        self.cluster_session_ids = cluster_session_ids
        self.backfill_session_ids = backfill_session_ids
        self.entity_session_ids = entity_session_ids
        self.covered_entities = covered_entities
        self.already_covered_entities = already_covered_entities
        self.uncovered_entities = uncovered_entities
        self.minimum_shortfall = minimum_shortfall


def stable_rank(seed: int, *parts: str) -> str:
    """Return a stable seeded tie-breaker independent of database row ordering."""
    payload = ":".join((str(seed), *parts)).encode()
    return hashlib.sha256(payload).hexdigest()


def list_session_summary_embeddings(store: Store) -> list[dict]:
    """Read existing session-summary vectors for offline sampling."""
    reader = getattr(store, "list_session_summary_embeddings", None)
    if reader is not None:
        return reader()
    rows = store.conn.execute(
        "SELECT s.id, v.embedding FROM vec_sessions v "
        "JOIN sessions s ON s.rowid = v.rowid ORDER BY s.id"
    ).fetchall()
    return [
        {
            "id": row["id"],
            "embedding": list(struct.unpack(f"<{len(row['embedding']) // 4}f", row["embedding"])),
        }
        for row in rows
    ]


def list_all_entities(store: Store) -> list[dict]:
    """Read all entity associations for bounded coverage sampling."""
    reader = getattr(store, "list_all_entities", None)
    if reader is not None:
        return reader()
    rows = store.conn.execute(
        "SELECT session_id, entity_type, entity_value FROM entities "
        "ORDER BY entity_type, entity_value, session_id"
    ).fetchall()
    return [dict(row) for row in rows]


def select_sessions(
    store: Store,
    *,
    done: set[str],
    max_sessions: int,
    clusters: int,
    sessions_per_cluster: int,
    min_sessions: int,
    sessions_per_entity: int,
    seed: int,
) -> SessionSelection:
    """Select a deduplicated bounded mix of clustered and entity-covered sessions."""
    candidates = [row for row in list_session_summary_embeddings(store) if row["id"] not in done]
    candidate_ids = {row["id"] for row in candidates}
    cluster_session_ids = cluster_nearest_sessions(
        candidates, clusters, sessions_per_cluster, seed
    )[:max_sessions]
    backfill_session_ids = farthest_first_backfill(
        candidates,
        cluster_session_ids,
        min(min_sessions, max_sessions) - len(cluster_session_ids),
        seed,
    )
    selected = cluster_session_ids + backfill_session_ids
    entity_sessions: dict[str, set[str]] = {}
    for entity in list_all_entities(store):
        session_id = entity["session_id"]
        if session_id not in candidate_ids:
            continue
        entity_key = f"{entity['entity_type']}:{entity['entity_value']}"
        entity_sessions.setdefault(entity_key, set()).add(session_id)
    covered = []
    already_covered = []
    uncovered = []
    entity_session_ids = []
    entity_keys = sorted(
        entity_sessions,
        key=lambda key: (-len(entity_sessions[key]), stable_rank(seed, key)),
    )
    for entity_key in entity_keys:
        candidates_for_entity = sorted(
            entity_sessions[entity_key],
            key=lambda session_id: stable_rank(seed, entity_key, session_id),
        )
        unselected = [
            session_id for session_id in candidates_for_entity if session_id not in selected
        ]
        if not unselected:
            already_covered.append(entity_key)
            continue
        if len(selected) >= max_sessions:
            uncovered.append(entity_key)
            continue
        additions = unselected[:sessions_per_entity]
        additions = additions[: max_sessions - len(selected)]
        if not additions:
            uncovered.append(entity_key)
            continue
        selected.extend(additions)
        entity_session_ids.extend(additions)
        covered.append(entity_key)
    selected = selected[:max_sessions]
    return SessionSelection(
        selected,
        cluster_session_ids,
        backfill_session_ids,
        entity_session_ids,
        covered,
        already_covered,
        uncovered,
        max(0, min_sessions - len(candidates)),
    )


def print_selection_preview(selection: SessionSelection, available: int) -> None:
    """Print bounded generation work before initializing the OpenAI client."""
    print(f"Selected {len(selection.session_ids)} of {available} eligible sessions.")
    print(
        "Selection: "
        f"{len(selection.cluster_session_ids)} cluster-derived, "
        f"{len(selection.backfill_session_ids)} backfill, "
        f"{len(selection.entity_session_ids)} entity-added."
    )
    if selection.minimum_shortfall:
        print(
            "Warning: only "
            f"{available} eligible embedded sessions; short of the minimum by "
            f"{selection.minimum_shortfall}."
        )
    call_count = len(selection.session_ids)
    print(f"Maximum LLM calls: {call_count} (up to {call_count * 3} questions).")
    covered = len(selection.covered_entities)
    already_covered = len(selection.already_covered_entities)
    uncovered = len(selection.uncovered_entities)
    print(
        "Entity coverage: "
        f"{covered} added, {already_covered} already covered, {uncovered} uncovered."
    )
    if selection.uncovered_entities:
        print("Uncovered entities: " + ", ".join(selection.uncovered_entities))


def reset_generation_state(out: Path, checkpoint: Path) -> None:
    """Remove only this dataset's resumable generation state."""
    for path in (out, checkpoint):
        if path.exists():
            path.unlink()


def load_existing_dataset(path: Path) -> dict:
    """Load a prior portable dataset, or return an empty dataset state."""
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "questions": [],
            "corpus": [],
            "processed_session_ids": [],
        }
    dataset = json.loads(path.read_text())
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset schema in {path}; rerun with --reset")
    if not isinstance(dataset.get("questions"), list) or not isinstance(
        dataset.get("corpus"), list
    ):
        raise ValueError(f"Invalid dataset in {path}; rerun with --reset")
    processed = dataset.get("processed_session_ids")
    if processed is not None and (
        not isinstance(processed, list)
        or not all(isinstance(session_id, str) for session_id in processed)
    ):
        raise ValueError(f"Invalid processed session ids in {path}; rerun with --reset")
    return dataset


def processed_session_ids(dataset: dict) -> set[str]:
    """Return generated session ids, preserving legacy question-only state."""
    explicit = dataset.get("processed_session_ids")
    if explicit is not None:
        return set(explicit)
    return {item["session_id"] for item in dataset["questions"] if item.get("session_id")}


def resume_session_ids(dataset: dict, checkpoint: Path) -> set[str]:
    """Return dataset construction state without trusting legacy checkpoints."""
    del checkpoint
    return processed_session_ids(dataset)


def searchable_messages(session: dict) -> list[dict]:
    """Extract the exact searchable passages for one session."""
    return [
        {"id": message["id"], "session_id": session["id"], "content": message["content"]}
        for message in session["messages"]
        if not message["is_thinking"]
        and not message["is_system_instruction"]
        and not message["is_injected"]
        and message["content"].strip()
    ]


def sample_context(messages: list[dict]) -> str:
    """Return a bounded text sample from searchable session messages."""
    return "\n".join(message["content"] for message in messages)[:CONTEXT_LIMIT]


def longest_quoted_run(question: str, context: str) -> int:
    """Return the longest contiguous word run shared by question and context."""
    previous = [0] * (len(context.lower().split()) + 1)
    longest = 0
    for q_word in question.lower().split():
        current = [0] * len(previous)
        for index, c_word in enumerate(context.lower().split()):
            if q_word == c_word:
                current[index + 1] = previous[index] + 1
                longest = max(longest, current[index + 1])
        previous = current
    return longest


def generate_session_questions(
    client: OpenAI, model: str, session: dict, context: str
) -> list[dict]:
    """Ask the LLM for synthetic paraphrase questions about one session."""
    if not context.strip():
        return []
    response = client.chat.completions.parse(
        model=model,
        response_format=GeneratedQuestionsOutput,
        messages=[{"role": "user", "content": PROMPT + context}],
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        logger.warning("No parsed response for session %s; skipped", session["id"])
        return []
    accepted = []
    for item in parsed.questions:
        question = item.question.strip()
        topic = item.topic.strip().lower() or "general"
        if question and longest_quoted_run(question, context) <= MAX_QUOTED_RUN:
            accepted.append({"question": question, "topic": topic})
    return accepted


def next_question_number(questions: list[dict]) -> int:
    """Return the next monotonic question number without reusing an id."""
    numbers = [int(q["id"].removeprefix("q-")) for q in questions if q["id"].startswith("q-")]
    return max(numbers, default=-1) + 1


def parse_args() -> argparse.Namespace:
    """Parse and validate bounded dataset-generation arguments."""
    parser = argparse.ArgumentParser(description="Generate a synthetic portable eval dataset")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=EVALS_DIR / "dataset.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=DEFAULT_MAX_SESSIONS,
        help=(
            f"Max new sessions sent to the LLM (default {DEFAULT_MAX_SESSIONS}; "
            f"hard max {HARD_MAX_SESSIONS})"
        ),
    )
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
    parser.add_argument("--sessions-per-cluster", type=int, default=DEFAULT_SESSIONS_PER_CLUSTER)
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=DEFAULT_MIN_SESSIONS,
        help="Minimum cluster-derived and backfill sessions before entity sampling",
    )
    parser.add_argument(
        "--sessions-per-entity",
        type=int,
        default=DEFAULT_SESSIONS_PER_ENTITY,
        help="Max eligible sessions added for each entity value",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic selection seed")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview selection without creating an OpenAI client"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Discard this output and checkpoint first"
    )
    args = parser.parse_args()

    if not 1 <= args.max_sessions <= HARD_MAX_SESSIONS:
        parser.error(f"--max-sessions must be between 1 and {HARD_MAX_SESSIONS}")
    if args.clusters < 1:
        parser.error("--clusters must be at least 1")
    if args.sessions_per_cluster < 1:
        parser.error("--sessions-per-cluster must be at least 1")
    if not 1 <= args.min_sessions <= args.max_sessions:
        parser.error("--min-sessions must be between 1 and --max-sessions")
    if args.sessions_per_entity < 1:
        parser.error("--sessions-per-entity must be at least 1")
    return args


def generate_selected_sessions(
    store: Store,
    selection: SessionSelection,
    client: OpenAI,
    model: str,
    questions: list[dict],
    done: set[str],
) -> list[dict]:
    """Generate questions and return sessions included in this build."""
    question_number = next_question_number(questions)
    selected_sessions = []
    for session_id in selection.session_ids:
        session = store.get_session(session_id)
        if session is None:
            continue
        selected_sessions.append(session)
        messages = searchable_messages(session)
        generated = generate_session_questions(client, model, session, sample_context(messages))
        for item in generated:
            questions.append(
                {
                    "id": f"q-{question_number}",
                    "question": item["question"],
                    "topic": item["topic"],
                    "session_id": session["id"],
                    "cwd": session.get("cwd"),
                }
            )
            question_number += 1
        done.add(session["id"])
    return selected_sessions


def main() -> None:
    """Build or incrementally extend one self-contained eval dataset."""
    args = parse_args()

    load_dotenv(find_dotenv(usecwd=True), override=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out.parent / f".{args.out.stem}.checkpoint.json"
    if args.reset:
        reset_generation_state(args.out, checkpoint_path)
    dataset = load_existing_dataset(args.out)
    done = resume_session_ids(dataset, checkpoint_path)
    questions = dataset["questions"]

    store = Store(args.db)
    available = [row for row in list_session_summary_embeddings(store) if row["id"] not in done]
    selection = select_sessions(
        store,
        done=done,
        max_sessions=args.max_sessions,
        clusters=args.clusters,
        sessions_per_cluster=args.sessions_per_cluster,
        min_sessions=args.min_sessions,
        sessions_per_entity=args.sessions_per_entity,
        seed=args.seed,
    )
    print_selection_preview(selection, len(available))
    if args.dry_run:
        return
    logger.info("%d new sessions selected (%d already done)", len(selection.session_ids), len(done))

    selected_sessions = generate_selected_sessions(
        store, selection, OpenAI(), args.model, questions, done
    )

    # The dataset carries the exact passages used by both evaluators. Store is
    # needed only for this construction step, never when running the eval.
    corpus_by_id = {passage["id"]: passage for passage in dataset["corpus"]}
    for session in selected_sessions:
        corpus_by_id.update({passage["id"]: passage for passage in searchable_messages(session)})
    dataset["corpus"] = sorted(corpus_by_id.values(), key=lambda passage: passage["id"])
    dataset["topics"] = sorted({question["topic"] for question in questions})
    dataset["generator"] = {"model": args.model, "synthetic": True}
    dataset["processed_session_ids"] = sorted(done)
    dataset["fingerprint"] = dataset_fingerprint(dataset)
    args.out.write_text(json.dumps(dataset, indent=2) + "\n")

    print(f"Wrote {len(questions)} questions and {len(dataset['corpus'])} passages to {args.out}")
    print("Synthetic paraphrase dataset: review samples; it is not independent ground truth.")
    for question in questions[:10]:
        print(f"  [{question['topic']}] {question['question']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
