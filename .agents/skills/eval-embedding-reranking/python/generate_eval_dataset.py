#!/usr/bin/env python
"""Build a portable synthetic retrieval evaluation dataset from archived sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from agent_oracle.store import Store

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
EVALS_DIR = Path.home() / ".agent-oracle" / "evals"
DEFAULT_DB = Path.home() / ".agent-oracle" / "index.db"
DEFAULT_MODEL = "gpt-5.6-luna"
CONTEXT_LIMIT = 8000
MAX_QUOTED_RUN = 5

PROMPT = """\
You are given a sample of a coding-agent session transcript (messages between a \
developer and an AI coding agent).

Generate 1-3 questions the developer might later ask to find this session again \
(e.g. "how did I set up hybrid search with sqlite-vec?"). Rules:
- Each question must be answerable by this session's content.
- Paraphrase: never copy phrases longer than 5 words from the transcript.
- Write like a real recall attempt, not a summary.
- Assign each question one short topic label.

Return ONLY JSON like: {{"questions": [{{"question": "...", "topic": "..."}}]}}

Transcript:
"""


def dataset_fingerprint(dataset: dict) -> str:
    """Return a stable fingerprint of the portable corpus and question set."""
    payload = {
        "schema_version": dataset["schema_version"],
        "questions": dataset["questions"],
        "corpus": dataset["corpus"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_checkpoint(path: Path) -> set[str]:
    """Return session ids already processed, from the checkpoint file."""
    if path.is_file():
        return set(json.loads(path.read_text())["done"])
    return set()


def save_checkpoint(path: Path, done: set[str]) -> None:
    """Persist processed session ids while a dataset build is in progress."""
    path.write_text(json.dumps({"done": sorted(done)}))


def reset_generation_state(out: Path, checkpoint: Path) -> None:
    """Remove only this dataset's resumable generation state."""
    for path in (out, checkpoint):
        if path.exists():
            path.unlink()


def load_existing_dataset(path: Path) -> dict:
    """Load a prior portable dataset, or return an empty dataset state."""
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "questions": [], "corpus": []}
    dataset = json.loads(path.read_text())
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset schema in {path}; rerun with --reset")
    if not isinstance(dataset.get("questions"), list) or not isinstance(
        dataset.get("corpus"), list
    ):
        raise ValueError(f"Invalid dataset in {path}; rerun with --reset")
    return dataset


def processed_session_ids(dataset: dict) -> set[str]:
    """Return session ids already represented by the portable dataset."""
    return {
        item["session_id"]
        for key in ("questions", "corpus")
        for item in dataset[key]
        if item.get("session_id")
    }


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
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT + context}],
    )
    try:
        candidates = json.loads(response.choices[0].message.content or "{}").get("questions", [])
    except json.JSONDecodeError:
        logger.warning("Unparseable response for session %s; skipped", session["id"])
        return []
    accepted = []
    for item in candidates:
        question = str(item.get("question", "")).strip()
        topic = str(item.get("topic", "general")).strip().lower() or "general"
        if question and longest_quoted_run(question, context) <= MAX_QUOTED_RUN:
            accepted.append({"question": question, "topic": topic})
    return accepted


def next_question_number(questions: list[dict]) -> int:
    """Return the next monotonic question number without reusing an id."""
    numbers = [int(q["id"].removeprefix("q-")) for q in questions if q["id"].startswith("q-")]
    return max(numbers, default=-1) + 1


def main() -> None:
    """Build or incrementally extend one self-contained eval dataset."""
    parser = argparse.ArgumentParser(description="Generate a synthetic portable eval dataset")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=EVALS_DIR / "dataset.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--limit", type=int, default=0, help="Max new sessions to process (0 = all)"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Discard this output and checkpoint first"
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True), override=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out.parent / f".{args.out.stem}.checkpoint.json"
    if args.reset:
        reset_generation_state(args.out, checkpoint_path)
    dataset = load_existing_dataset(args.out)
    done = load_checkpoint(checkpoint_path) | processed_session_ids(dataset)
    questions = dataset["questions"]
    question_number = next_question_number(questions)

    store = Store(args.db)
    all_summaries = store.list_sessions(limit=100_000)
    pending = [summary for summary in all_summaries if summary["id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    logger.info("%d new sessions to process (%d already done)", len(pending), len(done))

    client = OpenAI()
    for index, summary in enumerate(pending):
        session = store.get_session(summary["id"])
        if session is None:
            continue
        messages = searchable_messages(session)
        generated = generate_session_questions(
            client, args.model, session, sample_context(messages)
        )
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
        if (index + 1) % 25 == 0 or index + 1 == len(pending):
            save_checkpoint(checkpoint_path, done)

    # The dataset carries the exact passages used by both evaluators. Store is
    # needed only for this construction step, never when running the eval.
    corpus_by_id = {passage["id"]: passage for passage in dataset["corpus"]}
    for summary in all_summaries:
        session = store.get_session(summary["id"])
        if session is not None:
            corpus_by_id.update(
                {passage["id"]: passage for passage in searchable_messages(session)}
            )
    dataset["corpus"] = sorted(corpus_by_id.values(), key=lambda passage: passage["id"])
    dataset["topics"] = sorted({question["topic"] for question in questions})
    dataset["generator"] = {"model": args.model, "synthetic": True}
    dataset["fingerprint"] = dataset_fingerprint(dataset)
    args.out.write_text(json.dumps(dataset, indent=2) + "\n")
    save_checkpoint(checkpoint_path, done)

    print(f"Wrote {len(questions)} questions and {len(dataset['corpus'])} passages to {args.out}")
    print("Synthetic paraphrase dataset: review samples; it is not independent ground truth.")
    for question in questions[:10]:
        print(f"  [{question['topic']}] {question['question']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
