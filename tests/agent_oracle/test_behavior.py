"""Tests for OMP-compatible user behavior statistics."""

from __future__ import annotations

import pytest

from agent_oracle.behavior import compute_user_message_metrics, summarize_messages


def test_metrics_match_omp_short_prose_rules() -> None:
    """Count OMP's distinct behavior signals on short prose."""
    metrics = compute_user_message_metrics(
        "NO THIS IS WRONG!!! You didn't fix it. I already said this still fails, dude :("
    )

    assert metrics.yelling == 1
    assert metrics.profanity == 0
    assert metrics.anguish == 3
    assert metrics.negation == 1
    assert metrics.repetition == 2
    assert metrics.blame == 1


def test_metrics_exclude_formatted_or_structured_prompts() -> None:
    """Retain volume counts while suppressing OMP formatted prompt signals."""
    metrics = compute_user_message_metrics("NO!!!\nI already said this\nYou didn't fix it")

    assert metrics.chars > 0
    assert metrics.words == 9
    assert metrics.yelling == metrics.profanity == metrics.anguish == 0
    assert metrics.negation == metrics.repetition == metrics.blame == 0


def test_metrics_match_omp_classifier_regressions() -> None:
    """Keep OMP's profanity, anguish, and correction boundaries intact."""
    assert compute_user_message_metrics("oh FUCK this is bullshit, damn it").profanity == 3
    assert compute_user_message_metrics("why!!! seriously??? omg!?!?!?").anguish == 3
    assert compute_user_message_metrics("no extensions to the landing page").negation == 0
    metrics = compute_user_message_metrics(
        "no, you broke it AGAIN. i told you it still doesnt work"
    )
    assert (metrics.negation, metrics.repetition, metrics.blame) == (1, 2, 1)
    assert compute_user_message_metrics("match foo:(bar) in code").anguish == 0
    assert compute_user_message_metrics("😀").chars == 2


def test_summary_excludes_injected_messages_and_groups_reliably() -> None:
    """Aggregate only real user messages by date, agent, and project."""
    report = summarize_messages(
        [
            {
                "content": "Wrong, you missed it",
                "timestamp": "2026-08-01T12:00:00+00:00",
                "agent": "codex",
                "cwd": "/work/a",
                "is_injected": 0,
            },
            {
                "content": "NO!!!",
                "timestamp": "2026-08-01T13:00:00+00:00",
                "agent": "codex",
                "cwd": "/work/a",
                "is_injected": 1,
            },
        ]
    )

    assert report["totals"]["user_messages"] == 1
    assert report["totals"]["negation"] == 1
    assert report["totals"]["blame"] == 1
    assert "total_frustration" not in report["totals"]
    assert report["daily"][0]["date"] == "2026-08-01"
    assert report["agents"][0]["agent"] == "codex"
    assert report["projects"][0]["cwd"] == "/work/a"


@pytest.mark.parametrize(
    ("text", "field", "expected"),
    [
        ("", "chars", 0),
        ("   \n\t ", "words", 0),
        ("STOP DOING THAT NOW", "yelling", 1),
        ("Hi there, please STOP doing THAT immediately, it is really annoying.", "yelling", 0),
        ("OK", "yelling", 0),
        ("WHY IS THIS BROKEN? FIX IT NOW!! please.", "yelling", 2),
        ("call getHTMLParser then exit", "yelling", 0),
        ("Follow AGENTS.md and SYSTEM.md carefully.", "yelling", 0),
        ("HELLO", "yelling", 0),
        ("WHAT THE HELL", "yelling", 1),
        ("CMOOON", "yelling", 1),
        ("import classes from module", "profanity", 0),
        ("git status shows the rebase failed", "profanity", 0),
        ("run git commit then git push", "profanity", 0),
        ("dummy data, blast radius, and config knob", "profanity", 0),
        ("this is garbage, useless and horrible work", "profanity", 0),
        ("nooo whyyy", "anguish", 2),
        ("ok!! sure??", "anguish", 0),
        ("waiting.. still loading...", "anguish", 0),
        ("ugh", "anguish", 1),
        ("ughh", "anguish", 1),
        ("argh", "anguish", 1),
        ("grr", "anguish", 1),
        ("what!!!111", "anguish", 1),
        ("are you serious!?!?!??111", "anguish", 1),
        ("port 8111 please", "anguish", 0),
        ("no this is the renderer", "negation", 1),
        ("nope, still wrong", "negation", 1),
        ("nah look at this", "negation", 1),
        ("wrong file", "negation", 1),
        ("nvm got it", "negation", 1),
        ("no extensions to the landing page", "negation", 0),
        ("no-op change for the flag", "negation", 0),
        ("no - not that one", "negation", 1),
        ("no—wait", "negation", 1),
        ("no i meant the other one", "negation", 1),
        ("no, wait", "negation", 1),
        ("now everything works", "negation", 0),
        ("nobody knows why", "negation", 0),
        ("normal operation resumed", "negation", 0),
        ("i instantly get Finalizing ->\nNo speech detected", "negation", 0),
        ("Authentication failed\n\nWrong user name or password", "negation", 0),
        ("[Image #1] nope still broken", "negation", 1),
        ("thats not what i wanted", "negation", 1),
        ("that's not right", "negation", 1),
        ("this is not what i meant at all", "negation", 1),
        ("ok but this makes no sense", "negation", 1),
        ("ok but this makes zero sense", "negation", 1),
        ("the docs explain it well and it makes sense", "negation", 0),
        ("i meant the other file", "repetition", 1),
        ("i told you to skip it", "repetition", 1),
        ("i asked the committee to review", "repetition", 0),
        ("so i asked a bunch of experts", "repetition", 0),
        ("i asked you for json not yaml", "repetition", 1),
        ("you're not doing AST rewriting like i asked", "repetition", 1),
        ("the agent still works fine", "repetition", 0),
        ("it still doesnt work", "repetition", 1),
        ("still the same issue", "repetition", 1),
        ("still failing on darwin", "repetition", 1),
        ("you broke the layout", "blame", 1),
        ("you didnt update AGENTS", "blame", 1),
        ("you missed a callsite", "blame", 1),
        ("you forgot to commit", "blame", 1),
        ("you keep doing that", "blame", 1),
        ("can you fix the bug?", "blame", 0),
        ("why would u delete the config?", "blame", 1),
        ("why did you list the request as is?", "blame", 1),
        ("why is this slow", "blame", 0),
        ("stop touching git", "blame", 1),
        ("ok. stop reverting things", "blame", 1),
        ("the loop keeps stopping", "blame", 0),
        ("please stop making yolo changes", "blame", 0),
        ("<xml>NO THIS IS WRONG</xml>", "yelling", 0),
    ],
)
def test_metrics_match_omp_regression_matrix(text: str, field: str, expected: int) -> None:
    """Match every relevant OMP user-metrics regression boundary."""
    assert getattr(compute_user_message_metrics(text), field) == expected


def test_metrics_suppress_long_prose_after_stripping() -> None:
    """Suppress all signals when OMP sees three or more prose lines."""
    metrics = compute_user_message_metrics(
        "no this is wrong, you broke it, i meant the other one.\n"
        "please undo and try again.\nacceptance: green tests.\nthanks!"
    )
    assert metrics.chars > 0 and metrics.words > 0
    assert all(
        getattr(metrics, field) == 0
        for field in ("yelling", "profanity", "anguish", "negation", "repetition", "blame")
    )


def test_metrics_match_omp_sad_emoticon_boundaries() -> None:
    """Count OMP sad emoticons only at prose boundaries."""
    assert compute_user_message_metrics("nope still same :(").anguish == 1
    assert compute_user_message_metrics(":((").anguish == 1
    assert compute_user_message_metrics("match foo:(bar) in code").anguish == 0


def test_metrics_strip_ascii_xml_tags_before_scoring() -> None:
    """Use OMP's ASCII XML-tag parsing contract before scoring prose."""
    metrics = compute_user_message_metrics("<note lang='en'>NO THIS IS WRONG</note>")
    assert metrics.yelling == 0


def test_summary_uses_sqlite_utc_dates_for_offset_timestamps() -> None:
    """Group offset timestamps on the same UTC day used by the store filter."""
    report = summarize_messages(
        [
            {
                "content": "hello",
                "timestamp": "2026-08-02T00:30:00+02:00",
                "agent": "codex",
                "cwd": "/work/a",
                "is_injected": 0,
            }
        ]
    )

    assert report["daily"][0]["date"] == "2026-08-01"
