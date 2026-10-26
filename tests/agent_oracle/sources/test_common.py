"""Tests for bounded source file reading."""

from pathlib import Path
from unittest.mock import patch

from agent_oracle.sources import claude, codex, factory, omp


def test_parsers_stream_source_files(tmp_path: Path) -> None:
    """Session parsers never materialize an entire JSONL file as text."""
    path = tmp_path / "session.jsonl"
    path.write_text("{}\n" * 10)
    with patch.object(Path, "read_text", side_effect=AssertionError("unbounded read")):
        for parser in (
            codex.parse_codex_session,
            claude.parse_claude_session,
            factory.parse_factory_session,
            omp.parse_omp_session,
        ):
            parser(path)
