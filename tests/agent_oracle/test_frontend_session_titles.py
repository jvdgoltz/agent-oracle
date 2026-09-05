"""Tests for frontend session-title rendering contracts."""

import re
from pathlib import Path

FRONTEND = Path(__file__).parents[2] / "frontend" / "src"


def test_session_api_types_include_nullable_titles() -> None:
    """List, detail, and search result types expose nullable session titles."""
    api = (FRONTEND / "lib" / "api.ts").read_text()

    assert api.count("title: string | null;") >= 3


def test_overview_and_detail_render_session_titles() -> None:
    """Both session pages render titles using dedicated title styles."""
    overview = (FRONTEND / "routes" / "+page.svelte").read_text()
    detail = (FRONTEND / "routes" / "sessions" / "[id]" / "+page.svelte").read_text()

    assert '<h3 class="card-title">{session.title}</h3>' in overview
    assert re.search(r'<h1\b[^>]*\bclass="session-title"[^>]*>\s*\{session.title\}\s*</h1>', detail)
