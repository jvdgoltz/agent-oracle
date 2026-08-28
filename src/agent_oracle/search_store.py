"""Helpers for SQLite full-text search queries."""


def sanitize_fts_query(query: str) -> str:
    """Escape a user query into a safe FTS5 MATCH expression."""
    tokens = query.strip().split()
    if not tokens:
        return ""
    return " ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
