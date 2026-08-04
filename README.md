# Agent Oracle

Local web app that archives coding agent sessions (Codex, Factory Droid, Claude Code)
into `~/.agent-oracle/` and makes them searchable with text, vector, and hybrid search.
Sessions are enriched by an LLM with entities and summaries. The backend also serves
coding agents themselves via REST + MCP, so agents can search past sessions.

## Stack

- Backend: Python 3.11+, FastAPI, SQLite (FTS5 + sqlite-vec), FastEmbed embeddings,
  OpenAI API for enrichment, FastMCP. Source in `src/agent_oracle/`.
- Frontend: SvelteKit + TypeScript in `frontend/`.

## Setup

```bash
uv sync                       # backend
cd frontend && npm install    # frontend
cp .env.example .env          # fill in OPENAI_API_KEY
```

## Run

```bash
uv run uvicorn agent_oracle.api:app --reload   # backend, http://localhost:8000
cd frontend && npm run dev                     # frontend, http://localhost:5173
```

## Development

- Backend: `uv run ruff check --fix`, `uv run ruff format`, `uv run ty check`, `uv run pytest`
- Frontend: `npm run check`, `npm run lint`, `npm run build`
- Everything: `uvx pre-commit run --all-files` (Python + TypeScript hooks; the
  TypeScript hooks activate as soon as files under `frontend/` change)

Rules for agents and contributors live in `AGENTS.md` and `CONVENTIONS.md`.
