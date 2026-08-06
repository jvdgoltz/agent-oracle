# Coding Agent Instructions

## Project: Agent Oracle
Local web app that archives coding agent sessions (Codex, Factory Droid, Claude Code,
Oh My Pi) into `~/.agent-oracle/` and makes them searchable (text / vector / hybrid)
with LLM-enriched entities and summaries. The backend also serves coding agents via REST + MCP.

- Backend: Python 3.11+, FastAPI, SQLite (FTS5 + sqlite-vec), FastEmbed, OpenAI API, FastMCP.
  Source in `src/agent_oracle/`, tests in `tests/` (one `test_*.py` per source module).
- Frontend: SvelteKit + TypeScript in `frontend/` (scaffolded later).

### Commands
- Install: `uv sync` (backend), `npm install` (frontend, once scaffolded)
- Lint/format: `uv run ruff check --fix` and `uv run ruff format`
- Typecheck: `uv run ty check`
- Tests: `uv run pytest`
- Full quality gate: `uvx pre-commit run --all-files`

### Backend verification
Whenever making changes to the backend, test the affected endpoints by calling
them (e.g. via `curl http://localhost:8731/api/...`) and the MCP endpoints.
Do not rely on unit tests alone to verify backend behavior.

### Running services
Both the backend and frontend are managed by `launchd` and may already be
running in the background. Before starting either service manually, check
whether it is already running:

```bash
launchctl list | grep com.agent-oracle
```

If a service is already loaded, its `--reload` flag will pick up code changes
automatically; there is no need to start a second instance. The `install.sh`
script in the repository root installs, loads, and starts both services.

## Coding Style
- Simple functions, proper abstraction. Keep functions short. Code should be self-explanatory. Repository folder structure should be self-explanatory.
- Every line of code should have intent, and the intent should come from the user instructions.
- Maintain clear mapping between source filenames and corresponding `test_*` filenames.
- Each module, function, and class should have a concise docstring.
- No need for backward compatibility or forward compatibility.

## Security
- For high-risk operations, make sure to always stop and confirm with the user before execution. Examples:
  - Irreversible removals like `rm -rf`
  - Changing infrastructure like `terraform apply`
  - Database migrations or schema changes
  - Modifying access control, permissions, or authentication logic
  - Running scripts that affect multiple systems or environments
  - Publishing packages or deploying to production
  - Using custom ad-hoc scripts to bypass any tool or environment restrictions
- When installing or updating dependencies, verify before installation that the selected version has no known
  supply-chain compromise reports or critical CVEs.
- Always pin dependencies to exact versions in manifests or lockfiles instead of using floating version ranges.
- Do not install anything known or suspected to be a security risk.
- Never hardcode secrets, API keys, passwords, or tokens in code. Use environment variables or secure vault services.
- Do not log or expose sensitive information like credentials, PII, or authentication tokens.
- Do not commit secrets or credentials to version control. If accidentally committed, treat as compromised and rotate
  immediately.

## Communication
- Be professional and concise but do communicate in full sentences. Avoid acknowledgements ("Got it!", "You are right!"), banter, and small talk.
- Always present and explain your plan to the user before implementing anything. Explain trade-offs and why you recommend certain solutions.
- Ask clarifying questions when the context is unclear or ambiguous.

## Approach
- Do not rely on your internal knowledge about APIs, libraries, and tools. Assume that it might be outdated. Use web search and web fetch to retrieve relevant documentation.
- Implement the basic happy path of any functionality or feature first. Only after confirmation from the user, implement the edge cases.
- Only implement features and functionality that the user asked for.
- Use red/green TDD.
- If stuck after 2 failed implementation attempts, stop and present a blocker summary with concrete next options.
- Before handoff, run the smallest relevant tests/lint for touched code and report pass/fail explicitly.
- Self-improvement: If the user corrects a mistake or gives feedback, suggest a change to the `AGENTS.md` to prevent the same mistake from happening again.

## Error Handling & Debugging
- No need for excessive try-except blocks and edge case handling. Don't wrap imports in try-except blocks.
- Use `logging` or equivalent instead of `print`. Make use of `debug`, `warning` and `error` levels in addition to `info`.
- Prefer explicit failures over silent bugs. Avoid generic except Exception blocks.
- If you for some reason implement one, add `logging.error(..., exc_info=True)` or equivalent to surface the stack trace.

## Documentation
- Code is the documentation. Do not create plan files, decision logs, or any other markdown documentation for code.
- Agents are expected to answer questions by reading the code. Make sure they can: self-explanatory names and concise, descriptive docstrings on every module, class, and function.
- The only markdown files in this repo are `README.md`, `AGENTS.md`, `CLAUDE.md`, and `CONVENTIONS.md`.

## Hard constraints

@CONVENTIONS.md
