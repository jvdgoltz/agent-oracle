# Conventions

Hard, non-negotiable conventions for this repository. They apply to humans and agents alike. If a task conflicts with one of them, stop and ask.

## Philosophy

- Code is the documentation. Agents must not create prose documentation, plan files, or decision logs; markdown in this repo is limited to the files the user chooses to keep.
- Code answers questions by being read: self-explanatory names, small single-purpose functions, proper abstraction, and a concise, descriptive docstring on every module, class, and function.
- There are no backward- or forward-compatibility obligations.

## Quality

- Red/green TDD: the failing test is written before the implementation, and test files mirror the source structure.
- Every module, class, and function carries a concise, descriptive docstring. Code must answer questions about itself by being read.
- All configured quality gates (format, lint, typecheck, tests, and pre-commit hooks for every language in the repo) pass before work is considered done. Never weaken or bypass a gate to make a check pass.

## Dependencies

- Dependencies are pinned to exact versions in lockfiles; lockfiles are never edited by hand.
- New or upgraded dependencies are vetted for known supply-chain compromises and critical CVEs before being added.

## Security

- Secrets are never committed, hardcoded, or logged; credentials come from the environment.
- This project reads only local agent session data and writes only to its own local data directory; data leaves the machine only through deliberate LLM API calls.
