# AGENTS.md

Gymnasium benchmark for causal reinforcement learning in Magic: The Gathering, with an explicit SCM over strategic game variables.

## Package manager

`uv`. Run everything through it — `uv run pytest`, `uv run ruff check .`, `uv sync`, `uv add <pkg>`. Reach for `pip` or a manually activated `.venv` only when a task cannot be expressed as a `uv` command.

Dependencies live in two places that must stay in sync: `[project.optional-dependencies].dev` (for `pip install -e ".[dev]"`) and `[dependency-groups].dev` (what `uv sync` installs). Edit both.

## Testing

`uv run pytest` skips tests marked `slow` — that exclusion is baked into `addopts`. Before calling a change done, run the full suite: `uv run pytest -m ""`.

## Conventions

Beyond what `[tool.ruff]` enforces:

- `import typing as tp`, and reference typing constructs as `tp.*` — never `from typing import ...`.
- No nested functions. Lift a helper to module scope instead.
- Branch names follow Gitflow: `feature/`, `bugfix/`, `hotfix/`, `release/`, `support/`, `task/`, or `chore/`, lowercase segments.

`pre-commit` enforces these on commit, but `validate_conventions.py` runs warn-only — it prints violations and exits 0, so a clean commit is not a clean bill of health. Read its output and fix what it reports.

## Where things are

`README.md` documents the four CLI workflows (`mtg-train`, `mtg-eval`, `mtg-gameplay`, `mtg-research`), the action and observation spaces, the causal model, and how to add a card, archetype, agent, or causal variable. Read it before extending the environment or reproducing benchmark results.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `jackswisher/mtg-agent`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
