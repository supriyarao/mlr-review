# CLAUDE.md — conventions for this repository

This repo was built with an AI coding agent. This file is the agent's standing brief; it is
committed deliberately, as a record of how the codebase is meant to be extended.

## What this is

A **multi-tenant agentic platform** whose first use case is MLR (Medical, Legal, Regulatory)
review of pharmaceutical promotional content. All data is synthetic and all products and
companies are fictional.

The single most important structural idea: **MLR is a use case, not the system.** Read the
layer boundary below before adding any file.

## The layer boundary (non-negotiable)

```
platform_core/   reusable agent platform — tenancy, gates, HITL, audit, retrieval, runtime, evals
mlr/             the MLR use case — state, schemas, prompts, nodes, graph, MLR-specific metrics
api/, ui/        delivery surfaces
```

- `platform_core` **must not import from `mlr`**, `api`, or `ui`. `tests/test_import_boundary.py`
  parses the AST of every file under `platform_core/` and fails the build if it does.
- If something feels like it belongs in `platform_core` but mentions a claim, a finding, a
  severity, or a reviewer tier by name, it belongs in `mlr/` instead — or the platform piece
  needs a more general shape (see "policy as data").
- A second use case should be addable without editing a single file in `platform_core`.

## Policy as data, not code

Tenant-specific rules live in `corpus/tenants/*.yaml` and are validated by
`platform_core/tenancy.py`. Severity mappings, escalation tiers, confidence thresholds and
MLR-specific policy switches (head-to-head evidence requirements, ISI placement rules) are all
configuration. Adding a tenant must never mean writing an `if tenant_id == ...` branch.

Every tenant config has a `version`, and the sha256 `config_hash()` of the canonical config is
written into every audit event — so an auditor can tell which policy produced a decision.

## Humans decide

The system never approves content. Every path ends at a human: clean pieces go to a standard
reviewer, blockers and anything ungrounded or low-confidence escalate to the tenant's higher
tier. This is a product requirement, not a limitation to be optimised away.

## Code conventions

- **Python 3.12**, managed with `uv`. `uv sync` to install, `uv run` to execute.
- **Every node gets a docstring** that says (a) why the node exists, and (b) whether it is an
  **LLM decision** or **deterministic code**. Reviewers of this repo need to tell those apart
  at a glance.
- **Mark trade-offs inline** with a `# Trade-off:` comment explaining what was given up and why.
  These are deliberate discussion points, not apologies.
- Prefer simple, readable code over clever abstractions. This codebase is read by people
  evaluating judgement, not just correctness.
- Pydantic v2 for every schema that crosses a boundary (LLM output, API payload, config file).
- Prompts live in `mlr/prompts/` as versioned files, one per node. **Bump the version string
  when you change a prompt** — the version is recorded in the audit log and in eval results,
  so a prompt change must be attributable.

## Models

One factory, `platform_core/llm.py`, with three interchangeable backends selected by
`MLR_LLM_BACKEND`:

| Backend | When | Notes |
|---|---|---|
| `fake` | tests, CI, `make eval-offline` | Scripted/recorded responses. Forced on in `tests/conftest.py`. No network. |
| `ollama` | local development | Free. Default `qwen2.5:7b`. Needs `ollama serve`. |
| `anthropic` | live demo, committed eval baseline | `claude-sonnet-5` (strong) / `claude-haiku-4-5` (fast). |

The graph only ever asks for a **tier** (`strong` or `fast`), never a model name — cost tiering
is a configuration decision. Model ids come from `MLR_MODEL_STRONG` / `MLR_MODEL_FAST`.

Note for the `anthropic` backend: these models reject `temperature`, `top_p`, `top_k` and
`thinking.budget_tokens` with a 400. Do not set them. Use `thinking={"type": "adaptive"}` where
reasoning depth earns its cost, and omit it on cheap classification nodes.

## Testing

- `make test` must pass **offline, with no API key and without Ollama running**. If a new test
  needs a model, it uses `FakeLLM`.
- `make eval-offline` must reproduce `evals/baselines/latest.json` exactly, by replaying
  cassettes from `evals/cassettes/`.
- Both `make test` and `make lint` must pass before any commit.

## Safe to publish

This repo is public. Treat everything committed as published.

- Never commit `.env`, `data/`, `evals/results/`, or `PROMPT.md`.
- No API keys or tokens anywhere, including inside recorded cassettes — scrub key-like strings
  and request headers before saving a recording.
- No real drug names, real companies, or real patient data. Fictional only.
- No employer, client, or interview-brief text.
- Do commit `evals/baselines/` and `evals/cassettes/` so a stranger can reproduce the offline
  eval with no API key.

## Commits

Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`), scoped where it helps
(`feat(platform):`, `feat(mlr):`). Small and reviewable — the history is part of what readers of
this repo evaluate.
