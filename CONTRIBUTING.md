# Contributing

This is a demonstration project: a multi-tenant agent platform whose first use case is MLR review
of pharmaceutical promotional content. Issues and pull requests are welcome. All data is synthetic
and every product and company is fictional; please keep it that way.

## Getting set up

```bash
make setup     # create the venv, install dependencies, create .env from .env.example
make seed      # index the synthetic corpus and load each tenant's rules
make test      # the whole suite: offline, no API key, no model server
make demo      # API on :8000, UI on :8501
```

Nothing above needs an API key. The tests force a fake model backend and `make eval-offline`
replays recorded model responses.

## Before you open a pull request

```bash
make lint
make test
make eval-offline              # must reproduce evals/baselines/latest.json
uvx pre-commit run --all-files # secret scan, large-file check, formatting
```

Install the hooks once with `uvx pre-commit install` and they run on every commit.

## What reviewers will look for

- **The layer boundary.** `platform_core/` is the reusable platform and must not import from
  `mlr/`, `api/` or `ui/`. `ui/` reaches the system over HTTP only. `tests/test_import_boundary.py`
  enforces both. If a platform change needs to mention a claim, a finding or a reviewer tier by
  name, it belongs in `mlr/`.
- **Policy is data.** Tenant differences live in `corpus/tenants/`. A change that adds
  `if tenant_id == ...` will be asked to become configuration.
- **Humans decide.** No change may introduce a path where the system approves content. Every run
  ends at a reviewer.
- **Say what calls a model.** Every graph node has a docstring that states why it exists and
  whether it is an LLM decision or deterministic code.
- **Prompt changes are versioned.** Bump the version string in `mlr/prompts/` when you change a
  prompt. It is recorded in the audit log and in eval results.
- **Trade-offs are written down.** Mark a deliberate compromise with a `# Trade-off:` comment that
  says what was given up and why.
- **Contracts are generated.** The example responses in `tests/fixtures/api/` come from
  `make fixtures`. Do not edit them by hand.

## Commits

[Conventional commits](https://www.conventionalcommits.org/), scoped where it helps:
`feat(platform):`, `feat(mlr):`, `fix(api):`, `test:`, `docs:`, `ci:`, `chore:`. Keep them small.

## Never commit

- `.env`, `data/`, `evals/results/`, or anything containing an API key or token. Recorded eval
  cassettes are scrubbed of headers and key-like strings before they are saved.
- Real drug names, real companies, or real patient data.

See [SECURITY.md](SECURITY.md) for how to report a vulnerability or a leaked secret.
