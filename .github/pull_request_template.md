## What this changes

<!-- One or two sentences. What is different for a user or a reviewer of this repo? -->

## Why

<!-- The problem this solves. Link an issue if there is one. -->

## How it was checked

- [ ] `make lint` and `make test` pass
- [ ] `make eval-offline` reproduces the committed baseline, or the baseline change is explained below
- [ ] `uvx pre-commit run --all-files` is clean

## Checklist

- [ ] `platform_core/` still imports nothing from `mlr/`, `api/` or `ui/`
- [ ] No tenant-specific branch in code; tenant differences are configuration
- [ ] No path lets the system approve content; every run still ends at a person
- [ ] New or changed nodes say why they exist and whether they are an LLM decision or deterministic code
- [ ] Changed prompts have a bumped version string
- [ ] Deliberate compromises are marked with a `# Trade-off:` comment
- [ ] No secrets, no real product or company names, no real patient data; cassettes are scrubbed

## Eval impact

<!-- Blocker recall and false-escalation rate before and after, if this touches prompts, checks,
     gates, routing or tenant policy. "None" is a fine answer for anything else. -->
