# Security

## What this project is

A demonstration. It runs on synthetic data about fictional products and is **not for review of
real promotional material**. It is not medical, legal or regulatory advice.

It is built to be run locally. It is not hardened for exposure to an untrusted network, and the
list under "Known limitations" says where.

## Reporting a vulnerability

Please report privately, not in a public issue:

1. Open the repository's **Security** tab and choose **Report a vulnerability** (GitHub private
   vulnerability reporting).
2. Describe what you found, how to reproduce it, and what it allows.

You should get a reply within a week. Because this is a demo with no production deployment, fixes
are made on `main` and there are no supported back-versions.

## If you find a secret in the repository

Treat it as a vulnerability and report it the same way. Do not open a public issue quoting it.
A leaked key is revoked first and removed second: deleting it from the latest commit does not
remove it from history.

The repository is scanned with [gitleaks](https://github.com/gitleaks/gitleaks) in a pre-commit
hook and across the full history in CI.

## Known limitations

These are deliberate simplifications for a demo, listed so nobody mistakes them for a design:

- **No authentication or authorisation.** The API accepts any caller. The reviewer tier is a
  request field standing in for a login; a real deployment derives it from the authenticated user.
  The API does refuse a decision from a tier other than the one a run was routed to, but it cannot
  verify who is asking.
- **Tenant isolation is enforced in the data layer, not at the HTTP edge.** Retrieval and the rule
  registry are bound to a tenant, so one tenant's evidence and rules cannot be read through
  another's. The API does not authenticate which tenant a caller belongs to.
- **The audit log is append-only by convention.** It checks that a run's sequence numbers are
  complete and unique, which catches a dropped or duplicated event. It is not a cryptographic
  hash chain and does not resist someone with write access to the database.
- **Tenant-authored rule text reaches a model.** It is neutralised before it is placed in a
  prompt and flagged to the author when it looks like an instruction, which reduces prompt
  injection and does not eliminate it.
- **Uploads are parsed, not sandboxed.** Files are limited to 5 MB and to PDF, Markdown and text.
  Exported spreadsheets store submitted text as text, never as formulas.
- **Local storage is unencrypted.** SQLite databases and the vector store live under `data/`.

## Handling data

Do not put real patient data, real product material or confidential documents into this system.
With the `anthropic` backend, submitted text is sent to the model provider; with `ollama` it stays
on your machine; with `fake` nothing leaves the process.
