# Running the demo

Written down because the two things that stop this running are both invisible: the configuration
file is gitignored, and the runtime data is rebuilt rather than stored.

## Known-good snapshots

| tag | date | what was verified |
|---|---|---|
| `demo-2026-10-07` | 7 Oct 2026 | Both submission paths against a real model, all seven UI pages, every run endpoint, 16/16 offline eval |

`git tag -n99 demo-2026-10-07` prints the full record of what was tested at that commit.

## Getting back to a tagged version

```bash
git fetch --all --tags
git switch --detach demo-2026-10-07   # or: git switch -c from-demo demo-2026-10-07
uv sync                               # uv.lock is committed, so this is exact
make seed                             # rebuilds data/, ~30s, no API key needed
make demo                             # API on :8000, UI on :8501
```

Detached HEAD is fine for looking and running. Branch from it if you intend to change anything.

To come back to current work: `git switch main`.

## The two things git does not have

**`.env`** is gitignored, deliberately — it holds the API key. Without it the app falls back to the
fake backend, which can only replay reviews of sample pieces it already has recordings for. An
edited or new piece then returns **zero findings**, which reads as a clean bill of health rather
than as "I could not review this". Keep a copy of `.env` somewhere outside the repo.

`make setup` creates one from `.env.example` if it is missing. Fill in at least:

```
MLR_LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
# only if your key is an organisation key rather than workspace-scoped
ANTHROPIC_WORKSPACE_ID=wrkspc_...
```

**`data/`** is gitignored and regenerable: `make seed` rebuilds the per-tenant vector store and
reloads the tenant rules. It is deterministic, needs no key, and takes about thirty seconds. Past
runs live in `data/*.db` and are not worth preserving -- a run recorded before a schema change
cannot always be deserialised by later code, and the honest fix is to delete the databases and
submit the piece again.

## Which backend you are on, and why it matters

`GET /health` reports it, and the UI shows it. The three differ in kind, not just in quality:

| backend | needs | per review | reviews new text? |
|---|---|---|---|
| `anthropic` | a key | ~45-70s, ~$0.10-0.13 | yes |
| `ollama` | `ollama serve`, `qwen2.5:1.5b` | ~4 min, free | yes, poorly |
| `fake` | nothing | instant, free | **no** |

`fake` replays recorded answers for the sample pieces. It is the right choice for a scripted
walkthrough and the wrong one the moment anybody edits the text, because an unrecognised piece
comes back with no findings. Switch to a real backend for the "let me try something" part of a
demo.

On `ollama` both tiers deliberately use the same small model: see the comment on `DEFAULT_MODELS`
in `platform_core/llm.py` for why a larger local model makes a review not finish at all.

## Adding a tenant

`make new-tenant ID=... PRODUCT=...` scaffolds one and `make check-tenant ID=...` says what is
still a stub. See [adding a tenant](adding-a-tenant.md).

## Tagging a new known-good version

After testing by hand:

```bash
git tag -a demo-YYYY-MM-DD -m "what you verified, and what was open"
git push origin demo-YYYY-MM-DD
```

Record what you actually exercised rather than "works" -- a tag whose message says which paths
were tested is the difference between a snapshot you can trust and one you have to re-test.
