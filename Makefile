# MLR Review Agent - Makefile
#
# Everything here must work with no API key. The default model backend is a local
# Ollama model; the test suite forces the FakeLLM backend so it needs neither.

.DEFAULT_GOAL := help
.PHONY: help setup seed new-tenant check-tenant api ui demo loop test lint fmt eval eval-offline eval-replay eval-baseline graph fixtures smoke clean

help: ## Show available targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install all dependencies
	uv sync
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

seed: ## Index the synthetic corpus into a per-tenant vector store
	uv run python -m scripts.seed

# Onboarding a tenant, in two halves. The scaffold writes every file the platform requires with a
# valid-but-TODO body; the check says which of those bodies are still stubs. Split that way on
# purpose: a half-finished tenant seeds and runs perfectly happily, and then answers "cannot
# verify" to every claim, which reads as a model problem rather than as missing data.
new-tenant: ## Scaffold a tenant: make new-tenant ID=acme_two PRODUCT=cardiva [NAME="..." PRODUCT_NAME="..."]
	@test -n "$(ID)" || (echo 'usage: make new-tenant ID=<tenant_id> PRODUCT=<product_id>'; exit 1)
	@test -n "$(PRODUCT)" || (echo 'usage: make new-tenant ID=<tenant_id> PRODUCT=<product_id>'; exit 1)
	uv run python -m scripts.new_tenant --id $(ID) --product $(PRODUCT) \
		$(if $(NAME),--name "$(NAME)",) $(if $(PRODUCT_NAME),--product-name "$(PRODUCT_NAME)",)

check-tenant: ## Report what a tenant is still missing: make check-tenant ID=acme_two
	@test -n "$(ID)" || (echo 'usage: make check-tenant ID=<tenant_id>'; exit 1)
	uv run python -m scripts.check_tenant --id $(ID)

# Backend for the demo processes. Overridable: `MLR_LLM_BACKEND=anthropic make demo`.
#
# Defaults to `fake` because the alternatives make the demo look broken rather than slow. A local
# model takes roughly two minutes per call on a CPU-only machine, so a review that makes nine calls
# sits at "running" for a quarter of an hour with no outward sign of progress, and `anthropic` needs
# a key that a fresh clone does not have. Under `fake` the LLM findings come from a fixture and
# everything else runs for real; the API reports the backend on /health and the service returns the
# sentence the UI should show.
# Taken from .env when it names one, because .env is where a reader was told to configure this and
# an exported default here would beat it -- which is how a key set in .env came to do nothing at
# all. Falls back to fake so a fresh clone with no key still has a working demo, and an explicit
# `MLR_LLM_BACKEND=anthropic make demo` still wins over both.
MLR_LLM_BACKEND ?= $(or $(shell sed -n 's/^MLR_LLM_BACKEND=//p' .env 2>/dev/null | tail -1),fake)
export MLR_LLM_BACKEND

api: ## Run the FastAPI backend on :8000
	uv run uvicorn api.main:app --reload --port 8000

ui: ## Run the Streamlit frontend on :8501
	uv run streamlit run ui/Home.py --server.port 8501

demo: ## Run API + UI together (the screen-share entry point)
	@echo "API  -> http://localhost:8000/docs"
	@echo "UI   -> http://localhost:8501"
	@echo "backend: $(MLR_LLM_BACKEND)"
	@trap 'kill 0' INT TERM; \
	uv run uvicorn api.main:app --port 8000 & \
	uv run streamlit run ui/Home.py --server.port 8501 & \
	wait

loop: ## Drive one review through the full loop (no API key). TRACE=1 prints each node's state
	$(if $(TRACE),MLR_TRACE=$(TRACE),) uv run python -m scripts.loop $(if $(PIECE),--piece $(PIECE),) $(if $(STEP),--step,) $(if $(TENANT),--tenant $(TENANT),)

test: ## Run the test suite (offline, no API key, no Ollama)
	uv run pytest

lint: ## Lint and format-check
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Autoformat
	uv run ruff check --fix .
	uv run ruff format .

eval: ## Run the golden set against a live model (needs a key; records nothing)
	uv run python -m scripts.run_evals --mode live --threshold-sweep default

# The offline and replay targets pin the backend instead of inheriting it. These three are the
# reproducibility contract -- a stranger with no API key must get the committed baseline back -- and
# they stopped honouring it the moment `.env` started being read, because the ambient backend then
# became whatever the last person configured for a live demo. An `anthropic` backend made
# `make eval-offline` hang instead of failing, which is the worst way for a hermetic target to break.
eval-offline: ## Run the golden set offline and compare against the committed baseline
	MLR_LLM_BACKEND=fake uv run python -m scripts.run_evals --mode offline --compare baseline --threshold-sweep default

eval-replay: ## Replay recorded real-model cassettes (deterministic AND a real measurement)
	MLR_LLM_BACKEND=fake uv run python -m scripts.run_evals --mode replay --compare baseline --threshold-sweep default

eval-baseline: ## Re-record the committed baseline from the current code
	MLR_LLM_BACKEND=fake uv run python -m scripts.run_evals --mode offline --threshold-sweep default --write-baseline

smoke: ## Run real promo pieces against a live model (local Ollama by default)
	uv run python -m scripts.smoke

fixtures: ## Regenerate the API example responses from the models (never hand-edit them)
	uv run python -m scripts.gen_api_fixtures

graph: ## Re-render the compiled graph to docs/graph.mmd
	uv run python -m scripts.render_graph

clean: ## Remove runtime data (databases, vector store, eval results)
	rm -rf data/ evals/results/* .pytest_cache .ruff_cache
