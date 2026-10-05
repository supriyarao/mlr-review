# MLR Review Agent - Makefile
#
# Everything here must work with no API key. The default model backend is a local
# Ollama model; the test suite forces the FakeLLM backend so it needs neither.

.DEFAULT_GOAL := help
.PHONY: help setup seed api ui demo test lint fmt eval eval-offline eval-replay eval-baseline graph fixtures smoke clean

help: ## Show available targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install all dependencies
	uv sync
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

seed: ## Index the synthetic corpus into a per-tenant vector store
	uv run python -m scripts.seed

api: ## Run the FastAPI backend on :8000
	uv run uvicorn api.main:app --reload --port 8000

ui: ## Run the Streamlit frontend on :8501
	uv run streamlit run ui/Home.py --server.port 8501

demo: ## Run API + UI together (the screen-share entry point)
	@echo "API  -> http://localhost:8000/docs"
	@echo "UI   -> http://localhost:8501"
	@trap 'kill 0' INT TERM; \
	uv run uvicorn api.main:app --port 8000 & \
	uv run streamlit run ui/Home.py --server.port 8501 & \
	wait

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

eval-offline: ## Run the golden set offline and compare against the committed baseline
	uv run python -m scripts.run_evals --mode offline --compare baseline --threshold-sweep default

eval-replay: ## Replay recorded real-model cassettes (deterministic AND a real measurement)
	uv run python -m scripts.run_evals --mode replay --compare baseline --threshold-sweep default

eval-baseline: ## Re-record the committed baseline from the current code
	uv run python -m scripts.run_evals --mode offline --threshold-sweep default --write-baseline

smoke: ## Run real promo pieces against a live model (local Ollama by default)
	uv run python -m scripts.smoke

fixtures: ## Regenerate the API example responses from the models (never hand-edit them)
	uv run python -m scripts.gen_api_fixtures

graph: ## Re-render the compiled graph to docs/graph.mmd
	uv run python -m scripts.render_graph

clean: ## Remove runtime data (databases, vector store, eval results)
	rm -rf data/ evals/results/* .pytest_cache .ruff_cache
