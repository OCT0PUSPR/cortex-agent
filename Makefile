# cortex-agent — developer & ops tasks.
# Run `make` or `make help` for the list of targets.

# Use bash for recipes (consistent behaviour across platforms).
SHELL := bash

# Allow overriding the interpreter, e.g. `make PYTHON=python3.11 test`.
PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help

.PHONY: help install install-dev install-train test lint format typecheck \
        security migrate run worker train docker-build docker-up docker-down \
        clean all

help: ## Show this help.
	@echo "cortex-agent — available targets:"
	@echo
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- setup ------------------------------------------------------------------
install: ## Install runtime dependencies.
	$(PIP) install -r requirements.txt

install-dev: ## Install minimal dev/CI deps + the package (editable).
	$(PIP) install -r requirements-min.txt
	$(PIP) install -e .

install-train: ## Install TinyBrain training deps (torch, etc.).
	$(PIP) install -r requirements-train.txt

# --- quality ----------------------------------------------------------------
test: ## Run the test suite with coverage.
	$(PYTHON) -m pytest --cov=cortex --cov-report=term-missing

lint: ## Lint and verify formatting (no changes made).
	$(PYTHON) -m ruff check cortex tests
	$(PYTHON) -m ruff format --check cortex tests

format: ## Auto-format the codebase.
	$(PYTHON) -m ruff format cortex tests

typecheck: ## Static type-check with mypy.
	$(PYTHON) -m mypy cortex --ignore-missing-imports

security: ## Run security scanners (bandit + pip-audit).
	$(PYTHON) -m bandit -r cortex -c pyproject.toml
	$(PYTHON) -m pip_audit

# --- database ---------------------------------------------------------------
migrate: ## Apply database migrations (alembic upgrade head).
	$(PYTHON) -m alembic upgrade head

# --- run --------------------------------------------------------------------
run: ## Run the API server with autoreload.
	$(PYTHON) -m uvicorn cortex.api.server:app --reload

worker: ## Run the arq queue worker.
	$(PYTHON) -m arq cortex.worker.worker_settings.WorkerSettings

train: ## Train the from-scratch TinyBrain model.
	$(PYTHON) -m cortex.tinybrain.train

# --- docker -----------------------------------------------------------------
docker-build: ## Build the Docker image (cortex-agent:latest).
	docker build -t cortex-agent:latest .

docker-up: ## Build and start the full stack (api + worker + redis + postgres).
	docker compose up --build

docker-down: ## Stop the stack and remove containers.
	docker compose down

# --- housekeeping -----------------------------------------------------------
clean: ## Remove caches and local SQLite/state artifacts.
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf build dist htmlcov
	rm -f .coverage .coverage.*
	rm -f .cortex/*.sqlite .cortex/*.db

all: lint typecheck test ## Run lint, typecheck, and tests.
