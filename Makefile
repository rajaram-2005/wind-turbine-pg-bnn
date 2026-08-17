# Aerovigil PG-BNN — local CI / task runner.
#
# `make ci` runs the full quality pipeline (the same checks any hosted CI runs).
# Individual targets give faster feedback during development.
# Portable equivalent (no `make` required): `bash scripts/ci.sh`.

PYTHON ?= python
PKG    := src/aerovigil_pg_bnn

.PHONY: help dev install serve lint format format-check typecheck security test build apps release-apps docker-lint ci clean

help: ## Show this help
	@echo "Aerovigil PG-BNN — local CI targets (run 'make ci' for everything):"
	@echo "  make ci           full pipeline: lint, format-check, typecheck, security, test, build"
	@echo "  make serve        one-port dashboard + all APIs on http://localhost:8080"
	@echo "  make lint         ruff check"
	@echo "  make format       ruff format (writes)"
	@echo "  make format-check ruff format --check"
	@echo "  make typecheck    mypy on the packaged model"
	@echo "  make security     bandit scan"
	@echo "  make test         pytest"
	@echo "  make build        sdist + wheel (+ twine check)"
	@echo "  make apps         package native multiplatform apps (dry-run)"
	@echo "  make release-apps package native multiplatform apps (Windows, macOS, Linux, Android)"
	@echo "  make docker-lint  hadolint on the Dockerfile (if installed)"
	@echo "  make dev          install api + dev dependencies"
	@echo "  make clean        remove build artifacts"

dev: ## Install api + dev dependencies (lint, test, build, serve tools)
	$(PYTHON) -m pip install -e ".[api,dev,demo]"

install: dev ## Alias for dev

serve: ## Run the complete application through one process and port
	$(PYTHON) -m uvicorn src.unified_app:app --host 0.0.0.0 --port 8080

lint: ## Lint with ruff
	ruff check .

format: ## Format with ruff (writes)
	ruff format .

format-check: ## Check formatting without writing (CI mode)
	ruff format --check .

typecheck: ## Type-check the packaged model with mypy
	mypy $(PKG)

security: ## Security scan the packaged model with bandit
	bandit -c pyproject.toml -r $(PKG) -ll

test: ## Run the test suite
	pytest

build: ## Build sdist + wheel and validate metadata
	$(PYTHON) -m build
	twine check dist/*.tar.gz dist/*.whl

apps: ## Package native multiplatform apps (dry-run mode)
	$(PYTHON) scripts/build_apps.py --dry-run

release-apps: ## Build and package native apps across all platforms
	$(PYTHON) scripts/build_apps.py --platform all

docker-lint: ## Lint the Dockerfile with hadolint (if installed)
	@command -v hadolint >/dev/null 2>&1 && hadolint -c .hadolint.yaml Dockerfile \
		|| echo "hadolint not installed; skipping (brew/pip install hadolint)"

ci: lint format-check typecheck security test build ## Run the full local CI pipeline
	@echo ""
	@echo "✅ All local CI checks passed."

clean: ## Remove build artifacts
	rm -rf dist build *.egg-info src/*.egg-info
