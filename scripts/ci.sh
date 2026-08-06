#!/usr/bin/env bash
# Portable local CI for Aerovigil PG-BNN — no `make` required.
# Mirrors `make ci`. Wire it into any hosted CI (CircleCI / Jenkins / GitLab /
# Drone / GitHub-hosted runner) by running:  bash scripts/ci.sh
set -euo pipefail

PKG="src/aerovigil_pg_bnn"

run() { printf '\n==> %s\n' "$*"; "$@"; }

run ruff check .
run ruff format --check .
run mypy "$PKG"
run bandit -c pyproject.toml -r "$PKG" -ll
run pytest
run python -m build
run twine check dist/*

printf '\n✅ All local CI checks passed.\n'
