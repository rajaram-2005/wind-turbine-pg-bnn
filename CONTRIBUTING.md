# Contributing to Aerovigil PG-BNN

Thank you for your interest in contributing to Aerovigil PG-BNN! This document provides guidelines for contributing to this project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

*   Use a clear and descriptive title
*   Describe the exact steps which reproduce the problem
*   Provide specific examples to demonstrate the steps
*   Describe the behavior you observed after following the steps
*   Explain which behavior you expected to see instead

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

*   Use a clear and descriptive title
*   Provide a step-by-step description of the suggested enhancement
*   Include specific examples to demonstrate the enhancement
*   Explain why this enhancement would be useful

### Pull Requests

*   Fill in the required template
*   Do not include issue numbers in the PR title
*   Include screenshots and animated GIFs in your pull request whenever possible
*   Follow the Python style guide
*   End all files with a newline
*   Make sure your code lints and tests pass

## Development Setup

```bash
# Clone the repository
git clone https://github.com/rajaram-2005/wind-turbine-pg-bnn.git
cd wind-turbine-pg-bnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,demo]"

# Install pre-commit hooks
pre-commit install
```

## Coding Standards

### Python Style

*   Follow PEP 8 guidelines
*   Use type hints for all function signatures
*   Write docstrings for all public functions and classes
*   Keep functions focused and small

### Code Quality

The project ships a [`Makefile`](Makefile) and a portable
[`scripts/ci.sh`](scripts/ci.sh) that run the full quality pipeline. For
day-to-day work:

```bash
make lint          # ruff check
make format        # ruff format (auto-fix in place)
make format-check  # ruff format --check (no writes)
make typecheck     # mypy on the packaged model
make security      # bandit scan
make test          # pytest
make build         # sdist + wheel (+ twine check)
```

### Running CI locally

Run the entire pipeline — the same checks any hosted CI runs — with one
command:

```bash
make ci            # if you have `make`
bash scripts/ci.sh # portable, no dependencies
```

`scripts/ci.sh` is intentionally portable: invoke it from CircleCI, Jenkins,
GitLab, Drone, or a GitHub-hosted runner to get identical checks everywhere.

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
feat: add new feature
fix: fix a bug
docs: documentation only changes
style: formatting, missing semi colons, etc; no code change
refactor: refactoring production code
test: adding missing tests, refactoring test; no production code change
chore: updating build tasks, package manager configs, etc; no production code change
```

## Testing

*   Write unit tests for all new functionality
*   Ensure all tests pass before submitting PR
*   Maintain or improve code coverage

## License

By contributing to Aerovigil PG-BNN, you agree that your contributions will be licensed under the MIT License.
