# Contributing to ctrldoc

Thanks for your interest. This document covers how to get a working development environment, how the codebase is organized, and the standards every change must meet.

## Development setup

```bash
git clone https://github.com/<your-username>/ctrldoc.git
cd ctrldoc
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
pytest                       # full suite
pytest -m "not slow"         # fast tests only
pytest -m family_niah        # one test family (see docs/TESTING.md)
pytest --cov=ctrldoc         # with coverage
```

## Code standards

- **Formatter:** `ruff format`
- **Linter:** `ruff check`
- **Type checker:** `mypy --strict src/ctrldoc/`
- **Tests:** every public function or class has at least one test.
- **Spec traceability:** every test file's module docstring must include a `SPEC-REF: §X.Y` line pointing at the section of [docs/SPEC.md](docs/SPEC.md) it covers.

## Commit style

Conventional Commits:

```
feat(L2): add reciprocal rank fusion to retrieval planner
fix(ingest): handle malformed PDF without panicking
test(verifier): add NLI calibration suite
docs(spec): clarify evidence pack token budget
```

Every commit message must include a `SPEC-REF: §X.Y` line in the trailer.

## Pull requests

- One logical change per PR.
- CI must pass (tests, lint, type check, public-leak scan, spec-trace coverage).
- Update [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]`.
- If the change introduces or modifies a design choice, add an ADR to [docs/DECISIONS.md](docs/DECISIONS.md).

## Reporting issues

Please include:

- A minimal reproducer (the smallest input that triggers the bug).
- Environment details (`python --version`, OS, `ctrldoc --version`).
- Expected vs. actual output.
- Relevant lines from `traces/<run_id>.jsonl` if applicable.

## Security

Please do not file public issues for security vulnerabilities. Email the maintainer directly (see repo metadata) and allow 30 days for triage before public disclosure.
