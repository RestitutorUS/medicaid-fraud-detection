

## 2026-03-22 — Phases 1-2: uv Migration + src/utils Refactor

**What:** 
- Phase1: Created pyproject.toml (deps pinned, dev: pytest/ruff etc.). uv lock/sync ready.
- Phase2: src/hhs_medicaid_fraud/ pkg (moved .py), utils.py (print_section, load_sample_duckdb, preprocess, execute_query_to_parquet, constants), removed dupes/added imports.

**Why:** Reproducible/fast env (uv), modular/DRY (~200 LOC saved), package-ready.

**Verify:** uv sync; cd src/hhs_medicaid_fraud; python explore_data.py (samples CSV, uses utils).

No functionality lost—string-exact replaces.