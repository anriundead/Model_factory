# VALIDATION_LOG

## Gate 1 Baseline

- `docker compose ps`: passed, service healthy.
- `docker compose exec -T mergekit-beta pwd`: passed.
- HTTP smoke: passed for `/healthz`, `/readyz`, `/api/models`, `/api/testset/list`, `/api/history`.
- Import check in `mergenetic`: passed.
- Unit tests in `mergenetic`: passed, `Ran 29 tests ... OK`.

## Gate 3 Validation

### Batch 3a

Deleted:

- `mergeKit_beta/investigate_4479.py`
- `mergeKit_beta/verify_subset.py`

Validation:

- `docker compose ps`: passed, service healthy.
- `/healthz`, `/readyz`, `/api/models`, `/api/testset/list`, `/api/history`: passed.
- Import check in `mergenetic`: passed.
- Unit tests in `mergenetic`: passed, `Ran 29 tests ... OK`.

### Batch 3b

Deleted:

- `mergeKit_beta/verify_test.py`

Validation:

- `docker compose ps`: passed, service healthy.
- `/healthz`, `/readyz`, `/api/models`, `/api/testset/list`, `/api/history`: passed.
- Import check in `mergenetic`: passed.
- Unit tests in `mergenetic`: passed, `Ran 29 tests ... OK`.

### Batch 3c

Deleted:

- `mergeKit_beta/scripts/patch_lm_eval_transformers5.py`

Validation:

- `docker compose ps`: passed, service healthy.
- `/healthz`, `/readyz`, `/api/models`, `/api/testset/list`, `/api/history`: passed.
- Import check in `mergenetic`: passed.
- Unit tests in `mergenetic`: passed, `Ran 29 tests ... OK`.

### Batch 3d

Deleted:

- `mergeKit_beta/eval_worker.py`

Validation:

- `docker compose ps`: passed, service healthy.
- `/healthz`, `/readyz`, `/api/models`, `/api/testset/list`, `/api/history`: passed.
- Import check in `mergenetic`: passed.
- Unit tests in `mergenetic`: passed, `Ran 29 tests ... OK`.

## Gate 4 Repo Hygiene

- `.gitignore` already covers `__pycache__/`, `*.pyc`, `.pytest_cache/`, logs, and merge outputs.
- No `.gitignore` change was needed.
- Python caches recreated by tests were removed from inside the container because they were root-owned bind-mount artifacts.
