# Cleanup Baseline 20260704_102317

## Scope

- Continuation of the low-risk cleanup plan.
- Compose root: `/home/a/Workspace/Model_factory`
- Service: `mergekit-beta`
- Container workdir verified as `/app/ServiceEndFiles/Workspaces/mergeKit_beta`.

## Gate 1 Baseline

- `docker compose ps`: `model_factory-mergekit-beta-1` is `Up` and `healthy`.
- `docker compose exec -T mergekit-beta pwd`: `/app/ServiceEndFiles/Workspaces/mergeKit_beta`.
- HTTP smoke passed:
  - `/healthz`
  - `/readyz`
  - `/api/models`
  - `/api/testset/list`
  - `/api/history`
- Import check passed in the `mergenetic` conda environment:
  - `import merge_manager; from app import app; import evolution.runner`
- Unit tests passed:
  - `python -m unittest discover -s tests`
  - `Ran 29 tests ... OK`

## Notes

- Worktree already had many unrelated modified and untracked files before this batch.
- This cleanup batch is limited to the tracked deprecated/debug script candidates listed in `DELETE_MANIFEST.tracked.md`.
- Core evaluation and evolution files are out of scope.
