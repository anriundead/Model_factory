# Cleanup Baseline - 20260703_171018

## Scope

Planned cleanup run for `mergeKit_beta`, using Docker Compose service `mergekit-beta`.

No cleanup was executed in this run because Gate 1 baseline runtime failed.

## Gate 0: Preflight

Status: PASS

- Working directory: `/home/a/Workspace/Model_factory`
- Compose service: `mergekit-beta`
- `docker compose config --quiet`: PASS
- `docker compose config --services`: `mergekit-beta`
- Image: `mergekit-beta:latest`
- Image id: `sha256:a9dfadb3b6c4e625a847d916666d7a1664f7d992a3ea815bcdf4f2ff73d01767`
- Image created: `2026-06-07T16:42:36.424711952+08:00`
- Port 5000 before startup: not occupied

## GPU Check

Host GPUs:

- `GPU-23348268-6430-c539-b7e5-762583f50e91`
- `GPU-3f409b14-e414-b97e-346b-5de726e75aaa`
- `GPU-7eff453d-60f0-37ed-92c1-0aec341c497d`
- `GPU-ddf96bec-7977-9a3c-8508-602946b44a56`

Compose device ids match the host GPU UUIDs.

## Resolved Volume Sources

- `/home/a/Workspace/Model_factory/mergeKit_beta` -> `/app/ServiceEndFiles/Workspaces/mergeKit_beta`
- `/home/a/Model_factory_data/models` -> `/data/Models`
- `/home/a/Model_factory_data/models_pool` -> `/data/models_pool`
- `/home/a/Model_factory_data/packages` -> `/app/ServiceEndFiles/Packages`
- `/home/a/Model_factory_data/merges` -> `/app/ServiceEndFiles/Workspaces/mergeKit_beta/merges`
- `/home/a/Workspace/Model_factory/hf_datasets_cache` -> `/data/hf_datasets`
- `/home/a/Workspace/Model_factory/eval_datasets_cache` -> `/data/eval_datasets`

All resolved volume source directories exist.

Environment risk noted:

- `/home/a/Model_factory_data/models_pool` owner/mode: `root:root 755`
- `/home/a/Model_factory_data/packages` owner/mode: `root:root 755`
- `/home/a/Workspace/Model_factory/hf_datasets_cache` owner/mode: `root:root 755`
- `/home/a/Workspace/Model_factory/eval_datasets_cache` owner/mode: `root:root 755`

Do not treat future cache write failures as cleanup regressions without checking permissions first.

## Git Snapshot

The worktree already had many modified and untracked files before this cleanup run.

Diff stat before cleanup:

```text
 docker-compose.yml                                 |  14 +-
 mergeKit_beta/.cursor/rules/RULES_INDEX.md         |   2 +-
 mergeKit_beta/DEVELOPMENT.md                       |  78 +++-
 mergeKit_beta/Dockerfile                           |   1 +
 mergeKit_beta/README.md                            |  25 +
 mergeKit_beta/app/__init__.py                      |  19 +-
 mergeKit_beta/app/repositories/__init__.py         |  40 ++
 mergeKit_beta/app/routes.py                        |  72 ++-
 mergeKit_beta/app/services.py                      | 515 +++++++++++++++++++--
 mergeKit_beta/config.py                            |  23 +-
 mergeKit_beta/environment.yml                      |   1 +
 mergeKit_beta/evolution/contracts.md               |  41 +-
 mergeKit_beta/evolution/runner.py                  |  89 ++--
 mergeKit_beta/evolution/vendor/vlm_merge/eval/prompt_mmlu.yaml | 7 +-
 mergeKit_beta/evolution/vendor/vlm_merge/eval_final.py | 2 +-
 mergeKit_beta/evolution/vendor/vlm_merge/run_vlm_search.py | 397 +++++++++++++---
 mergeKit_beta/scripts/verify_db_integration.py     |  50 ++
 mergeKit_beta/start_app.sh                         |   3 +-
 mergeKit_beta/static/app.js                        | 284 +++++++-----
 mergeKit_beta/static/styles.css                    |  28 +-
 mergeKit_beta/templates/index.html                 |   1 +
 21 files changed, 1373 insertions(+), 319 deletions(-)
```

## Gate 1: Baseline Runtime

Status: FAIL

Command:

```bash
docker compose up -d mergekit-beta
```

Result:

```text
Container model_factory-mergekit-beta-1 Starting
Error response from daemon: unknown or invalid runtime name: nvidia
```

Docker runtimes reported by the daemon only include `runc` and `io.containerd.runc.v2`; `nvidia` is not registered.

Because Gate 1 failed, no cleanup, deletion, movement, or `git rm --cached` was executed.
