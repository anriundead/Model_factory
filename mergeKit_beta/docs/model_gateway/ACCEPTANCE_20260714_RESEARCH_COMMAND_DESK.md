# Research Command Desk Acceptance (2026-07-14)

## Scope

This record covers the user-facing `/research` Command Desk visual batch only.
It must preserve all existing Gateway request, source, archive and local-only
conversation behavior.

## Baseline

- `docker compose config --quiet`: passed.
- `docker compose ps`: `mergekit-beta` healthy; ClamAV, PostgreSQL and Redis
  healthy; research worker running.
- `/healthz`, `/readyz`, `/api/models`, `/api/testset/list`, `/api/history`,
  `/research` and `/model-gateway`: all returned HTTP 200.
- `python -m unittest discover -s tests`: 137 passed in 9.118 seconds.
- `node --check` passed for `research.js` and `research_archive.js`.
- Existing warnings: Swig deprecation messages and automatic CUDA-platform
  detection. Neither produced a test failure.
- Baseline worktree is already dirty outside this batch; its full `git status`
  was recorded before edits. This batch must not modify or clean those files.

## GPU And Process Baseline

- GPU 0, 1 and 3: 18 MiB, no compute process.
- GPU 2: external `/usr/bin/python3` PID 1667367 using 1116 MiB and external
  `VLLM::EngineCore` PID 1720239 using 12484 MiB, total 13630 MiB.
- No model, merge, evaluation, Ray, vLLM, upload, URL import or research task
  has been started by this visual batch.

## Rollback

Pre-change archive:

`/tmp/mergenetic_command_desk_20260714_224048.tar.gz`

SHA-256:

`9382f34bfe2f006c625050027b64f3490e413308894be35d297a1079f666f9a9`

It contains only the pre-change research template, CSS, JavaScript, portal
test and skills record. To roll back this batch, extract the archive into a
temporary directory and copy only those five files back to the same workspace
paths, then repeat the baseline commands. Do not reset Git, change database
state, stop Compose services, or touch external GPU 2 processes.

## Automated Gates

- TDD Red: the two Command Desk contracts failed before production changes
  because the old portal did not contain the new material tokens, semantic
  regions or interaction markers.
- Portal contracts: `python -m unittest tests.model_gateway.test_gateway_portal`
  passed 24 tests after implementation.
- Full suite: `python -m unittest discover -s tests` passed 139 tests in
  9.160 seconds. The two additional tests are the Command Desk contracts.
- JavaScript syntax: `node --check research.js` and `research_archive.js`
  passed.
- Whitespace: `git diff --check` passed.
- CSS prohibition scan found no `linear-gradient`, `radial-gradient`,
  `backdrop-filter`, `animation: infinite`, or legacy `#D9362B` primary token.
- Runtime smoke after implementation: all seven baseline endpoints returned
  HTTP 200.

The visual browser acceptance is intentionally pending user review, per the
selected continuous-execution checkpoint.

## Browser Evidence

Pending user review and synthetic-data browser acceptance. Browser tests must
use local synthetic data only and must not send an API Key or invoke a model.
