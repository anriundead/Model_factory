# Task 4 Report

Status: DONE

Commits:
- `feat: add model publication tasks and API`

Files changed:
- `mergeKit_beta/app/model_publication_tasks.py`
- `mergeKit_beta/app/repositories/__init__.py`
- `mergeKit_beta/app/routes.py`
- `mergeKit_beta/app/services.py`
- `mergeKit_beta/merge_manager.py`
- `mergeKit_beta/tests/test_model_publication_tasks.py`
- `mergeKit_beta/tests/test_model_publication_routes.py`

Design notes:
- Publication materialization remains staging-only until explicit physical GPU validation. Existing Task 3 manifest, commit, registration, reconciliation, locking, and deletion authority are reused.
- `CUDA_VISIBLE_DEVICES` is never changed in the Flask worker. Functional validation has a CLI child-process contract and scopes the explicit GPU IDs only in that child environment.
- GPU validation rejects empty or duplicate selection and GPU 2, then performs a publication-local read-only `nvidia-smi` UUID/memory/compute-process preflight plus `core.gpu_topology.query_gpus` before functional validation.
- Existing models are copied independently file-by-file with cancellation checks. Recipe outputs accept a staging override; VLM recipes run `materialize_full_vlm` after language-model materialization.
- Queued publications are re-enqueued after restart. Interrupted materializing/running publication tasks become failed with `error_code=interrupted`; validating and registration_pending states remain available to their explicit validation/reconciliation owners.
- Admin-only routes resolve sources from a managed relative recipe or core model ID, implement idempotency, explicit validation, cancellation, manifest access, and Task 3 protected deletion.

TDD failure evidence:
- Initial focused red command:
  `docker compose --env-file /home/a/Workspace/Model_factory/.env -f docker-compose.yml -f mergeKit_beta/logs/model_gateway/acceptance/20260716_model_publication_baseline/compose-task0-override.yml run --rm --no-deps mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests/test_model_publication_tasks.py tests/test_model_publication_routes.py -v`
  Result: expected failure. All three task tests failed with `ModuleNotFoundError: No module named 'app.model_publication_tasks'`. The initial route fixture also exposed the missing `model_gateway` SQLAlchemy bind, corrected in the focused test fixture before implementation.
- VLM red command:
  `docker compose --env-file /home/a/Workspace/Model_factory/.env -f docker-compose.yml -f mergeKit_beta/logs/model_gateway/acceptance/20260716_model_publication_baseline/compose-task0-override.yml run --rm --no-deps mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests/test_model_publication_tasks.py -v`
  Result: expected failure. `test_vlm_recipe_materializes_full_visual_model` failed with `Expected 'materialize_full_vlm' to have been called once. Called 0 times.`

Focused tests:
- Command:
  `docker compose --env-file /home/a/Workspace/Model_factory/.env -f docker-compose.yml -f mergeKit_beta/logs/model_gateway/acceptance/20260716_model_publication_baseline/compose-task0-override.yml run --rm --no-deps mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests/test_model_publication_tasks.py tests/test_model_publication_routes.py -v`
- Result: `Ran 6 tests in 0.632s`, `OK`.

Full tests:
- Command:
  `docker compose --env-file /home/a/Workspace/Model_factory/.env -f docker-compose.yml -f mergeKit_beta/logs/model_gateway/acceptance/20260716_model_publication_baseline/compose-task0-override.yml run --rm --no-deps mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests`
- Result: `Ran 235 tests in 12.513s`, `OK`.
- `git diff --check`: passed with no output.

GPU process before/after:
- Before: GPU 0 `130/24576 MiB`, GPU 1 `19/24576 MiB`, GPU 2 `19/24576 MiB`, GPU 3 `19/24576 MiB`; UUIDs were `GPU-23348268-6430-c539-b7e5-762583f50e91`, `GPU-3f409b14-e414-b97e-346b-5de726e75aaa`, `GPU-7eff453d-60f0-37ed-92c1-0aec341c497d`, and `GPU-ddf96bec-7977-9a3c-8508-602946b44a56`; `nvidia-smi --query-compute-apps` returned no entries.
- After: identical UUIDs and memory values; `nvidia-smi --query-compute-apps` returned no entries.
- The ephemeral test container reported that no NVIDIA driver was available. No Task 4 test started a GPU/model process.

Concerns:
- Task 7 remains responsible for running the real model execution acceptance gates, including a real CMMMU sample. Task 4 provides the GPU-scoped subprocess interface and VLM image-validation path without running GPU workloads in automated tests.
