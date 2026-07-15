# Model Serving Acceptance Run: 2026-07-12

## Purpose

Run the first end-to-end acceptance of a real 7B merged model through the
administrator publish API and the OpenAI-compatible user API. The model service
uses one GPU and is stopped after the smoke test.

## Security Record

- `MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN` was generated with 256 bits of entropy and is
  stored only in the ignored root `.env` file.
- The token value must not be copied into this document, application logs,
  recipes, task metadata, or shell history.
- A temporary user API key is created for this acceptance run, then revoked and
  its local plaintext file is removed.

## Component And Document Paths

| Purpose | Path |
| --- | --- |
| Compose service and runtime configuration | `/home/a/Workspace/Model_factory/docker-compose.yml` |
| Local secrets, ignored by Git | `/home/a/Workspace/Model_factory/.env` |
| Flask application bootstrap and restart recovery | `/home/a/Workspace/Model_factory/mergeKit_beta/app/__init__.py` |
| Serving API, authentication, ORM and vLLM process management | `/home/a/Workspace/Model_factory/mergeKit_beta/app/model_gateway/` |
| Serving portal template and assets | `/home/a/Workspace/Model_factory/mergeKit_beta/templates/model_gateway/console.html`, `/home/a/Workspace/Model_factory/mergeKit_beta/static/model_gateway/console.css`, `/home/a/Workspace/Model_factory/mergeKit_beta/static/model_gateway/console.js` |
| Serving configuration | `/home/a/Workspace/Model_factory/mergeKit_beta/config.py` |
| Serving tests | `/home/a/Workspace/Model_factory/mergeKit_beta/tests/model_gateway/test_gateway_auth.py`, `/home/a/Workspace/Model_factory/mergeKit_beta/tests/model_gateway/test_gateway_runtime.py`, `/home/a/Workspace/Model_factory/mergeKit_beta/tests/model_gateway/test_gateway_routes.py`, `/home/a/Workspace/Model_factory/mergeKit_beta/tests/model_gateway/test_gateway_portal.py` |
| Architecture and lifecycle design | `/home/a/Workspace/Model_factory/mergeKit_beta/docs/model_gateway/ARCHITECTURE.md` |
| Implementation history and fusion evidence | `/home/a/Workspace/Model_factory/mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md` |
| This acceptance record | `/home/a/Workspace/Model_factory/mergeKit_beta/docs/model_gateway/ACCEPTANCE_20260712.md` |
| Source model output used for this run | `/home/a/Workspace/Model_factory/mergeKit_beta/merges/eec970f3/output` |
| Fusion metadata and logs | `/home/a/Workspace/Model_factory/mergeKit_beta/merges/eec970f3/metadata.json`, `/home/a/Workspace/Model_factory/mergeKit_beta/merges/eec970f3/bridge.log`, `/home/a/Workspace/Model_factory/mergeKit_beta/merges/eec970f3/subprocess_output.log` |
| Fusion recipe | `/home/a/Workspace/Model_factory/mergeKit_beta/recipes/eec970f3.json` |
| Container project path | `/app/ServiceEndFiles/Workspaces/mergeKit_beta` |

## Acceptance Profile

- Fusion task: `eec970f3`, using the two 7B text-capable Qwen-family source
  models recorded in its recipe and metadata. The natural evolutionary run
  completed with `n_eval=20` and produced the output path listed above.
- Published service ID: `784e0284-d5d0-4062-8b12-9daf8a905c80`; service model
  name: `serving-smoke-7b-acceptance`.
- Runtime GPU: GPU `3` only; GPU `2` remains reserved for an external service.
- vLLM safety settings: `bfloat16`, `gpu_memory_utilization=0.78`,
  `max_model_len=4096`, `max_num_seqs=2`, `max_num_batched_tokens=4096`.
- User smoke: `GET /v1/models`, followed by one non-streaming
  `POST /v1/chat/completions` request and request-status verification.
- Cleanup: revoke the temporary user key, stop the serving model, remove the
  temporary plaintext key file, and confirm GPU 3 is released.

## Results

- The first disposable service definition was rejected by vLLM because
  `max_num_batched_tokens=2048` was lower than `max_model_len=4096`. No model
  process remained after that failed startup. The accepted profile above fixes
  only that invalid relation.
- vLLM loaded the merged model on GPU 3 and passed its loopback health check.
- A temporary, single-model user API key listed exactly one available model.
  Its non-streaming OpenAI-compatible chat request returned HTTP 200 with
  `serving smoke passed`, `prompt_tokens=26`, `completion_tokens=5`, and
  `total_tokens=31`. The persisted serving request was `success` and its usage
  record matched the returned values.
- The temporary user key was changed to `revoked`, and all plaintext temporary
  key and response files were removed from `/tmp`.
- The service was stopped after the request. GPU 3 returned to its 18 MiB
  baseline; GPU 2 remained at its pre-existing external-service allocation.

## Corrective Note: Recovery Process Boundary

During this acceptance, an independent maintenance Python process imported the
Flask app. Before correction, every such import executed restart recovery and
could clear the PID of a still-running vLLM process. The running model then
remained outside the database lifecycle.

The recovery action is now gated by `MERGEKIT_MODEL_GATEWAY_RUNTIME_PROCESS=1`, set
only by `/home/a/Workspace/Model_factory/mergeKit_beta/start_app.sh`. Thus the
long-lived Flask process still restores services to manual management after a
real restart, while CLI and maintenance imports do not change live service
state. The regression test is in
`/home/a/Workspace/Model_factory/mergeKit_beta/tests/model_gateway/test_gateway_runtime.py`.

## Final Verification

- `mergekit-beta` was force-recreated through Docker Compose and became
  healthy at port `5000`; its working directory is
  `/app/ServiceEndFiles/Workspaces/mergeKit_beta`.
- `/healthz`, `/readyz`, `/api/models`, `/api/testset/list`, and
  `/api/history` all returned valid responses after the restart.
- The container import check for `merge_manager`, `app`, `evolution.runner`,
  and `app.model_gateway.runtime` succeeded.
- `python -m unittest discover -s tests` completed with `63` passing tests.
- No vLLM or Ray serving process remained after cleanup. GPU 0, GPU 1, and
  GPU 3 were each at 18 MiB used; GPU 2 retained only its pre-existing external
  workload at 17,370 MiB.

## Model Gateway Regression Run

After the namespace migration, the real merged 7B model was started again
through `/api/model-gateway/admin/*` and exercised through the unchanged user
`/v1/*` API. A temporary API key restricted to the single published model
listed exactly that model, then completed a non-streaming chat response with
`prompt_tokens=27`, `completion_tokens=5`, and `total_tokens=32`. Its request
status endpoint reported `success` and the persisted usage matched the API
response. The temporary key was revoked and its plaintext files were removed.

The regression also identified and corrected a shutdown safety defect: a
missing persisted PID could leave a marked vLLM child on GPU 3, while a zombie
child could be mistaken for a live process. `stop_service` now locates only a
process with the exact `MERGEKIT_MODEL_GATEWAY_SERVICE_ID` marker and treats a
Linux zombie process as exited. A real start/stop lifecycle verified that the
service ended as `stopped`, no vLLM or Ray process remained, and GPU 3 returned
to 18 MiB used. The suite completed with 65 passing tests.

### Follow-up: Running Request Cancellation

Running request cancellation is now accepted by the real-model regression:

- Users may provide a UUID `X-Request-Id` before sending a chat request; the
  gateway persists it, returns it, and forwards it to vLLM.
- The vLLM compatibility entrypoint exposes a loopback-only abort route guarded
  by the per-service internal key. It calls the vLLM 0.7.0 engine client's
  `abort()` method for `chatcmpl-<gateway request id>`.
- A 7B request was observed as `running`, cancelled through the user API, and
  finalized as `canceled` with `user_canceled`; vLLM reported zero running
  requests and no usage record was created for the cancelled inference.
- The portal generates a UUID before submission, so its cancellation control
  has the request handle without waiting for the normal 60-second sync window.

Stopping a vLLM service now also enumerates all live processes carrying its
service marker, rather than assuming the wrapper and engine share one PID.
The final real start/stop regression ended in `stopped` with no vLLM or Ray
process left and GPU 3 at 18 MiB used.

## Rollback

1. Stop the service through `POST /api/model-gateway/admin/model-services/<id>/stop`.
2. Revoke the temporary API key in `serving_api_keys`.
3. Remove the local temporary key file under `/tmp`.
4. To disable administrator API access, remove `MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN`
   from `/home/a/Workspace/Model_factory/.env` and recreate only
   `mergekit-beta`.
