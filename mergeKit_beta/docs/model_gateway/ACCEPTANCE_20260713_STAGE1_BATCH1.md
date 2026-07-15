# Stage 1 Batch 1 Acceptance - 2026-07-13

## Scope

- Dedicated `model_gateway` SQLAlchemy bind with SQLite compatibility fallback.
- Public URL and upload admission checks.
- Research file/job creation, user-owned status/cancel APIs, API Key disable/revoke.
- Citation-safe text chunking and TTL-bound `ResearchChunk` persistence model.
- `/research` user workspace without administrator controls.

## Fresh Verification

```text
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
Ran 78 tests ... OK

node --check mergeKit_beta/static/model_gateway/research.js
git diff --check
curl /research
curl /readyz
```

The `mergekit-beta` container was restarted after route changes. GPU 0, 1 and
3 remained at 18 MiB; GPU 2 retained its external 17,370 MiB workload.

## Explicitly Not Accepted Yet

PostgreSQL migration, Redis delivery, ClamAV, parsing, retrieval, citations,
rate limits, Nginx and real-model research smoke are not part of this batch.

## Batch 2: Runtime Dependency And Storage Cutover

- Built candidate image `mergekit-beta:research-stage1-candidate` with the pinned
  PostgreSQL, Redis, document parsing and retrieval dependencies from
  `environment.yml`. Its runtime environment contains no proxy variables.
- Preserved the pre-cutover image as
  `mergekit-beta:pre-research-stage1-20260713_164357`, then switched the running
  container to candidate image `sha256:7f38e4a9bcdb...658fe6b8`.
- Started only the `research` profile PostgreSQL and Redis services. PostgreSQL and
  Redis health checks passed; Redis AOF is enabled with `appendfsync everysec`.
- Migrated the Gateway data from the legacy SQLite database: 2 model services,
  4 API Keys, 9 requests and 7 usage rows. Source/target counts and all API Key
  hashes matched; a second migration copied zero rows.
- Confirmed the running Gateway bind reports PostgreSQL and reads 4 API Keys from
  the new database. Core model-factory SQLite remains in place.
- Fresh acceptance: `/healthz`, `/readyz`, `/research`, `/api/models`,
  `/api/testset/list` and `/api/history` passed; full container test suite ran
  90 tests successfully; `git diff --check` and profile Compose validation passed.
- GPU 0, 1 and 3 remained at 18 MiB. GPU 2 retained the same external Python and
  vLLM processes and 17,370 MiB allocation throughout.

Redis delivery code exists, but the Worker, parsing, scanning, retrieval and
real-model research execution remain explicitly unaccepted.

## Batch 3: CPU Parser Primitive

- Added isolated, no-GPU text extraction primitives for PDF, DOCX and PPTX.
  Extracted sections retain page, paragraph or slide locators for later citations.
- Empty documents and PDFs without a usable text layer are rejected as
  `no_text_layer`; encrypted PDFs are rejected as `password_protected`.
- DOC and PPT are intentionally rejected as `legacy_conversion_required` until a
  separate LibreOffice conversion container and malware scan gate are accepted.
- Fresh acceptance: full container suite ran 94 tests successfully; all three
  Compose services were healthy; GPU 0, 1 and 3 remained at 18 MiB and GPU 2
  remained at 17,370 MiB.
- Rebuilt the final accepted image as
  `sha256:c440d2f2b23cfd64fb80e41ab943ed30539f98d0fe129d5c334bf195d4004b0d`.
  `.dockerignore` now excludes `mergeKit_beta/runtime/`, preventing PostgreSQL
  and Redis bind-mount data from entering the build context or blocking builds
  due to container-owned permissions.

## Batch 4: Scan-First File Processing

- Verified local ClamAV image provenance against Docker Hub's official linux/amd64
  config digest before use. The ClamAV service has no host port mapping and is
  reachable only over the Compose network.
- Verified clamd health, clean scan acceptance and canonical EICAR rejection
  across `mergekit-beta -> model-gateway-clamav` using the INSTREAM protocol.
- Added scan-first source processing: a research source remains unusable until it
  is `ready`; clean PDF/DOCX/PPTX sources produce TTL-bound chunks and remove the
  original. Infected and parsing-rejected sources remove their originals.
- A temporary DOCX acceptance source was processed through the live scanner and
  parser, verified at paragraph locator `1`, then its Key, file row, chunk and
  temporary file were deleted.
- Final accepted image:
  `sha256:f2beab491be4f648957575d34d21fa1db500410471fe80e3c966c0f15a0fdcdf`.
  Rollback tag: `mergekit-beta:pre-research-stage1-scan-20260713_173413`.
- Fresh acceptance: all four profile services healthy; full container suite ran
  100 tests successfully; GPU 0, 1 and 3 remained at 18 MiB and GPU 2 remained
  at 17,370 MiB. No Mihomo proxy listener remained after validation.

## Batch 5: Durable File Worker

- Added `model-gateway-research-worker`, a no-GPU Redis Streams consumer for
  uploaded source files. It requires the internal Worker Token, consumes only
  file IDs and runs scan-first processing; it does not start Flask task workers
  or model inference.
- User-facing upload now returns `received`; `GET /api/model-gateway/files/<id>`
  provides owner-scoped status polling. Research jobs reject files until `ready`.
- Real HTTP acceptance created a temporary API Key, uploaded a valid DOCX,
  observed `received -> ready`, and removed all temporary Key, file and chunk rows.
- Restart acceptance stopped only the file Worker, observed Redis stream lag `1`,
  recreated the Worker, then observed the same source reach `ready`; all temporary
  records were removed.
- Final accepted image:
  `sha256:f94977976c430b1f680d0951e50e721dda342691de5bed0768f73a28fd7d0c79`.
  Rollback tag: `mergekit-beta:pre-research-stage1-file-worker-20260713_175237`.
- Fresh acceptance: full container suite ran 106 tests successfully; all profile
  services were healthy, Redis pending/lag were zero, Worker had no GPU device
  request, GPU 0/1/3 stayed at 18 MiB and GPU 2 stayed at 17,370 MiB.

## Batch 6: CPU BGE-M3 Encoder Runtime

- Downloaded only the local, ignored BGE-M3 model artifact and its ONNX external
  data from fixed Hugging Face revision
  `5617a9f61b028005a4858fdac845db406aefb181`. No model is downloaded at
  application startup.
- A first candidate exposed a deterministic C++ ABI conflict when the pip ONNX
  wheel loaded before SQLite. It was rolled back before acceptance. The final
  Worker-only fix preloads Conda's compatible `libstdc++.so.6`; the main
  application and GPU-serving processes keep their existing runtime environment.
- Final accepted image:
  `sha256:64d04dbc86e04f557b50bfec233c416c425fa6f2bfc43c7c0c1d6068a9df7124`.
  Rollback tag: `mergekit-beta:pre-stage1-bge-abi-fixed-20260713_182816`.
- Fresh acceptance: five HTTP smoke endpoints passed; main imports passed;
  the full suite ran 108 tests successfully. The live no-GPU Worker imported
  `onnxruntime==1.20.1`, produced two `(1024,)` CPU vectors with unit norms and
  passed its startup preload guard. GPU 0/1/3 stayed at 18 MiB and GPU 2 stayed
  at its external 17,370 MiB allocation. No proxy listener remained.

## Batch 7: Citation Retrieval Core

- Added owner-scoped hybrid retrieval with temporary FAISS inner-product search
  plus a bounded lexical score. It accepts only an API Key ID, explicit file
  IDs and unexpired `ResearchChunk` rows, then returns the source file, chunk
  and original page/paragraph/slide locator needed for later answer citations.
- The retrieval core has no endpoint or Worker registration yet. It retains no
  vector file or document text beyond the existing TTL-bound chunks, and does
  not start a language model.
- Fresh acceptance: all 109 tests passed, health/readiness passed and all GPUs
  retained the Batch 6 allocation snapshot.

## Batch 8: Research Execution Lifecycle Boundary

- Verified that vLLM is loopback-only inside the main application container.
  A future research executor will be a runtime-owner background thread there;
  the independent scan/file Worker remains no-GPU and never calls a model.
- Added an idempotent failure terminal transition which honors concurrent user
  cancellation and clears the Worker lease. It is the only permitted failure
  path for executor-side citation and upstream errors.
- Fresh acceptance: all 110 tests passed, health/readiness passed, and GPU
  allocation remained unchanged. No model process was started.

## Batch 9: Real 7B Fusion Smoke

- Submitted standard `linear` bfloat16 fusion task `35214914` using the two
  locally mounted, architecture-compatible Qwen2 7B text models. It completed
  successfully and registered
  `research_gateway_smoke_20260713_210042` at
  `merges/35214914/output`.
- The result contains four safetensors shards (about 15.23 GB total), model
  index, config and tokenizer files. It remains stopped as a serving model;
  no vLLM process was started in this batch.
- GPU 2's external vLLM allocation remained unchanged. The main Flask process
  retained an approximately 298 MiB PyTorch CUDA context on GPU 0 after the
  completed merge; no merge subprocess remained. Later vLLM acceptance must
  select GPU 1 or 3 after a fresh preflight.

## Batch 10: Citation Contract

- Added the prompt and output citation contract for research execution. Each
  retrieved evidence chunk is assigned a stable `[S<number>]`; when citations
  are required, missing or out-of-range references are terminal validation
  errors rather than user-visible research results.
- This batch does not start a model service or research execution thread. It
  prepares the tested contract for the next main-container Redis consumer.
- Fresh acceptance: all 111 tests passed, health/readiness passed, and GPU 2
  remained isolated from the fusion smoke task.

## Batch 11: Real vLLM Gateway Smoke And Spawn Fix

- Found and fixed two independent serving blockers. A broad vLLM `PYTHONPATH`
  entry had allowed `app/model_gateway/queue.py` to shadow the stdlib `queue`;
  the replacement is a hooks-only `sitecustomize.py` directory. This keeps the
  Transformers compatibility patch active in vLLM spawn children without
  exposing application modules on the import path.
- The newly fused task `35214914` has valid weights but copied an invalid
  tokenizer from one parent; it is explicitly not a serving candidate. A
  separate previously validated Qwen2 7B merged model was used for the
  runtime smoke instead.
- Manually started the verified model only on GPU 1, called `/v1/chat/completions`
  with a temporary allowlisted API Key, received HTTP 200 and vLLM usage
  `prompt=28`, `completion=6`, `total=34`. The temporary Key was revoked and
  the service was manually stopped afterward. GPU 1 returned to 18 MiB; GPU 2
  was unchanged.
- Fresh acceptance: all 112 tests passed, health/readiness passed and no model
  service remains running.

## Batch 12: Merged Tokenizer Repair Gate

- Added a post-merge tokenizer gate. A valid output tokenizer is preserved; an
  invalid one may be repaired only from a parent whose `vocab_size`, BOS/EOS
  and PAD configuration exactly matches the output. Otherwise the merge is not
  accepted as a serving candidate.
- Applied the same gate to task `35214914`: the invalid inherited tokenizer was
  backed up under `tokenizer_repair_backup/`, then replaced from the compatible
  valid Qwen parent. Weight shards were not modified.
- The repaired, newly fused 7B model was manually started on GPU 1 and returned
  HTTP 200 with the expected response and `34` vLLM tokens. Its temporary API
  Key was revoked and the service stopped; GPU 1 returned to 18 MiB and GPU 2
  was unchanged.
- Fresh acceptance: all 113 tests passed, health/readiness passed and no model
  service remains running.

## Batch 13: End-To-End Text Research Execution

- Added the main-container Redis research consumer. It claims durable jobs,
  retrieves API-Key-owned evidence with BGE-M3/FAISS, calls only the manually
  started loopback vLLM service, validates `[S<number>]` citations, writes an
  atomic TTL result and commits via the existing lease-aware lifecycle.
- A real DOCX source completed `received -> ready`; a real research question
  completed through the repaired fused 7B model. The result correctly answered
  `42 percent`, cited `[S1]`, and mapped the citation to paragraph `1`.
- The job status API now returns an owner-scoped result summary with answer,
  citation numbers and file/locator references, without returning full retrieval
  chunk text.
- The temporary API Key was revoked; temporary source, chunks, job, payload,
  result and local DOCX were removed; vLLM was manually stopped. Redis reported
  `pending=0` and `lag=0`; GPU 1 returned to 18 MiB and GPU 2 was unchanged.
- Fresh acceptance: all 114 tests passed, health/readiness passed after a full
  main-container restart.

## Batch 14: User Workspace Result Lifecycle

- Closed the user-visible research loop in `/research`: uploaded sources now
  refresh their processing status, and a research job polls only while it is
  non-terminal. The workspace blocks submission until every selected source is
  `ready`.
- Completed jobs render the answer, `[S<n>]` citation numbers and source
  page/paragraph/slide locators. Failed, expired and canceled jobs stop polling
  and show a terminal error code. All user-controlled text is rendered with DOM
  text nodes rather than HTML injection.
- Added a portal behavior test and an API contract test confirming that the
  completed-job endpoint exposes locators but not retrieved source text.
- Fresh local verification: targeted portal/API suite passed (4 tests),
  `node --check static/model_gateway/research.js`, live `/research` and static
  script HTTP checks, and `git diff --check` passed. No service restart, model
  start or GPU work was required. GPU 0/1/3 remained at 18 MiB; GPU 2 retained
  the external 17,370 MiB workload.
- Full follow-up gate: Compose configuration, container health, `/healthz`,
  `/readyz`, `/api/models`, `/api/testset/list`, `/api/history` and `/research`
  all passed. Container imports passed and the complete suite ran 116 tests
  successfully. The startup log still reports the pre-existing missing Flask
  migration directory and uses its documented `db.create_all()` fallback; this
  did not originate in Batch 14 and is tracked as a production migration
  hardening item before public-edge release.

## Batch 15: Cancellation And Output-Format Contract

- Added the user-facing `取消当前研究` control. It is visible only for a current,
  cancelable task and reuses the owner-scoped cancellation endpoint. Queued
  cancellation becomes terminal immediately; running cancellation stays in
  `cancel_requested` until the existing Worker lifecycle confirms it.
- Corrected a contract gap: `output_format` was stored but not used by the
  executor. Markdown preserves the existing citation contract. JSON now requires
  `{ "answer": "... [S1]", "citations": [1] }`; invalid JSON, non-numeric
  citation declarations and declaration/inline mismatches fail before a result
  is written.
- Recreated only `mergekit-beta` after verifying Redis research `pending=0`,
  `lag=0`, no queued/running research jobs and all Gateway model services
  stopped. PostgreSQL, Redis, ClamAV and the no-GPU file Worker were not
  recreated.
- Fresh post-restart gate: Compose validation, five existing HTTP APIs plus
  `/research`, imports and the in-container JSON contract all passed. The full
  suite ran 119 tests successfully; `node --check` and `git diff --check`
  passed. GPU 0/1/3 remained 18 MiB; GPU 2 retained its external 17,370 MiB
  workload. Redis research remained `pending=0`, `lag=0`.

## Batch 16: Chat-First Three-Mode Workspace

- Added three local layouts to `/research`: `chat` is the first-visit default,
  `research` preserves the existing document/citation task flow, and `focus`
  hides the source sidebar and secondary controls for reading. The selected
  layout is the only new browser-local value (`mergeneticResearchWorkspace`);
  messages and research content are not persisted by this UI.
- Chat and focus mode now use the existing API-Key-authenticated
  `/v1/chat/completions` endpoint without source files. Research mode remains
  explicit: it requires ready sources and creates the existing durable research
  job. Completed cited research answers are projected into the in-memory chat
  timeline once per job.
- Added visible hover, press and keyboard-focus feedback for controls, source
  rows and citation surfaces. Mode transitions and new entries use GSAP only for
  `transform` and opacity, with a reduced-motion immediate path and no persistent
  animation.
- Fresh automated gate: Compose configuration, `/healthz`, `/readyz`,
  `/api/models`, `/api/testset/list`, `/api/history` and `/research` all passed.
  The full container suite ran 125 tests successfully; Node syntax, live portal
  markup/style checks and `git diff --check` passed. Redis research was
  `pending=0`, `lag=0`; GPU 0/1/3 stayed at 18 MiB and GPU 2 retained its
  external 17,370 MiB allocation. No model, fusion or vLLM process was started.
- Visual screenshot automation remains unaccepted: Firefox 152 headless starts,
  accepts `--screenshot` and can write `/tmp`, but produces no image artifact in
  this environment. Manual desktop/mobile visual inspection is therefore still
  required through the live portal before declaring visual polish accepted.
