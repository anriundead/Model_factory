# Stage 1 Internal Pilot Hardening

> Status: approved for implementation on 2026-07-15.
> Scope: invited-user text research only. It does not add VLM document vision,
> billing, a public Nginx edge, or a second worker framework.

## Goal

Make the existing research portal safe for a small internal pilot without
changing the model-factory merge, evaluation, or evolution paths. A Key must
not be able to exhaust parsing, retrieval, or model-serving capacity. Research
source content remains short-lived and is removed automatically.

## Decisions

| Control | Default | Reason |
| --- | ---: | --- |
| Chat requests | 20 per Key per minute | Enough for interactive use while bounding accidental loops. |
| Research submissions | 4 per Key per hour | Research work is comparatively expensive and asynchronous. |
| Active research jobs | 1 per Key | Prevents one Key from building an unbounded backlog. |
| Imported source bytes | 500 MiB per Key per UTC day | Leaves room for normal PDF comparison while bounding disk and parser work. |
| Source lifetime | 24 hours | Matches the agreed no-long-term-server-retention policy. |

PostgreSQL is the production source of truth for quota windows. Redis remains
the durable-delivery transport only. SQLite retains compatible behavior for
local development, but it is not the multi-process production authority.

## Web Source Boundary

A URL is a short-lived web research source, not a user-download feature. The
no-GPU Worker fetches an admitted public HTTP(S) URL, checks every redirect,
scans the retrieved representation and passes it into the existing chunk and
retrieval pipeline. Stage 1 accepts HTML and PDF only.

- HTML yields citation-ready title, visible paragraphs, table text and image
  alt/caption context. Script, style and hidden navigation text are excluded.
- Image pixels are not sent to a model in Stage 1. Visual interpretation of a
  web image is a Stage 2 Qwen-VL capability.
- The final canonical URL remains owner-scoped metadata and citations retain a
  web paragraph locator. The fetched body is deleted after successful parsing,
  like an uploaded original.
- Every initial URL and redirect target must be public HTTP(S). Loopback,
  private, Docker, link-local and metadata addresses are rejected before a
  connection is made. Response bytes remain bounded by the existing 50 MiB
  source limit.

## Boundaries

- Rate and quota checks apply after user API Key authentication and before a
  protected operation is accepted.
- Administrator management routes are not counted as user traffic.
- Canceling a research job releases its active-job slot when it becomes a
  terminal state. A paused offline-model job remains active until canceled or
  completed, so an offline service cannot accumulate hidden backlog.
- A rejected request returns HTTP `429`, a stable error code, a retry interval,
  and no source or model content.
- TTL cleanup deletes only data below `runtime/model_gateway/`; it never follows
  an arbitrary database path. Usage and audit metadata remain content-free.
- Closing, refreshing, switching, or clearing browser-local sessions never
  deletes server research data or stops a task. The 24-hour TTL prevents an
  accidental browser action from destroying a user's sources. Any future
  immediate server deletion must be a separately named, confirmed operation;
  the current local “remove” controls remain local-only.
- The existing no-GPU file Worker owns periodic cleanup. It may not start a
  model, use CUDA, or terminate model-factory processes.

## Delivery Batches

1. Web sources: secure HTML/PDF acquisition, scan-first processing, web
   locators and actual-public-page acceptance.
2. Quotas: durable window counters, configuration, API enforcement, portal
   feedback, and regression tests.
3. TTL cleanup: expired file/job/chunk cleanup, path containment, Worker timer,
   and regression tests.
4. Real acceptance: a stopped verified Qwen 7B service is started only after a
   fresh GPU snapshot confirms GPU 0 or 1 is free. A neutral public scientific
   PDF travels through the actual portal/API, scanner, file Worker, parser,
   BGE-M3 retrieval, Redis consumer, vLLM and citation validator.

## Acceptance Gates

- Baseline and post-change container suite use
  `/opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests`.
- `docker compose config --quiet`, `docker compose --profile research config
  --quiet`, and the five existing HTTP smoke endpoints pass.
- Unit tests are regression evidence only. Acceptance uses a real public web
  page plus a real PDF upload, real Worker delivery, real parsing and a real 7B
  vLLM response.
- GPU 2 is snapshotted before and after; its UUID, process list and memory use
  must remain unchanged. GPU 0 or 1 is stopped and rechecked after acceptance.
- Temporary Key, source, chunks, payload and result are removed. The temporary
  low TTL and low quota test settings are restored to the defaults above.

## Rollback

- Stop only the test model service through its administrator API; never kill an
  unmarked process.
- Revert the quota/cleanup commit, recreate only `mergekit-beta` and
  `model-gateway-research-worker`, then restore the saved Compose environment.
- If cleanup misbehaves, stop only `model-gateway-research-worker`; research
  ingestion pauses while the core model factory and existing vLLM services stay
  untouched.
- Do not run a destructive database rollback during acceptance. Retain quota
  metadata and clean only records created by the temporary Key.
