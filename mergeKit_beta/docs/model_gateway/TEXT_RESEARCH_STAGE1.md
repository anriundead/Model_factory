# Text Research Stage 1

## Runtime Boundary

- Gateway tables use the `model_gateway` SQLAlchemy bind. Production sets
  `MERGEKIT_MODEL_GATEWAY_DATABASE_URL` to PostgreSQL; empty keeps local SQLite compatibility.
- Sources are API-Key private. Uploads first enter `runtime/model_gateway/quarantine/`.
- Public URLs are only admitted after HTTP(S), DNS and public-address checks.

## Safe Defaults

| Setting | Default | Reason |
| --- | --- | --- |
| Source size | 50 MiB | Bounds conversion time and memory. |
| Sources per job | 10 | Supports comparison without unbounded context. |
| Source TTL | 24 hours | Matches the short-lived research privacy policy. |
| Worker concurrency | 1 | Host memory is shared with model-factory workloads. |
| GPU use | none | Parsing and embedding cannot consume serving/fusion GPUs. |

## Delivery Status

- Implemented: dedicated bind, source admission, upload quarantine, URL queue records,
  research-job creation, `/research` workspace, API Key revocation, idempotent research
  submission, durable job leases and cancellation/restart reconciliation rules.
- Implemented and runtime-verified: PostgreSQL Gateway bind, repeatable SQLite-to-PostgreSQL
  copy, Redis AOF profile, and the candidate image dependencies for document parsing,
  PostgreSQL and Redis.
- Implemented and runtime-verified: CPU text extraction with stable page,
  paragraph and slide locators for PDF/DOCX/PPTX. Legacy DOC/PPT use the private
  Apache POI parser rather than unsafe in-process binary parsing.
- Implemented and runtime-verified: a private ClamAV `clamd` service, standard
  INSTREAM client, and scan-first file processor. Clean sources become `ready`
  only after chunk persistence; infected, encrypted, damaged or textless sources
  are rejected and their originals removed.
- Implemented and runtime-verified: a dedicated no-GPU Redis Streams file Worker.
  Uploads enqueue only `file_id`; Worker restart recovery reconciles database
  `received` sources and never invokes model inference.
- Implemented and runtime-verified: local BGE-M3 dense encoding through the
  pinned ONNX export (`5617a9f61b028005a4858fdac845db406aefb181`). It uses
  `CPUExecutionProvider`, one ONNX intra/inter-op thread and 1024-token input
  bounds. The Worker alone preloads Conda's C++ runtime to keep the pip ONNX
  wheel isolated from the model-factory CUDA and Conda ABI stack.
- Implemented and runtime-verified: owner-scoped FAISS plus lexical retrieval core.
  It builds an in-memory index per call and returns only citation-ready chunks
  from the request Key and explicit file scope. The main-container research
  consumer uses it before loopback vLLM inference and requires valid `[S<n>]`
  citations before committing a result.
- Implemented and runtime-verified: the `/research` workspace polls short-lived
  source processing and research-job status, then renders only an answer,
  citation numbers and source locators. It never receives stored chunk text.
- Implemented and runtime-verified: Markdown and JSON research output contracts.
  JSON requires an object with `answer` and `citations`; citation numbers must
  match the `[S<n>]` markers in `answer` before a result is committed.
- Implemented and automated-test-verified: `/research` has a chat-first local
  workspace. Chat and focus modes call the existing API-Key-authenticated
  `/v1/chat/completions` endpoint without sources; research mode preserves the
  short-lived, citation-required document workflow. Conversation messages are
  held only in browser memory, while the selected `chat`/`research`/`focus`
  layout is the sole new browser-local preference.
- Implemented and runtime-verified: public URL ingestion, Key quotas and
  TTL-bound BGE-M3 vector sidecars. Query-time retrieval embeds only the
  question; a missing sidecar falls back to lexical retrieval.
- Implemented and runtime-verified: research prompts are context-budgeted and
  a loopback model connection failure pauses the job for manual recovery.
- Implemented and runtime-verified: research vLLM token usage is written in
  the same terminal database transition as a completed job using
  `vllm_research_response`; canceled and failed jobs create no usage record.
- Implemented and runtime-verified: vLLM child processes use CLI import mode,
  preventing the Flask restart-recovery hook from erasing their PID/PGID.
- Pending: Nginx public-edge profile and Stage 2 VLM document research.

## Research Execution Boundary

Published vLLM processes bind only to `127.0.0.1` inside the main
`mergekit-beta` container. The future research-job consumer must therefore run
as a single, runtime-owner background thread in that container, guarded by
`MERGEKIT_MODEL_GATEWAY_RUNTIME_PROCESS=1`. It must not be added to the
separate file Worker container or made reachable through the Compose network.

The lifecycle has an explicit `fail_research_job` transition: only the Worker
holding a running lease may fail a job; a concurrent user cancel wins and is
finalized as `canceled`. This is the required terminal path for missing
payloads, upstream failures and citation validation failures.

## Research Profile Preflight

`docker compose --profile research up` is intentionally not part of the normal
startup path. Before enabling it, set these ignored `.env` values:

```text
MERGEKIT_GATEWAY_POSTGRES_PASSWORD=<long-random-secret>
MERGEKIT_MODEL_GATEWAY_DATABASE_URL=postgresql+psycopg://model_gateway:<secret>@model-gateway-postgres:5432/model_gateway
MERGEKIT_MODEL_GATEWAY_REDIS_URL=redis://model-gateway-redis:6379/0
MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND=redis
```

The profile provisions PostgreSQL, Redis, ClamAV and the no-GPU file Worker.
Do not claim FAISS retrieval or model-backed research answers are enabled until
their dedicated acceptance gates pass. Rollback is
`docker compose --profile research stop`; the legacy SQLite Gateway database is
neither modified nor removed.
