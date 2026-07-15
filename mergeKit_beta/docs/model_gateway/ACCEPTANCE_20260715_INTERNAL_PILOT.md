# Stage 1 Internal Pilot Acceptance

Date: 2026-07-15

## Verified

- Real invited-Key calls completed through `/v1/chat/completions` and the
  document/web research APIs.
- A public PDF upload and a public web source reached `ready`, produced
  TTL-bound vectors, and completed a citation-validated research task.
- Research result scope matched the submitted source IDs. Chat and research
  token usage were recorded separately as `vllm_response` and
  `vllm_research_response`.
- The Qwen 7B service ran only on GPU 1. Its PID/PGID persisted, and the
  administrator stop endpoint returned GPU 1 to its idle baseline.
- The temporary Key was revoked. Sources remain available only until the
  configured 24-hour TTL; browser/session actions did not delete them.
- Gateway PostgreSQL was stamped at Alembic revision `fb3e39249ef0` after a
  clean-schema check. A temporary database verified upgrade, repeated upgrade,
  downgrade, and re-upgrade before production stamping.
- Seven Redis pending deliveries were audited as duplicates of one terminal
  canceled job and safely ACKed. Both Gateway stream groups now have zero
  pending messages.

## Final Gates

- `docker compose config --quiet`: passed.
- `/healthz` and `/readyz`: passed after container recreation.
- Container test suite: 173 tests passed.
- No vLLM or Ray process remained after acceptance. GPU 2 was not used.

## Follow-up

Stage 2 may add Qwen-VL image and selected PDF/PPT page understanding. It must
keep vLLM loopback-only, administrator-managed model lifecycle, TTL cleanup,
and the Stage 1 owner/citation boundaries.
