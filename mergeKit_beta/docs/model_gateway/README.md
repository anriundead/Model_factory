# Model Gateway

`model_gateway` is the isolated subsystem that publishes completed merged
models through the OpenAI-compatible `/v1/*` API. It is separate from the
model factory's repository, fusion, evaluation, and evolution workflows.

## Layout

| Responsibility | Source path |
| --- | --- |
| Gateway API, vLLM lifecycle, authentication, and ORM | `mergeKit_beta/app/model_gateway/` |
| Administrator portal | `mergeKit_beta/templates/model_gateway/console.html` |
| Portal styles and browser logic | `mergeKit_beta/static/model_gateway/` |
| Gateway-only tests | `mergeKit_beta/tests/model_gateway/` |
| vLLM runtime logs | `mergeKit_beta/logs/model_gateway/` |
| Architecture | `ARCHITECTURE.md` |
| Implementation history | `IMPLEMENTATION_HISTORY.md` |
| Real-model acceptance evidence | `ACCEPTANCE_20260712.md` |

## URL Boundaries

- Administrator portal: `/model-gateway`
- Administrator API: `/api/model-gateway/admin/*`
- User inference API: `/v1/models`, `/v1/chat/completions`, and `/v1/requests/*`

The former `/serving` and `/api/serving/admin/*` paths are intentionally
absent. `/v1/*` remains unchanged because it is the external user contract.

## Configuration

All gateway configuration uses the `MERGEKIT_MODEL_GATEWAY_*` prefix. The
administrator token is read from the ignored repository-root `.env` file as
`MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN`; never place its plaintext value in code,
documentation, or logs.

## Persistence Boundary

The SQL table names retain their original `serving_*` names so this path-only
migration preserves existing service records, audit records, and revoked key
history. They are a database compatibility boundary, not a module, URL, or
filesystem namespace. A physical schema rename requires a separate data
migration and acceptance run.
