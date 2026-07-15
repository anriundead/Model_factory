# Request Handle And Cancellation

## Scope

Provide users with a stable request handle and make cancellation stop the
matching vLLM inference, not merely change the gateway database state.

## Request Identity

The gateway-generated `ServingRequest.id` is the public request handle. Every
chat response includes it in `X-Request-Id`; asynchronous `202` responses also
keep `request_id` in their JSON body. The gateway forwards that value as
`X-Request-Id` to vLLM.

With vLLM 0.7.0 OpenAI chat serving, the corresponding engine request ID is
`chatcmpl-<gateway request id>`. This mapping is deterministic, so it needs no
new database column or migration.

## Private Abort Boundary

`app/model_gateway/vllm_entrypoint.py` patches the vLLM OpenAI app builder
before vLLM starts. The patch adds a loopback-only
`POST /internal/model-gateway/abort/<request_id>` route. It requires the
per-service internal API key and calls `await app.state.engine_client.abort()`.
The public vLLM API remains unchanged.

The Flask gateway invokes that private endpoint only after it confirms that the
authenticated API key owns the gateway request and the model service is still
running. A successful abort changes the gateway request directly to `canceled`.
Background upstream completion and error paths preserve that terminal state.

## Acceptance

1. A non-streaming response returns `X-Request-Id` equal to its persisted
   gateway request ID.
2. The vLLM command enables request-ID headers and the private abort route
   rejects calls without the internal key.
3. Cancelling a running request calls the expected loopback abort URL, returns
   success, and the final database state remains `canceled`.
4. A real 7B smoke test completes a normal request, then cancels a long request
   without leaving vLLM, Ray, temporary keys, or GPU allocation behind.
