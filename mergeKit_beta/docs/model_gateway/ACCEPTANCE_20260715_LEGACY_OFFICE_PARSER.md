# Legacy Office Parser Acceptance Record

Date: 2026-07-15

## Change

The unshipped LibreOffice converter was replaced with a direct Apache POI
legacy parser. The service name is `model-gateway-legacy-parser`; its source
is `mergeKit_beta/model_gateway_legacy_parser/`. Its Worker client is
`app/model_gateway/legacy_parser.py`.

## Build And Unit Verification

- Python Worker/client contract tests pass in the running `mergekit-beta`
  image.
- Java locator contract tests run during the parser Docker image build.
- The image was built as `mergekit-legacy-office-parser:latest` without GPU
  access. The accepted image ID is
  `sha256:924e12c70520ca4b1e1eaedf22de791eeda1873c1dd8a12e06582a071ac96b49`
  (about 273 MB). Build-only Mihomo was stopped by the build shell on
  completion.
- No LibreOffice package or binary was added to the host, `mergekit-beta`, or
  research Worker images.

## Runtime Verification

- `docker compose --profile research config --quiet` passed.
- The parser container is healthy and only has the internal
  `model-gateway-private` network. It has no published port, uses a read-only
  root filesystem, and retains the Compose limits of one CPU, 1 GiB memory and
  128 PIDs.
- The Worker received `401` for a deliberately wrong parser token.
- A completion review added a missing-format guard. A valid token without
  `X-Source-Format` now returns the contract `400` response instead of an
  uncaught null error; the rebuilt image passed its Java tests and was rolled
  into the healthy parser container.
- Disposable Apache POI test assets were downloaded through a temporary
  Mihomo proxy, copied only to the Worker `/tmp`, parsed over the private
  network, then deleted. `47304.doc` yielded one valid `paragraph` section;
  a legacy `.ppt` fixture yielded three valid `slide` sections.
- `python -m unittest discover -s tests` in `mergekit-beta` passed all 145
  tests. `/healthz`, `/readyz`, `/api/models`, `/api/testset/list` and
  `/api/history` all returned successfully.
- GPU 0, 1 and 3 remain at 18 MiB; GPU 2 remains at 13630 MiB, unchanged from
  the pre-activation snapshot. The parser has no GPU access and the Worker
  reports GPU functionality unavailable.
- Temporary documents, runtime proxy, and parser test artifacts were removed.

## Deployment Repair

The running Worker had an internal token that was absent from the host
environment and `.env`, so a normal Compose recreation would have lost it.
The existing value was recovered without printing it and persisted to the
local `.env` with mode `0600`. A separate 256-bit parser token was generated
there at the same time. Neither value is recorded in this document or Git.

## Rollback

To stop only legacy parsing while leaving the rest of Model Factory running:

```bash
docker compose --profile research stop model-gateway-legacy-parser
```

Legacy DOC/PPT jobs then return retryable `legacy_parser_unavailable`; PDF,
DOCX and PPTX remain available. To restore the pre-parser Worker environment,
remove only the two `MERGEKIT_MODEL_GATEWAY_*TOKEN` lines added during this
batch from the local `.env`, then recreate the research Worker with an
operator-provided token.

All parser acceptance gates are complete.
