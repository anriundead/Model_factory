# Legacy Office Parser

## Purpose

The private `model-gateway-legacy-parser` service extracts text from admitted
legacy `.doc` and `.ppt` research sources. It replaces the unshipped
LibreOffice converter. It never creates PDF/PPTX intermediates and never
claims a DOC page number that the original format cannot reliably provide.

## Contract

| Source | Extractor | Citation locator | Reason |
|---|---|---|---|
| `.doc` | Apache POI HWPF | `paragraph N` | Old Word pagination depends on the rendering environment. |
| `.ppt` | Apache POI HSLF | `slide N` | Slide order is part of the source format. |

The Worker sends an already ClamAV-scanned file to `POST /parse` over the
internal `model-gateway-private` network. The parser returns JSON sections
only. The Worker validates locator kinds, chunks the text, then deletes the
original quarantine file. No raw source or parsed text is logged by the
service.

Apache POI is used directly rather than adding Apache Tika. Tika delegates
these Office formats to POI but does not improve the required DOC paragraph or
PPT slide citation contract, so a second parser layer would only add image and
patching surface.

## Safety Limits

| Setting | Default | Why |
|---|---:|---|
| `MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_TIMEOUT_SECONDS` | `45` | Kills a stuck child parser before a queue slot is held indefinitely. |
| `MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_MAX_MIB` | `50` | Matches the upload limit and bounds input/output payloads. |
| container CPU | `1.0` | Parsing is intentionally serialized; it must not compete with model inference. |
| container memory | `1 GiB` | The child JVM has `-Xmx384m`; the remaining headroom covers POI and the HTTP parent. |
| `/work` tmpfs | `768 MiB` | Holds at most one bounded source plus response, then is deleted after each request. |

The service has no GPU, published port, writable image layer, or access to
the default Compose network. It accepts only the Worker token. The parent
spawns one JVM child per request and force-kills it on timeout, so a malformed
Office file cannot leave a hung POI parser owning the queue slot.

## Rollback

Stop only this service with:

```bash
docker compose --profile research stop model-gateway-legacy-parser
```

Without the parser URL/token, legacy files return retryable
`legacy_parser_unavailable`; PDF, DOCX and PPTX processing continues
unchanged. Restoring the previous unshipped LibreOffice files is not required
for application recovery.
