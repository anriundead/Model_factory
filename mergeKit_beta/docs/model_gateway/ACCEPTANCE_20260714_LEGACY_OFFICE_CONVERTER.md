# Legacy Office Converter Acceptance (2026-07-14)

## Status

Implementation and no-GPU automated checks are complete. Deployment acceptance
is blocked because the isolated LibreOffice image could not complete its APT
package download through the currently available proxy path. No converter
container, upload conversion, model or GPU task was started.

## Proxy Gate Evidence

The initial credential-redacted probe found:

- shell `HTTP_PROXY` and `HTTPS_PROXY`: missing;
- shell `NO_PROXY`: missing;
- ignored root `.env` proxy keys: missing;
- active `clash` or `mihomo` process: missing.

An existing Mihomo binary and configuration were then independently recovered:

- binary: `/home/a/zhangqi/5090/mihomo-local/mihomo`;
- valid simplified config: `config-simple.yaml` using the compatible geodata
  directory `/home/a/.config/mihomo`;
- process: PID `2980437`, local HTTP proxy `127.0.0.1:18890`;
- proxy checks: USTC release metadata and GitHub returned HTTP 200.

The older full Mihomo configuration still references incompatible geodata and
was not started. The plan forbids a direct download fallback.

## Implemented Boundary

- `app/model_gateway/legacy_converter.py` provides a bounded Worker client for
  `.doc -> .pdf` and `.ppt -> .pptx`, validates token/header/format/magic,
  enforces a 50 MiB output cap and removes temporary output on every failure.
- `file_processor.py` now scans the original source, converts only legacy DOC
  or PPT, scans the converted output again, then reuses current parsing/chunk
  persistence. Converter unavailability is retryable; terminal conversion
  errors reject the source without a text-only fallback.
- `model_gateway_legacy_converter/` contains a standalone standard-library
  HTTP server and dedicated LibreOffice Dockerfile. Compose defines a private
  internal network, no host port, no GPU, no source/database/model volume,
  read-only filesystem, dropped capabilities, no-new-privileges, tmpfs work
  directory, 1 CPU, 1 GiB, 128 PID and 45-second limits.
- `docker compose config --quiet` and `docker compose --profile research
  config --quiet` both passed.

## Automated Gates

- Legacy client, server, parser, file processor and file Worker tests: 20
  passed.
- Full container suite: 150 passed in 9.655 seconds.
- `git diff --check`: passed.
- GPU snapshot after checks matched baseline exactly. GPU 0/1/3 remain at
  18 MiB without compute processes; GPU 2 retains only the external Python and
  VLLM processes at 13630 MiB total.

The suite emitted existing Swig deprecation and CUDA platform-detection
warnings only.

## Image Build Blocker

Three proxy-bound build attempts were made without creating an image:

1. USTC HTTPS failed before `ca-certificates` existed in the minimal Ubuntu
   base because the proxy certificate issuer was not trusted.
2. USTC HTTP bootstrap passed signature checks but returned intermittent proxy
   `502` for `jammy-updates`.
3. Tsinghua and then Aliyun HTTP mirrors passed metadata probes; APT finite
   retries still encountered proxy-upstream failures while fetching the 133 MiB
   LibreOffice dependency set.

The Dockerfile now uses the verified Aliyun signed HTTP bootstrap source and
finite APT retry/timeout settings. It does not disable package signature
verification or accept untrusted TLS certificates. The Compose build uses host
network only at build time so the local proxy can be reached; the service
runtime remains on the internal-only network.

## Unchanged Runtime Safety

No `docker compose up` for the converter, LibreOffice runtime container, GPU
process, model process, research job, upload or source conversion was invoked.

## Resume Gate

Before resuming deployment, select or repair a proxy upstream that can sustain
the LibreOffice package download. Then rerun only the proxy-bound image build,
verify its digest and isolation properties, configure the ignored dedicated
converter token, and perform the planned no-GPU DOC/PPT acceptance. Do not
directly install LibreOffice on the host or main application image.

The temporary Mihomo process started for this batch is stopped after the
automated checks because no download remains in progress. Its verified restart
inputs are the existing binary, the `config-simple.yaml` configuration and the
compatible `/home/a/.config/mihomo` geodata directory; it listens locally only
on port 18890 when active.
