# Gateway Image Build And Rollback

## Scope

The Stage 1 research dependencies are verified in a candidate image before the
running `mergekit-beta:latest` tag is replaced. Build-time proxy values are
passed as Docker `ARG`s only and must not remain in the runtime environment.

## Acceptance Order

1. Build `mergekit-beta:research-stage1-candidate` from the repository root.
2. Verify required imports in a one-shot, no-GPU container.
3. Run the model-gateway unit suite from a candidate container.
4. Tag the candidate as `mergekit-beta:latest`, recreate only `mergekit-beta`,
   then run health, readiness and API smoke tests.

## Rollback

Before retagging, preserve the current image as a timestamped
`mergekit-beta:pre-research-stage1-<timestamp>` tag. If candidate verification
or runtime smoke fails, retag that preserved image as `mergekit-beta:latest`
and recreate only `mergekit-beta`. PostgreSQL and Redis are not started or
deleted by an image rollback.

## 2026-07-13 Record

- Preserved image: `mergekit-beta:pre-research-stage1-20260713_164357`.
- First accepted candidate image: `sha256:7f38e4a9bcdb6cb83485a56b903e6c8fdac7fe17b05c76e16aa5020b658fe6b8`.
- Final accepted image after the migration and parser changes:
  `sha256:c440d2f2b23cfd64fb80e41ab943ed30539f98d0fe129d5c334bf195d4004b0d`.
- A second pre-cutover tag, `mergekit-beta:pre-research-stage1-runtime-20260713_171211`,
  preserves the first candidate until ordinary Docker image retention cleanup.
- Scan-first processor image: `sha256:f2beab491be4f648957575d34d21fa1db500410471fe80e3c966c0f15a0fdcdf`.
  Rollback tag: `mergekit-beta:pre-research-stage1-scan-20260713_173413`.
- Durable file Worker image: `sha256:f94977976c430b1f680d0951e50e721dda342691de5bed0768f73a28fd7d0c79`.
  Rollback tag: `mergekit-beta:pre-research-stage1-file-worker-20260713_175237`.
- Build used a temporary host-loopback Mihomo proxy. The proxy process was stopped
  before runtime cutover, and the accepted image has no runtime proxy variables.

## BGE-M3 CPU Encoder - 2026-07-13

- An initial ONNX candidate was rejected and retained as
  `mergekit-beta:failed-stage1-bge-20260713_182203`: importing the pip ONNX
  wheel before SQLite selected the system C++ runtime, which lacks
  `CXXABI_1.3.15` required by the Conda ICU build. No Gateway data service was
  restarted during its rollback.
- The accepted image is
  `sha256:64d04dbc86e04f557b50bfec233c416c425fa6f2bfc43c7c0c1d6068a9df7124`.
  Rollback tag: `mergekit-beta:pre-stage1-bge-abi-fixed-20260713_182816`.
- `onnxruntime==1.20.1` adds no Torch, Transformers, NumPy or CUDA upgrade.
  Only `model-gateway-research-worker` sets
  `LD_PRELOAD=/opt/conda/envs/mergenetic/lib/libstdc++.so.6`; the main Flask,
  vLLM and model-factory processes do not inherit it.
