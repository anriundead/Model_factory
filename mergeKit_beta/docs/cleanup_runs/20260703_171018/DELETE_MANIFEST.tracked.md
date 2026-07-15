# DELETE_MANIFEST.tracked - 20260703_171018

Status: GENERATED ONLY. Not executed.

Gate 1 baseline runtime failed before tracked cleanup, so none of these files were deleted or moved.

## Candidates

| Path | Git tracked | Category | Evidence | Risk | Rollback |
|---|---:|---|---|---|---|
| `mergeKit_beta/scripts/patch_lm_eval_transformers5.py` | yes | deprecated script | `DEVELOPMENT.md` says `lm_eval 0.4.11` with transformers 5.x no longer needs this patch; reference search found only that doc mention. | medium | `git restore mergeKit_beta/scripts/patch_lm_eval_transformers5.py` |
| `mergeKit_beta/eval_worker.py` | yes | deprecated script | `merge_manager.py` comments say YAML eval directly calls `lm_eval` and no longer uses `eval_worker.py`; reference search found no active import/subprocess reference to this file. | medium | `git restore mergeKit_beta/eval_worker.py` |
| `mergeKit_beta/investigate_4479.py` | yes | temporary debug | Reference search found no active references. | low | `git restore mergeKit_beta/investigate_4479.py` |
| `mergeKit_beta/verify_subset.py` | yes | temporary verification | Reference search found no active references. | low | `git restore mergeKit_beta/verify_subset.py` |
| `mergeKit_beta/verify_test.py` | yes | temporary verification | Reference search found no active references to this file; unrelated `/tmp/mergekit_verify_test_model_path_do_not_use` string appears in `scripts/verify_db_integration.py`. | low | `git restore mergeKit_beta/verify_test.py` |

## Explicit Keep

- `mergeKit_beta/app.py.legacy`: keep as read-only archive.
- `mergeKit_beta/scripts/run_vlm_search_bridge.py`: keep because `MERGEKIT_EVOLUTION_LEGACY_BRIDGE` still supports it.
- `mergeKit_beta/app.db`: keep.
- `mergeKit_beta/recipes/`: keep.
- `mergeKit_beta/testset_repo/yaml/`: keep.
- Core evaluation/evolution files: keep.
