# DELETE_MANIFEST.tracked

Execution batch: Gate 3 tracked deprecated code cleanup.

| Path | Category | Git tracked | Evidence | Risk | Batch | Rollback |
| --- | --- | --- | --- | --- | --- | --- |
| `mergeKit_beta/investigate_4479.py` | 临时排障 | yes | Repository reference search found no active reference. | low | 3a | `git restore mergeKit_beta/investigate_4479.py` |
| `mergeKit_beta/verify_subset.py` | 临时验证 | yes | Repository reference search found no active reference. | low | 3a | `git restore mergeKit_beta/verify_subset.py` |
| `mergeKit_beta/verify_test.py` | 临时验证 | yes | Repository reference search found no active reference to this file. The unrelated string `/tmp/mergekit_verify_test_model_path_do_not_use` appears in `scripts/verify_db_integration.py`. | low | 3b | `git restore mergeKit_beta/verify_test.py` |
| `mergeKit_beta/scripts/patch_lm_eval_transformers5.py` | 废弃脚本 | yes | `DEVELOPMENT.md` states lm_eval 0.4.11 already supports transformers 5.x and no longer needs this patch. Script docstring also marks it deprecated. No active runtime reference found. | medium | 3c | `git restore mergeKit_beta/scripts/patch_lm_eval_transformers5.py` |
| `mergeKit_beta/eval_worker.py` | 废弃脚本 | yes | `merge_manager.py` states YAML eval directly calls `lm_eval` and no longer uses `eval_worker.py` to avoid recursion. No active import/subprocess reference found. | medium | 3d | `git restore mergeKit_beta/eval_worker.py` |

## Out Of Scope

- `mergeKit_beta/app.py.legacy`
- `mergeKit_beta/scripts/run_vlm_search_bridge.py`
- `mergeKit_beta/app.db`
- `mergeKit_beta/recipes/`
- `mergeKit_beta/testset_repo/yaml/`
- `mergeKit_beta/merge_manager.py`
- `mergeKit_beta/evolution/vendor/vlm_merge/run_vlm_search.py`
- `mergeKit_beta/config.py`
- `mergeKit_beta/app/services.py`
