# DELETE_MANIFEST.runtime - 20260703_171018

Status: GENERATED ONLY. Not executed.

Gate 1 baseline runtime failed before cleanup, so these files were not deleted.

## Candidate Rules

Allowed only after baseline runtime passes and user confirms:

- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `_debug_*.log`
- `logs/*.log`
- `logs/merge/*.log`
- `*.bak.20*`
- `app.db.exported`

Rollback: runtime artifacts are normally reproducible. If a future run needs preservation, copy to `/tmp/mergekit_cleanup_backup/<timestamp>/` before deleting.

## Candidate Paths

```text
mergeKit_beta/__pycache__/config.cpython-311.pyc
mergeKit_beta/__pycache__/config.cpython-312.pyc
mergeKit_beta/__pycache__/config.cpython-313.pyc
mergeKit_beta/__pycache__/merge_manager.cpython-311.pyc
mergeKit_beta/__pycache__/merge_manager.cpython-312.pyc
mergeKit_beta/_debug_c8d559.log
mergeKit_beta/app.db.exported
mergeKit_beta/app/__pycache__/__init__.cpython-311.pyc
mergeKit_beta/app/__pycache__/__init__.cpython-312.pyc
mergeKit_beta/app/__pycache__/__init__.cpython-313.pyc
mergeKit_beta/app/__pycache__/admin.cpython-311.pyc
mergeKit_beta/app/__pycache__/dataset_info.cpython-311.pyc
mergeKit_beta/app/__pycache__/db_read_layer.cpython-311.pyc
mergeKit_beta/app/__pycache__/db_read_layer.cpython-312.pyc
mergeKit_beta/app/__pycache__/extensions.cpython-311.pyc
mergeKit_beta/app/__pycache__/logging_config.cpython-311.pyc
mergeKit_beta/app/__pycache__/logging_config.cpython-312.pyc
mergeKit_beta/app/__pycache__/models.cpython-311.pyc
mergeKit_beta/app/__pycache__/routes.cpython-311.pyc
mergeKit_beta/app/__pycache__/routes.cpython-312.pyc
mergeKit_beta/app/__pycache__/services.cpython-311.pyc
mergeKit_beta/app/__pycache__/services.cpython-312.pyc
mergeKit_beta/app/__pycache__/state.cpython-311.pyc
mergeKit_beta/app/__pycache__/state.cpython-312.pyc
mergeKit_beta/app/repositories/__pycache__/__init__.cpython-311.pyc
mergeKit_beta/core/__pycache__/__init__.cpython-311.pyc
mergeKit_beta/core/__pycache__/__init__.cpython-312.pyc
mergeKit_beta/core/__pycache__/__init__.cpython-313.pyc
mergeKit_beta/core/__pycache__/gpu_lock.cpython-311.pyc
mergeKit_beta/core/__pycache__/gpu_topology.cpython-311.pyc
mergeKit_beta/core/__pycache__/gpu_topology.cpython-312.pyc
mergeKit_beta/core/__pycache__/path_utils.cpython-311.pyc
mergeKit_beta/core/__pycache__/path_utils.cpython-312.pyc
mergeKit_beta/core/__pycache__/path_utils.cpython-313.pyc
mergeKit_beta/core/__pycache__/process_manager.cpython-311.pyc
mergeKit_beta/core/__pycache__/process_manager.cpython-312.pyc
mergeKit_beta/evolution/__pycache__/__init__.cpython-311.pyc
mergeKit_beta/evolution/__pycache__/progress_io.cpython-311.pyc
mergeKit_beta/evolution/__pycache__/runner.cpython-311.pyc
mergeKit_beta/evolution/vendor/vlm_merge/__pycache__/run_vlm_search.cpython-311.pyc
mergeKit_beta/evolution/vendor/vlm_merge/eval/prompt_mmlu.yaml.bak.20260416
mergeKit_beta/evolution/vendor/vlm_merge/eval/prompt_mmlu.yaml.bak.20260416b
mergeKit_beta/evolution/vendor/vlm_merge/eval/prompt_mmlu.yaml.bak.20260416c
mergeKit_beta/evolution/vendor/vlm_merge/eval_final.py.bak.20260416
mergeKit_beta/evolution/vendor/vlm_merge/eval_final.py.bak.20260416b
mergeKit_beta/evolution/vendor/vlm_merge/run_vlm_search.py.bak.20260416
mergeKit_beta/evolution/vendor/vlm_merge/run_vlm_search.py.bak.20260416b
mergeKit_beta/evolution/vendor/vlm_merge/run_vlm_search.py.bak.20260416c
mergeKit_beta/logs/app.log
mergeKit_beta/logs/merge/merge_20260403.log
mergeKit_beta/logs/merge/merge_20260404.log
mergeKit_beta/logs/merge/merge_20260407.log
mergeKit_beta/logs/merge/merge_20260408.log
mergeKit_beta/logs/merge/merge_20260409.log
mergeKit_beta/logs/merge/merge_20260410.log
mergeKit_beta/logs/merge/merge_20260411.log
mergeKit_beta/logs/merge/merge_20260412.log
mergeKit_beta/logs/merge/merge_20260413.log
mergeKit_beta/logs/merge/merge_20260414.log
mergeKit_beta/logs/merge/merge_20260416.log
mergeKit_beta/logs/merge/merge_20260417.log
mergeKit_beta/logs/merge/merge_20260421.log
mergeKit_beta/logs/merge/merge_20260422.log
mergeKit_beta/logs/merge/merge_20260423.log
mergeKit_beta/logs/merge/merge_20260519.log
mergeKit_beta/logs/merge/merge_20260526.log
mergeKit_beta/logs/merge/merge_20260607.log
mergeKit_beta/logs/merge/merge_20260608.log
mergeKit_beta/logs/merge/merge_20260609.log
mergeKit_beta/tests/__pycache__/test_eval_limit_resolution.cpython-312.pyc
mergeKit_beta/tests/__pycache__/test_gpu_topology_pairs.cpython-312.pyc
mergeKit_beta/tests/__pycache__/test_hf_code_eval_env.cpython-312.pyc
mergeKit_beta/tests/__pycache__/test_limit_plan_smoke_cmds.cpython-312.pyc
```
