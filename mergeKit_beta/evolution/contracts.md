# 进化融合契约（Runner）

## 进程模型

- **应用 Worker** 仅拉起 **一个** 子进程：`MERGENETIC_PYTHON` + 本 Runner（默认 `python -m evolution.runner`）。
- Runner 内部再拉起 **`run_vlm_search.py` 子进程**，工作目录为 **`VLM_SEARCH_DIR`**（见下表默认）。

## CLI

- `--task-id`（必填）：对应 `merges/<task_id>/metadata.json`。

## 环境变量（摘录）

| 变量 | 含义 |
|------|------|
| `VLM_SEARCH_DIR` | **可选**。若未设置或为空：使用仓内 **`evolution/vendor/vlm_merge`**（内置快照）。若设为路径：使用该目录中的 `run_vlm_search.py` / `eval_final.py`（外置调试或回滚）。 |
| `MERGENETIC_PYTHON` | 运行 Runner 与子进程的 Python（与 `config.Config` 一致） |
| `HF_DATASETS_CACHE` / `HF_ENDPOINT` | 由 Runner 传入子进程；`eval_final` 内数据集缓存优先用 `HF_DATASETS_CACHE` |
| `MERGEKIT_EVOLUTION_LEGACY_BRIDGE` | 见 `config.Config`：为 `1`/`true`/`yes` 时 Worker 改用 `scripts/run_vlm_search_bridge.py` 入口（薄包装） |

## 产物（与历史行为一致）

- `merges/<task_id>/progress.json`、`bridge.log`、`vlm_search_results/`、`final_vlm`、命名输出目录、`output` 链接或副本、`metadata.json` 终态。
