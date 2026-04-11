# 进化融合契约（Runner）

## 进程模型

- **应用 Worker** 仅拉起 **一个** 子进程：`MERGENETIC_PYTHON` + 本 Runner（默认 `python -m evolution.runner`）。
- Runner 内部再拉起 `**run_vlm_search.py` 子进程**，工作目录为 `**VLM_SEARCH_DIR`**（见下表默认）。

## CLI

- `--task-id`（必填）：对应 `merges/<task_id>/metadata.json`。

## 环境变量（摘录）


| 变量                                  | 含义                                                                                                                       |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `VLM_SEARCH_DIR`                    | **可选**。若未设置或为空：使用仓内 `**evolution/vendor/vlm_merge`**（内置快照）。若设为路径：使用该目录中的 `run_vlm_search.py` / `eval_final.py`（外置调试或回滚）。 |
| `MERGENETIC_PYTHON`                 | 运行 Runner 与子进程的 Python（与 `config.Config` 一致）                                                                             |
| `HF_DATASETS_CACHE` / `HF_ENDPOINT` | 由 Runner 传入子进程；`eval_final` 内数据集缓存优先用 `HF_DATASETS_CACHE`                                                                |
| `MERGEKIT_EVOLUTION_LEGACY_BRIDGE`  | 见 `config.Config`：为 `1`/`true`/`yes` 时 Worker 改用 `scripts/run_vlm_search_bridge.py` 入口（薄包装）                              |
| `MERGEKIT_RUNNER_OWNS_PROGRESS`     | Runner 在拉起 `run_vlm_search.py` 时设为 `1`。**仅子进程可见**：`progress.json` 由 Runner 根据 stdout 的 `[eval]` 等更新；子进程不得覆盖。 |

## Ray 并行度与显存（TP=1 时的裁剪）

**给其他 Agent 的摘要**：此前仅检查「是否存在至少一张卡」空闲 ≥ `MERGEKIT_EVOLUTION_MIN_FREE_GB`（与 `MERGEKIT_EVAL_MIN_FREE_GB` 回退，默认 12GiB）即允许 `tp_size=1` 多卡并行；但用户请求的 `ray_num_gpus` 仍可能为 4，Ray 会在**所有**可见 GPU 上调度 worker，导致**余量不足的卡**（如仅 ~10GiB 空闲）仍承担 merge+eval，引发 **`CUDA out of memory`**（典型症状：子进程退出码 1，`torch.OutOfMemoryError` 栈在 `subprocess_output.log`）。

**实现**（`evolution/runner.py` 内 `_cap_ray_num_gpus_for_parallel_eval`）：

- 条件：`tp_size == 1` 且 `ray_num_gpus > 1`（`max_evals>0` 时已强制串行，不经过此逻辑）。
- 用 `core.gpu_topology.query_gpus()` 与**同一阈值**统计「达标」GPU 数量 `len(eligible)`。
- `effective = min(ray_num_gpus, len(eligible))`；若 `effective < ray_num_gpus` 或存在未达标卡，对子进程设置 **`CUDA_VISIBLE_DEVICES`** 为按空闲排序后的前 `effective` 张**达标**物理卡索引，避免 Ray 看到余量不足的卡。
- 无达标卡时退化为 `ray_num_gpus=1` 并打警告。
- 将 `ray_num_gpus_effective`、`evolution_cuda_visible_devices`（若有）、`ray_cap_reason` 写入 `metadata.json` 便于对账。

**性能**：仅在「部分 GPU 未达空闲阈值」或「请求并行数大于达标卡数」时降低并行度；全卡达标且请求数不超过卡数时**行为与裁剪前一致**。并行度降低会使**单位时间内完成的 eval 数**下降，总墙钟可能变长，但换取**避免 OOM 整任务失败**；全卡充足时**不慢**。

## 进度文件（Runner 模式）

- **`merges/<task_id>/progress.json`**：单一真相来源，由 **`evolution/runner.py`** 写入（含 `current_step`、`percent`、`message` 等），供 `/api/status` 与前端使用。
- **`merges/<task_id>/progress_mergenetic_debug.json`**（可选）：当上述 env 生效时，由 `run_vlm_search.py` 写入 worker 局部 step / evaluating / 异常摘要，**仅供排障**，API 不读。
- **CLI 单独运行** `run_vlm_search.py` 且未设置 `MERGEKIT_RUNNER_OWNS_PROGRESS` 时：行为与旧版一致，仍直接写 `--progress-file` 指向的文件。

## 产物（与历史行为一致）

- `merges/<task_id>/progress.json`、`bridge.log`、`vlm_search_results/`、`final_vlm`、命名输出目录、`output` 链接或副本、`metadata.json` 终态；另见上节 `progress_mergenetic_debug.json`。