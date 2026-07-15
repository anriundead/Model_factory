# 进化融合契约（Runner）

**关联系统文档**：HTTP 接口见 [`docs/API.md`](../docs/API.md)；数据库与 `merges/<task_id>/` 双写见 [`docs/DATABASE.md`](../docs/DATABASE.md)。

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

**Python 依赖（mergenetic）**：`vllm==0.7.0` 需同环境中安装 **`sphinx==7.4.7`**（已写入仓库根 `environment.yml` 与 Docker 构建）。缺失时 Runner 在导入 `run_vlm_search` 阶段即可因 `ModuleNotFoundError: sphinx` 失败；详见 [`DEVELOPMENT.md`](../DEVELOPMENT.md)「进化融合：vLLM 与 Sphinx」。

## Ray 并行度与显存（TP=1 时的裁剪）

### 与旧逻辑的差异（并行裁剪与可对账字段）

- **旧逻辑**：你在 `metadata.json` 里写 `ray_num_gpus=4`，系统倾向于“就按 4 并行去跑”。最多只做很粗的“最低空闲显存”判断，**不保证**每个 worker 都会落在“真的够跑一次 merge+eval 峰值”的 GPU 上，因此容易出现 **OOM**（甚至同卡挤多个进程）。
- **当前逻辑（方案 B）**：`ray_num_gpus` 只是“希望并行用几张卡”。Runner 会在启动子进程前先看每张卡的空闲显存，用 **单次评测峰值门槛**（默认 18GiB，可配）挑出“可并行的卡”，把并行度裁剪成 **`ray_num_gpus_effective`**，必要时用 `CUDA_VISIBLE_DEVICES` **只暴露这几张卡**给 Ray；并把“实际用了几张卡/哪些卡/为什么”写回 `metadata.json` 便于对账。

### 如何核对实际用卡与请求用卡

看 `merges/<task_id>/metadata.json` 这三个字段：

- **`ray_num_gpus`**：你请求的并行卡数（意图）。
- **`ray_num_gpus_effective`**：Runner 最终允许的实际并行卡数（事实）。
- **`evolution_cuda_visible_devices`**：子进程实际可见的 GPU 子集（事实）。若此字段为空/缺失，通常表示“无需收窄可见卡”（全卡都够或仅 1 卡）。

**真·N 卡并行（设计意图）**：`tp_size=1` 时，mergenetic 侧通过 Ray 的 `Pool`/`Actor` 以 **`num_gpus=1` 每 worker** 调度，理想情况下 **N 个并行 eval = N 张逻辑 GPU 上各一个进程**。Runner 在子进程启动前用 `CUDA_VISIBLE_DEVICES` 收窄可见卡集，使 Ray 只会在**余量足够**的卡上起 worker。

**给其他 Agent 的摘要**：仅按「空闲 ≥ 12GiB」一类下限仍可能低估 **单次 eval 的峰值显存**（merge + vLLM 等），导致多张卡「看起来够」却 OOM；或余量不足的卡仍被 Ray 看见并调度。典型症状：子进程退出码 1，`subprocess_output.log` 中 `torch.OutOfMemoryError`，甚至出现**同一逻辑 GPU 上多个进程**争显存。

**实现**（`evolution/runner.py` 内 `_cap_ray_num_gpus_for_parallel_eval`，方案 B）：

- 条件：`tp_size == 1` 且 `ray_num_gpus > 1`（`max_evals>0` 时已强制串行，不经过此逻辑）。
- `parallel_mib = max(MERGEKIT_EVOLUTION_MIN_FREE_GB, MERGEKIT_EVOLUTION_PEAK_GIB_PER_WORKER)` 换算为 MiB（前者与 `MERGEKIT_EVAL_MIN_FREE_GB` 回退，默认 12GiB；**峰值默认 18GiB**，可用 `MERGEKIT_EVOLUTION_PEAK_GIB_PER_WORKER` 覆盖）。
- 用 `core.gpu_topology.query_gpus()` 统计 `mem_free_mib >= parallel_mib` 的卡，得 `eligible_parallel`；`effective = min(ray_num_gpus, len(eligible_parallel))`。
- 若**无任何卡**达到 `parallel_mib`：串行 **`ray_num_gpus=1`**，并设 **`CUDA_VISIBLE_DEVICES`** 为「仅满足 min_free 的卡中空闲最大」的一张（`ray_cap_reason=no_gpu_meets_peak_gib`）。
- 若存在达标卡但少于全部可见卡或 `effective` 小于请求：对子进程设置 **`CUDA_VISIBLE_DEVICES`** 为按空闲排序后的前 `effective` 张达标卡（`ray_cap_reason=peak_gib_parallel_subset`）。
- 若连 min_free 都无卡满足：降为 1、不设 CVD 子集（`no_gpu_meets_min_free`）。
- `metadata.json` 可写入：`ray_num_gpus_effective`、`evolution_cuda_visible_devices`、`ray_cap_reason`、`evolution_peak_gib_per_worker`、`evolution_parallel_min_free_mib`。

**性能**：四张卡空闲均 ≥ `parallel_mib` 且请求 `ray_num_gpus=4` 时，**仍为 4 路并行**，与仅 min_free 裁剪且全达标时一致；峰值门槛更严时并行度下降，换稳定性。**若仍观测到同卡多进程**，需再查 Ray 版本/调度与进程内 fork 行为，不单靠本裁剪保证。

## 进度文件（Runner 模式）

- **`merges/<task_id>/progress.json`**：单一真相来源，由 **`evolution/runner.py`** 写入（含 `current_step`、`percent`、`message` 等），供 `/api/status` 与前端使用。
- **失败时**：通过 **`evolution/progress_io.py`** 合并写入 **`status: error`**，保留最后已知的 `current_step`、`total_expected_steps`、`current_best`、`percent` 等，并写入 **`error_detail`**、**`failed_at`**；`message` 为面向用户的摘要。勿整文件覆盖为仅 `{status,message}`，以免前端丢步数/best。
- **`merges/<task_id>/progress_mergenetic_debug.json`**（可选）：当上述 env 生效时，由 `run_vlm_search.py` 写入 worker 局部 step / evaluating / 异常摘要，**仅供排障**，API 不读。
- **CLI 单独运行** `run_vlm_search.py` 且未设置 `MERGEKIT_RUNNER_OWNS_PROGRESS` 时：行为与旧版一致，仍直接写 `--progress-file` 指向的文件。

## 与 API / 前端的终态对齐

- 应用层 **`GET /api/status`**：当任务仍在 Worker 内存但磁盘 **`metadata.json` 已为 `error`** 时，响应强制为 **`error`** 且 **`is_active: false`**（与磁盘 `success` → `completed` 对称）。进化任务在 metadata 未及时更新时，可读 **`progress.json` 的 `status==error`** 作为终态兜底（详见 `DEVELOPMENT.md` §9）。
- **GPU 报错**：`CUDA error: unspecified launch failure` 等属驱动/硬件/负载面，与 **OOM**（`OutOfMemoryError`）不同；缓解见上文 Ray 裁剪与运维（释显存、减并行、查 Xid），非单靠本契约能消除。

## 产物（与历史行为一致）

- `merges/<task_id>/progress.json`、`bridge.log`、`vlm_search_results/`、`final_vlm`、命名输出目录、`output` 链接或副本、`metadata.json` 终态；另见上节 `progress_mergenetic_debug.json`。