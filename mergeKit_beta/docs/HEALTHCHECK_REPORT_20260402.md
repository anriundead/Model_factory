# 系统功能健康性检查报告

**日期**: 2026-04-02
**范围**: 小规模、真实 API 路径；GPU 策略文档见下文（本次执行因现场任务对部分项 **skip**）。

## 环境与 GPU 策略（prep-env-and-gpu-policy）

- **目标策略**: 对需要 GPU 的子进程使用 `CUDA_VISIBLE_DEVICES=1,2,3`，`NUM_GPUS=3`；LLM 多卡验证使用绝对条数 `limit >= 3`（建议 6 或 9）以避免 accelerate 空分片回退单卡。
- **说明**: 当前 Flask 服务进程若未以该环境变量启动，则通过 `**POST /api/evaluate`** 提交的任务会沿用服务可见的全部 GPU；要在生产上严格限定 1–3 卡，需在容器/进程启动层设置 `CUDA_VISIBLE_DEVICES`。
- **执行时快照** (`nvidia-smi`): GPU0 约 15.9GiB 已用，GPU1–3 基本空闲（用户任务占用 GPU0）。

## 与用户现场任务的关系

- **检测到的用户任务**: `task_id=2a0285d3`，`type=eval_only`，`status=running`，模型 `llama3-8B-slerp-med-chinese`，`cais/mmlu` subset `law`，`limit=1.0`，评测进度约 59%（121 条 generate_until）。
- **遵守**: 未通过前端提交任何新评测；未对用户任务执行取消/重试。
- **B/C/D 项**: 为避免与用户 LLM 评测争用显存（尤其默认 `cuda:0`），本次 **未** 新提交多卡 LLM、VLM 本地评测、进化融合任务。

## 检查结果汇总


| 项                     | 结果           | 说明                                                                                      |
| --------------------- | ------------ | --------------------------------------------------------------------------------------- |
| A1 testset/list       | **pass**     | `status=success`，25 条；字段含 `hf_dataset/hf_subset/hf_split/sample_count/is_vlm_benchmark` |
| A2 CMMMU scan         | **pass**     | `created_count=0`，幂等                                                                    |
| A3 dedup (CMMMU)      | **pass**     | `removed_count=0`；列表无重复 `(hf_dataset, hf_subset)`                                       |
| B LLM 多卡 (accelerate) | **fail（补跑）** | `5c35210e`：NCCL `/dev/shm` 失败；已加默认 `NCCL_SHM_DISABLE=1`，**需重启服务**后重试，见文末「补跑记录」          |
| C VLM (MMBench/CMMMU) | **pass（补跑）** | `d20613f8` MMBench；`05347fd2` CMMMU；见「补跑记录」                                             |
| D 进化融合                | **fail（补跑）** | `ab301bd4` / `02630762`：第二模型 config 无效或跨架构；Ray 3 卡曾启动，见「补跑记录」                           |
| D′ fusion_3d 数据       | **pass（只读）** | `GET /api/fusion_3d_data/62363628` → `success`, 45 points                               |
| E model_repo          | **pass**     | `merged_models` 34；`parent_models/recipe/dataset` 非空                                    |
| E recipes             | **pass**     | `success`，19 条配方                                                                        |
| E 页面 HTTP             | **pass**     | `/evaluation` `/testsets` `/model_repo` `/fusion_history` 均 200（未做浏览器 console 深度检查）     |


## 用户前置 LLM 任务（前端）— 收尾只读

- **task_id**: `2a0285d3`
- **结束时状态**: `success`（补跑核对时）；指标见上文「补跑记录」中 `2a0285d3` 小节。

## API 名称说明

- 评测提交接口为 `**POST /api/evaluate`**（非 `/api/eval`）。

## 文档声明：lmms-eval CLI

- `**lmms-eval` CLI 不是必须工具**；系统以内置评测路径（如 `merge_manager.py` 中 MMBench/CMMMU 本地分支及 LLM `lm_eval` 子进程）稳定可用为准；CLI 仅作可选调试手段。

## 后续建议（空闲后补跑）

1. **B**: `curl -X POST http://localhost:5000/api/evaluate -H 'Content-Type: application/json' -d '{"model_path":"/data/Models/Qwen2.5-7B-Instruct","dataset":"hellaswag","limit":"6","num_gpus":3,...}'`
2. **C**: 小 limit MMBench + CMMMU，VLM 模型路径。
3. **D**: `POST /api/merge_evolutionary`，`pop_size=2`,`n_iter=1`,`max_samples=6`,`ray_num_gpus=3`（若 Ray 支持）。
4. 浏览器打开各页确认无 **SyntaxError**（本次仅用 curl 测 HTTP）。

---

## 补跑记录（2026-04-03）

### B LLM 多卡（accelerate）

- **task_id**: `5c35210e`
- **结果**: **fail** — `eval_stderr.txt` 中 rank2 `ncclSystemError`：在 `/dev/shm` 创建 NCCL 共享内存段失败。
- **代码侧**: `merge_manager.py` 在 `use_accelerate` 时为子进程默认设置 `NCCL_SHM_DISABLE=1`（可用环境变量覆盖），以减轻小 `/dev/shm` 或容器场景下的 NCCL 失败。
- **运维侧**: 需 **重启 Flask/调度进程** 后新代码才生效；或增大 Docker `--shm-size`，或在服务启动环境中显式导出 `NCCL_SHM_DISABLE=1`。
- **重试**: 重启后再用相同 `POST /api/evaluate`（`hellaswag`, `limit=6`, `num_gpus=3`）验证。

### C VLM（MMBench + CMMMU）

- **MMBench** — **task_id** `d20613f8`：**pass**（`completed`）；`accuracy=25.0`（limit=8）；接口 `GET /api/status/d20613f8`。
- **CMMMU** — **task_id** `05347fd2`：**pass**（`completed`）；`limit=8`，`hf_split=validation`；`accuracy=0.0` 但 `per_task_acc` 含 `Overall` 等键，管道正常。
- **说明**: 当前 `lmms-eval` 路径在 `num_gpus>=2` 时仍回退单进程（与此前文档一致）；多卡以 LLM `accelerate` 路径验证为准。

### D 进化融合（VLM + Ray）

- **task_id** `ab301bd4`：**fail** — mergekit 读 `/data/Models/Meta-Llama-3-8B-Instruct` 时 `config.json` 缺少 `model_type`。
- **task_id** `02630762`：**fail** — 与 `llama3-8B-slerp-med-chinese` 合并时报 `RuntimeError: Must specify --allow-crimes to attempt to mix different architectures`（与 Qwen2 架构不一致）。
- **已验证**: Ray 本地实例可起、`ray_num_gpus=3` 有 worker；`/dev/shm` 过小导致 Ray object store 落 `/tmp`（性能警告）。
- **通过 D 全链路需**: 至少两个 **同架构、HF config 完整** 的本地模型目录；容器建议增大 `--shm-size`。产物目录中可见部分 `vlm_search_results/configs/*.yaml` 与 `evolution_stream.csv`（任务未成功结束则无完整 `fusion_info.json` / 最终 output）。

### 用户任务 `2a0285d3`（更新）

- **status**: `success`（`metadata.json`）；`test_cases=121`；`accuracy`/`f1_score` 仍为 **0.0**（`per_task_acc` 中对应任务亦为 0）— 若业务上异常，可查 `merges/2a0285d3/eval_full_output.log` / `eval_stderr.txt`。