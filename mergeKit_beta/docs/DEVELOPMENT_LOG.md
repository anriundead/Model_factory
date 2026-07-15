# 开发进度文档：VLM 评测管线接入 & LLM 评测修复

> 起止时间：2026-04-01 ~ 2026-04-02
> 涉及模块：`merge_manager.py`、`app/models.py`、`app/routes.py`、`app/services.py`、前端 `evaluation.js`/`styles.css`/`testsets.html`/`evaluation.html`
> 容器：`model_factory_mergekit-beta_1` (Docker Compose)

---

## 文档维护（2026-04-13）

- 新增 [`API.md`](API.md)（HTTP 接口层）、[`DATABASE.md`](DATABASE.md)（ORM、双写与磁盘规范）。
- [`README.md`](../README.md)、[`DEVELOPMENT.md`](../DEVELOPMENT.md)、[`evolution/contracts.md`](../evolution/contracts.md) 已增加与上述文档的交叉引用。

---

## 一、背景与目标

项目原有评测管线仅支持文本 LLM（通过 `lm_eval` 框架），但测试集仓库中已包含 VLM（视觉-语言）基准（如 `lmms-lab/MMBench`、`lmms-lab/MME`）。提交 VLM 基准评测时会被 `lm_eval` 错误匹配或直接报错。

**目标：**
1. 为 VLM 模型实现独立评测路径（集成 `lmms-eval`）
2. 前端增加 VLM/LLM 标注与兼容性提醒
3. 修复 LLM 评测中因 `trust_remote_code` 缺失和 accelerate 多卡分片导致的失败

---

## 二、已完成工作

### Phase 1: VLM 评测管线后端

#### 1.1 容器环境安装 `lmms-eval`
- 在 `mergenetic` conda 环境安装 `lmms-eval==0.7.1`
- 修复安装后的任务 YAML 缺失问题：手动创建多个 `_default_template_*_yaml` 空文件
- 修复 `lmms-eval` 任务 YAML 中 `token: True` 强制要求 HF token 的问题：批量改为 `token: False`

#### 1.2 VLM 模型检测 (`merge_manager.py`)
新增模块级函数：
- **`_model_is_vlm(model_path) -> bool`**：检测模型是否包含视觉塔
  - 检查 `config.json` 中的 `vision_config`、`image_token_id`
  - 检查 `model_type` 和 `architectures` 关键词（`_vl`, `llava`, `internvl`, `omni`, `multimodal`）
  - 目录名启发式匹配
- **`_infer_lmms_model_backend(model_path) -> str`**：推断 `lmms-eval` 的 `--model` 参数（`qwen2_5_vl` / `llava` / `internvl2`）
- **`_is_vlm_benchmark_hf_dataset(hf_dataset) -> bool`**：判断数据集是否为 VLM 基准
- **`_infer_lmms_task_name(hf_dataset, hf_subset) -> str`**：HF 数据集到 `lmms-eval` 任务名的映射

#### 1.3 VLM 评测执行函数 (`merge_manager.py`)
新增 **`run_lmms_eval_stream()`**：
- 构建 `lmms-eval` CLI 命令
- **MMBench 本地离线评测**：由于 `lmms-eval` 的 MMBench 任务依赖 OpenAI 做指标计算且出错时 exit=0，实现了基于 `datasets` + `transformers` 的本地推理路径
  - 正确处理 tensor `dtype`：仅对浮点张量做 `bfloat16` 转换，保留 `input_ids` 等整数类型
  - 动态注入图像 token（如 `<|image_pad|>`）到 prompt
  - 解析生成文本提取多选答案（A/B/C/D），计算 accuracy
- **MME limit 调整**：自动将奇数 limit 调整为偶数（MME 需成对聚合）
- 结果解析归一化为 `{ acc, f1, samples, time, context, per_task_acc }`
- 完整日志写入 `eval_full_output.log`

#### 1.4 评测路由三路分支 (`run_eval_only_task`)
修改 `run_eval_only_task()` 实现三路分支判断：
1. **VLM 基准 + VLM 模型** → `run_lmms_eval_stream()`
2. **VLM 基准 + LLM 模型** → 直接报错阻止
3. **LLM 基准（含 YAML 模板）** → 原有 `run_lm_eval_stream()`

#### 1.5 `lm_eval` 自动发现改进
- 过滤通用 subset 名（`default`/`train`/`test`/`validation` 等）避免误匹配（如 `lmms-lab/MMBench` 的 `default` subset 命中 `fld_logical_formula_default`）

### Phase 2: 前端 VLM/LLM 标注 & 兼容性提醒

#### 2.1 API 层 (`app/models.py`)
- `TestSet.to_dict()` 新增 `is_vlm_benchmark` 字段，基于 `hf_dataset` 前缀（`lmms-lab/`）和 `notes` 关键词判断

#### 2.2 前端 UI (`evaluation.js` / `styles.css` / `evaluation.html` / `testsets.html`)
- 测试集下拉列表和卡片视图增加 **VLM/LLM 徽章**（紫色/蓝色，带 fadeIn 动画）
- 选择 VLM 测试集时显示 **兼容性提示横幅**（`#vlm-compat-hint`，slideDown 动画）
- 提交前 **客户端兼容性检查**：VLM 基准 + LLM 模型 → warning toast 阻止提交；LLM 基准 + VLM 模型 → info toast 允许提交
- 新增 **Toast 通知系统**（`showToast(message, type)`），支持 info/warning/error 三种类型

#### 2.3 数据库修正
- `bench-mmbench` 的 `hf_subset` 从 `default` 改为 `en`
- `bench-mmbench` 的 `hf_split` 从 `test` 改为 `dev`（dev split 有标注可计算 accuracy）

### Phase 3: LLM 评测管线修复（本次 P3）

#### 3.1 `trust_remote_code` 全局修复
**问题**：`qiaojin/PubMedQA` 评测失败，`lm_eval` 自动发现匹配到内置 `pubmedqa` 任务，该任务使用 `bigbio/pubmed_qa` 数据集（含自定义加载代码），但 `trust_remote_code` 未传递给数据集加载。

**根因**：
1. `run_lm_eval_stream` CLI 命令缺少 `--trust_remote_code` 全局标志（仅在 `model_args` 中设了，只对模型生效）
2. 任务验证阶段 `verify_tm.load_task_or_group()` 触发数据集加载，`datasets` 库抛出 `ValueError`，被 `except ValueError: raise` 重新抛出
3. 部署路径错误：`docker cp` 复制到 `/app/merge_manager.py`，但实际运行路径为 `/app/ServiceEndFiles/Workspaces/mergeKit_beta/merge_manager.py`

**修复**：
- CLI 命令两处（accelerate/单机）添加 `--trust_remote_code`
- 任务验证的 `except ValueError` 增加 `trust_remote_code` 关键词检测，匹配时跳过验证
- 模块顶部设置 `os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")`
- 修正部署路径

**影响范围**：所有使用含自定义加载代码的 HF 数据集的评测任务，已确认影响 `pubmedqa`（`bigbio/pubmed_qa`）和 `winogrande`（`allenai/winogrande`）。

#### 3.2 accelerate 多卡空分片修复
**问题**：`--limit 3` 配合 4 GPU accelerate 时，部分 GPU 分配到 0 个样本，报 `task.build_requests() did not find any docs!`

**修复**：在 `use_accelerate` 判断中增加 limit 检查——当 `limit` 为绝对数且小于 GPU 数时，回退到单卡模式。

---

## 三、修改文件清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `merge_manager.py` | 新增 + 修改 | +723 行：VLM 检测、`run_lmms_eval_stream`、三路分支、trust_remote_code 修复、accelerate 小 limit 回退 |
| `app/models.py` | 修改 | `to_dict()` 新增 `is_vlm_benchmark` 字段 |
| `app/routes.py` | 修改 | testset 列表 API 透传 VLM 标记 |
| `app/services.py` | 修改 | 评测任务推断逻辑增强 |
| `static/evaluation.js` | 修改 | VLM/LLM 徽章、兼容性 toast、hint 横幅 |
| `static/styles.css` | 新增 | `.testset-type-badge`、`.vlm-compat-hint`、`.toast-container` 样式 |
| `templates/evaluation.html` | 修改 | 新增 `#vlm-compat-hint` div |
| `templates/testsets.html` | 修改 | 卡片视图中增加 VLM/LLM 徽章 |
| SQLite `app.db` | 数据修正 | `bench-mmbench` subset/split 调整 |

---

## 四、已知问题与待办

### 4.1 数据集源不一致（低优先级）
`lm_eval` 自动发现将用户指定的 HF 数据集映射到内置任务，但内置任务可能使用不同的底层数据集。例如：
- 用户选 `qiaojin/PubMedQA` → 内置任务用 `bigbio/pubmed_qa`
- 用户选 `allenai/ai2_arc` → 内置任务可能用相同数据集

这不是 bug（内置任务有完整的 prompt/metric 定义），但用户可能困惑为什么评测的样本数与预期不一致。

**建议**：在 metadata 中记录实际使用的 `lm_eval` 任务名和底层数据集，前端可展示。

### 4.2 `lmms-eval` 仅支持部分 VLM 后端
当前 `_infer_lmms_model_backend` 硬编码了 `qwen2_5_vl`/`llava`/`internvl2` 三种后端。新 VLM 架构需手动添加映射。

**建议**：维护可扩展的配置文件或从模型 config 自动推断。

### 4.3 MMBench 本地评测精度
本地离线评测路径使用简单 exact-match 抽取多选答案，对于复杂生成（模型输出非标准格式）可能漏判。

**建议**：引入更鲁棒的答案抽取正则或 few-shot prompt。

---

## 五、验证记录

| 日期 | 任务 ID | 模型 | 数据集 | 结果 |
|------|---------|------|--------|------|
| 04-01 | `0d15fa6f` | Qwen2.5-VL-7B-Instruct | MMBench (en) | success, acc=0.0 (2 samples smoke test) |
| 04-02 | `d08f4075` | Qwen2.5-7B-Instruct | PubMedQA (pqa_labeled) | **error**: trust_remote_code |
| 04-02 | `bc7c80c6` | Qwen2.5-7B-Instruct | PubMedQA (pqa_labeled) | **error**: accelerate 空分片 |
| 04-02 | `bd734689` | Qwen2.5-7B-Instruct | PubMedQA (pqa_labeled) | **success**, acc=100% (3 samples) |

---

## 六、部署注意事项

1. **部署路径**：容器内实际代码路径为 `/app/ServiceEndFiles/Workspaces/mergeKit_beta/`，不是 `/app/`。`docker cp` 时注意目标路径。
2. **缓存清理**：部署后需删除 `__pycache__`（`find /app -name '__pycache__' -exec rm -rf {} +`）并重启容器。
3. **`lmms-eval` 环境**：已在容器 `mergenetic` 环境安装，若重建镜像需在 Dockerfile 中添加。

---

## 七、进化融合 text + vLLM TP=2 + Ray 稳定性修复（2026-04-09）

> 涉及模块：`evolution/vendor/vlm_merge/run_vlm_search.py`、`config.py`、`start_app.sh`、仓库根 `docker-compose.yml`（注释与可选 env）

### 背景

在 Docker + Ray 多 worker 下，同一 PoolActor 进程内**第二轮及以后**再次构造 vLLM `LLM()`（TP>1）时，PyTorch c10d/Gloo 的 TCPStore 可能连向容器 bridge IP（如 `172.x`），导致长时间超时，与仅设置 `MASTER_ADDR`/`MASTER_PORT` 不足以覆盖 vLLM 内部 `new_group` 路径有关。

### 实现要点

1. **子进程隔离（默认）**
   - `tensor_parallel_size > 1` 且 `MERGEKIT_VLLM_TP_SUBPROCESS` 未显式关闭时，每轮 MMLU vLLM 评测通过 `subprocess` 调用同一脚本 `--vllm-tp-eval-worker <job.json>`，子进程内执行 `_llm_text_eval_mmlu_vllm_in_process`。
   - 超时由 `MERGEKIT_VLLM_SUBPROCESS_TIMEOUT_S` 控制（默认 1200s）；失败仍可走既有 `_eval_with_fallback`（TCPStore / DistNetworkError 等关键字）降级 transformers。

2. **多机安全：单机 loopback / hosts 改为显式开关**
   - `ray.init(..., _node_ip_address=127.0.0.1)` **仅**在 `MERGEKIT_RAY_SINGLE_NODE_LOOPBACK=1` 或已设 `MERGEKIT_RAY_NODE_IP_ADDRESS` 时生效，不再因 TP>1 无条件绑定 127。
   - `start_app.sh` 改写 `/etc/hosts`：默认不执行；仅 `MERGEKIT_DOCKER_HOSTS_LOOPBACK_FIX=1` 或（`MERGEKIT_RAY_SINGLE_NODE_LOOPBACK=1` 且未显式 `MERGEKIT_DOCKER_HOSTS_LOOPBACK_FIX=0`）时执行。

3. **配置白名单**
   - `Config.evolution_subprocess_env_patch` 增加透传：`MERGEKIT_VLLM_TP_SUBPROCESS`、`MERGEKIT_VLLM_SUBPROCESS_TIMEOUT_S`、`MERGEKIT_RAY_SINGLE_NODE_LOOPBACK`、`MERGEKIT_RAY_POOL_RUNTIME_ENV` 等（及既有 Ray/vLLM 相关键）。

4. **可选 Ray Pool `runtime_env`**
   - `MERGEKIT_RAY_POOL_RUNTIME_ENV=1` 时为 pymoo Ray `Pool` 注入 `GLOO_SOCKET_IFNAME=lo`，作补充手段，不替代子进程默认。

### 运维回滚（摘要）

| 手段 | 作用 |
|------|------|
| `MERGEKIT_VLLM_TP_SUBPROCESS=0` | 回滚进程内 vLLM TP（有已知第二轮风险） |
| 去掉单机 loopback / hosts 相关 env | 恢复多机友好默认 |
| Git revert 对应提交 | 按阶段回滚代码 |

### 文档

- 运维与环境清单：`DEVELOPMENT.md`（Docker 环境表 + 使用注意事项 §1）
- 入口说明：`README.md`「进化融合使用注意」
