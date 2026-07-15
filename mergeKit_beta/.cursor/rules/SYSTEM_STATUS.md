# Mergenetic Pro（mergeKit_beta）系统现状文档

> **本文档为 AI Agent 必读**：每次会话开始时应阅读本文件，以了解系统当前状态、已实现功能、待修复项与待开发项。
>
> 最后更新：2026-04-16

---

## 1. 系统概述

**Mergenetic Pro** 是一个基于 Flask 的 LLM/VLM 模型融合与评测平台，提供 Web UI 和 REST API。核心能力：

- **标准模型融合**（mergekit / mergenetic）
- **进化融合**（Evolutionary Ties-Dare）
- **配方管理**（Recipe：保存、复现、一键应用进化融合结果）
- **LLM 评测**（lm_eval，支持多卡 accelerate）
- **VLM 评测**（lmms-eval CLI + CMMMU/MMBench 本地评测）
- **模型仓库**（注册、浏览、标签、删除）
- **测试集仓库**（HuggingFace 数据集下载、管理、自定义 YAML 模板）
- **融合历史 & 3D 散点图**（进化搜索空间可视化）
- **排行榜**（按测试集聚合模型评测结果）

---

## 2. 运行环境

### 2.1 部署架构

```
宿主机（4x RTX 3090）
  └─ Docker (docker-compose v1)
       └─ model_factory_mergekit-beta_1
            ├─ 镜像: mergekit-beta:latest (nvidia/cuda:12.4.1 + conda)
            ├─ 端口: 5000 -> 5000
            ├─ PID 1: /opt/conda/envs/mergenetic/bin/python (Flask app)
            └─ 代码: bind mount 宿主 mergeKit_beta -> 容器内（改即生效）
```

### 2.2 容器内环境

| 项 | 值 |
|---|---|
| Python 环境 | `/opt/conda/envs/mergenetic/bin/python` (3.11) |
| MERGENETIC_PYTHON | `/opt/conda/envs/mergenetic/bin/python` |
| Flask 入口 | `start_app.sh` -> `app/__init__.py:create_app()` |
| 数据库 | SQLite: `/app/.../mergeKit_beta/app.db` |
| 模型目录 | `/data/Models`（只读挂载，宿主 `../Models`） |
| 融合产物 | `merges/`（bind mount） |
| 配方目录 | `recipes/` |
| 评测缓存 | `/data/eval_datasets`、`/data/hf_datasets` |

### 2.3 关键依赖版本

| 包 | 版本 |
|---|---|
| torch | 2.5.1 |
| transformers | 5.3.0 |
| lm_eval | 0.4.11 |
| mergenetic | 0.1.1 |
| mergekit | 0.1.4 |
| accelerate | 1.3.0 |
| datasets | 2.20.0 |
| Flask | 3.1.2 |
| SQLAlchemy | 2.0.46 |

### 2.4 GPU 状态

**compose**：`Model_factory/docker-compose.yml` 使用 `runtime: nvidia` 与 `NVIDIA_VISIBLE_DEVICES=all`（docker-compose v1 不读 `deploy.resources`，须显式 nvidia runtime）。宿主机需安装 **nvidia-container-toolkit**。

**验证**：容器内 `conda run -n mergenetic python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` 应输出 `True` 与 GPU 数量。

---

## 3. 数据规模（当前）

| 数据 | 数量 |
|---|---|
| 基座模型 | 19 |
| 融合模型（已注册） | 37 |
| 融合历史任务 | 35（进化融合） |
| 评测历史 | 44 |
| 配方 | 20 |
| 测试集 | 25 |
| 任务目录 (merges/) | 79 |
| DB: tasks | 79 行 |
| DB: models | 37 行 |
| DB: evaluation_results | 3 行 |
| DB: testsets | 25 行 |

---

## 4. 代码结构

```
mergeKit_beta/
├── app/                        # Flask 应用包
│   ├── __init__.py             # create_app()：DB 初始化、路由注册
│   ├── routes.py               # 所有 HTTP 路由（~50+ 个 endpoint）
│   ├── services.py             # 业务编排、Worker、任务队列
│   ├── models.py               # ORM 模型（Task, Model, Testset, EvaluationResult, EvolutionStep, Tag）
│   ├── extensions.py           # db, migrate, admin 实例
│   ├── db_read_layer.py        # DB 只读组装层（文件回退兼容）
│   └── repositories/           # 数据访问封装
│       └── __init__.py
├── merge_manager.py            # 融合/评测核心引擎（3500+ 行）
├── config.py                   # 配置类
├── core/
│   └── process_manager.py      # 子进程管理
├── evolution/                  # 进化融合
│   ├── runner.py               # 进化搜索入口
│   └── vendor/vlm_merge/       # VLM 进化融合适配
├── templates/                  # 前端 HTML 页面
│   ├── index.html              # 模型融合主页
│   ├── evaluation.html         # 评测页
│   ├── model_repo.html         # 模型仓库
│   ├── fusion_history.html     # 融合历史 + 3D
│   ├── testsets.html           # 测试集仓库
│   └── test_history.html       # 测试历史
├── static/                     # CSS/JS/图标
├── scripts/                    # 工具脚本（19 个）
├── tests/                      # 单元测试
├── merges/                     # 任务产物目录
├── recipes/                    # 融合配方 JSON
├── testset_repo/               # 测试集数据
├── Dockerfile                  # 容器镜像定义
├── start_app.sh                # 启动脚本
└── docker-compose.yml          # 在上级 Model_factory/ 目录
```

### 4.1 分层职责

```
HTTP 请求 -> routes.py -> services.py -> repositories / merge_manager / 子进程
                                       -> db_read_layer（只读）
```

详见 `.cursor/rules/ARCHITECTURE_BOUNDARIES.md`。

---

## 5. 前端页面与路由

| 页面 | URL | 功能 |
|---|---|---|
| 模型融合 | `/` | 选模型、配参数、提交标准/进化融合 |
| 模型测试 | `/evaluation` | 选模型+测试集、设 limit/采样、提交评测 |
| 测试集仓库 | `/testsets` | 浏览/创建/搜索 HuggingFace 测试集 |
| 测试历史 | `/test_history` | 评测任务列表与结果 |
| 模型仓库 | `/model_repo` | 基座+融合模型列表、详情、配方、跳转3D |
| 融合历史 | `/fusion_history` | 进化融合任务、3D 搜索空间散点图 |

---

## 6. 已完成的修复计划

### 6.1 limit 语义与多卡修复 (**全部完成**)

- 阶段 1：多卡回退条件从 `lim >= 1` 改为 `lim > 1.0`（避免比例 1.0 被误判）
- 阶段 1b：全量时（`"1.0"` 或 `int 1`）不传 `--limit`
- 阶段 2：lmms-eval 子进程用 `load_dataset` + `_resolve_eval_dataset_cap` 换算整数 limit
- 独立函数 `_should_fallback_single_gpu_for_limit` + 11 条单测 + 6 条命令级烟测

### 6.2 模型仓库与评测修复 (**全部完成**)

- DB 读层修复、services 调用修复、测试集去重、VLM 模型分类、GPU 显存泄漏

### 6.3 LLM 评估 OOM 修复 (**全部完成**)

- bind mount 代码更新、GPU 预检机制、重启部署、验证

### 6.4 CMMMU/VLM 评测 (**全部完成**)

- 模型仓库 UI 修复、CMMMU 本地评测、MMBench 本地评测

### 6.5 VLM 评测管线 (**部分待完成**)

- 核心管线已完成（lmms-eval 安装、MMBench/MME/CMMMU 集成、端到端验证）
- **待做**：前端测试集 API 的 VLM 标签展示、兼容性检查 UI

### 6.6 3D 数据与 datasets (**全部完成**)

- 3D 散点图 API、数据集缓存、测试集增强

### 6.7 4-GPU 并行加速 (**部分待完成**)

- Docker GPU 配置、进化融合默认值、评测后端多卡已完成
- **待做**：verify-accelerate（容器内 GPU 可用性验证）、eval-service 优化

### 6.8 系统健康检查 (**基本完成**)

- 全链路烟测通过（LLM/VLM 评测、进化融合、模型仓库 UI）
- **待做**：postcheck-user-llm-task（用户侧真实 LLM 评测任务确认）

---

## 6.9 LLM/进化融合评测准确率修复（MMLU→CMMLU/CMMMU 兼容）（**已完成**）

本节用于解释一次历史上的「MMLU acc 异常偏低」以及在切换到中文/多模态数据集（CMMLU/CMMMU）时可能遇到的**同类字段兼容问题**。结论是：**已修复两类根因，并将修复做成通用归一化**，避免再次踩坑。

### 6.9.1 背景与现象

- **现象 A（MMLU 低分）**：单模型 `Qwen2.5-7B-Instruct` 在 `cais/mmlu` 上出现 ~0.10～0.20 的异常准确率，且 `pred_hist` 强烈偏向 `A/B`，与预期（~0.7+）不符。
- **现象 B（CMMLU/CMMMU 指标异常）**：在 `haonan-li/cmmlu`、`m-a-p/CMMMU` 上如果沿用 MMLU 的字段读取逻辑，容易出现 `gold_dist` 极端偏斜（例如几乎全 A），或选项为空导致模型无法作答。

### 6.9.2 根因与修复（代码路径与要点）

#### 根因 1：Chat Template tokenization 静默降级（已修复）

- **原因**：`tokenizer.apply_chat_template(..., tokenize=True, return_tensors="pt")` 在 transformers 的实现中返回 `BatchEncoding`（`{"input_ids": tensor, "attention_mask": tensor}`），而不是 tensor；旧代码用 `hasattr(tokenized, "tolist")` 判断失败后，静默 fallback 到 `tokenizer.encode(prompt)`，导致模型**没有收到 ChatML/系统消息结构化输入**，评测结果异常。
- **修复点**：`evolution/vendor/vlm_merge/run_vlm_search.py::_format_and_tokenize_prompt`
  正确从 `BatchEncoding["input_ids"]` 提取 token ids，避免 fallback。

#### 根因 2：样本字段不兼容导致「选项丢失 / gold 解析错误」（已修复）

- **MMLU choices**：HF datasets 返回的 `choices` 可能是 `numpy.ndarray`，如果仅按 `list/tuple` 判断会被当作非序列 → `choices=[]` → 题目无选项，准确率异常。
- **CMMLU**：字段为 `Question/A/B/C/D/Answer`，其中 `Answer` 是 `'A'/'B'/'C'/'D'`（不是 0-3）。
- **CMMMU**：字段为 `question/option1..4/answer`（答案为 `'A'/'B'/'C'/'D'`），且包含图像字段（`image_1..`）。
- **修复点**：新增 `run_vlm_search.py::_normalize_mcq_sample()`，统一将样本归一为：
  - `question: str`
  - `choices: list[str]`（优先 `choices` 非空；否则回退到 `A/B/C/D` 或 `option1..4`）
  - `answer_idx: int`（兼容 `0-3` 与 `'A'..'D'`）
  并在 **transformers 路径**与 **vLLM 路径**统一使用该归一化。

### 6.9.3 验收与回归建议（同事可直接照做）

**第一层：字段正确性（不跑模型也能验证）**
- 检查 `gold_dist` 不应极端偏斜（例如 20 个样本几乎全 A）。
- 检查 `empty_choices=0`（每题应有 4 个选项）。

**第二层：模型正确性（小样本）**
- 对 `cais/mmlu` 选一个子集跑 `max-samples=100`：`Qwen2.5-7B-Instruct` 应显著高于历史异常值（~0.27），一般可到 0.7+（视子集波动）。
- 对 `haonan-li/cmmlu` 选一个 config（如 `agronomy`）跑 `max-samples=20`，确保输出为单字母且 `pred_hist` 合理。

### 6.9.4 边界与注意事项（避免误判为“模型坏了”）

- **CMMMU 是多模态数据集**：`run_vlm_search.py --eval-mode text` 属于**纯文本 MCQ**，不会把 `image_*` 送入视觉模型；如果你用 VLM 做融合/评测，需要走 `lmms-eval` 或对应 VLM pipeline（见系统概述中的 VLM 评测）。
- **并发导致 OOM**：如果 GPU 上同时存在 2 个长期占用进程（每个 ~11GiB），再启动新的 transformers `model.to('cuda')` 容易 OOM。建议同一张卡上避免并发跑多个 `eval_final.py`/进化 worker，或使用 vLLM 路径并控制并发。

---

## 7. 模型仓库配方与 3D 修复（阶段 D）

**代码已实现，尚未在计划文件中标记完成（`limit与仓库配方3d合并修复_ff546680.plan.md` 的 todos 仍 pending）**。

已完成的改动：

| 改动 | 文件 |
|---|---|
| `merge_metadata_by_output_path` 用 `realpath` 比较 | `app/services.py` |
| `model_register` 更新时 `task_id=None` 不覆盖 | `app/repositories/__init__.py` |
| 扫描分支填充 `task_id` + `fusion_info` | `app/services.py` |
| DB 读层下发 `task_id`，path 侧 `fusion_info` 回退 | `app/db_read_layer.py` |
| 前端 `refreshRepoListThen` 改用 `/api/model_repo/list` | `templates/model_repo.html` |
| 详情弹窗「融合历史 · 3D」deep link | `templates/model_repo.html` |
| `fusion_history` 支持 `?open_task=` 自动打开 | `templates/fusion_history.html` |

---

## 8. 已知问题（已治理）

以下项已于 **2026-04-04** 按计划修复；本节保留摘要供 Agent 对照。

### 8.1 容器 GPU（已修复）

- **措施**：[`Model_factory/docker-compose.yml`](/home/a/Workspace/Model_factory/docker-compose.yml) 增加 `runtime: nvidia`、`NVIDIA_VISIBLE_DEVICES=all`；`deploy.resources` 保留供 compose v2+。
- **验证**：容器内 `torch.cuda.is_available()` 应为 `True`，`torch.cuda.device_count()` 与宿主机 GPU 数一致。

### 8.2 所谓「API 404」（文档勘误，非缺陷）

以下 URL **从未作为独立端点实现**，前端也未调用；此前为探测误报：

| 误用 URL | 正确用法 |
|---|---|
| `/api/testsets` | `GET /api/testset/list` |
| `/api/leaderboard` | 榜单数据在 `GET /api/testset/<testset_id>` 响应内 |
| `/api/queue_status` | 未实现；任务状态用 `GET /api/status/<task_id>` |

### 8.3 `app.db` 重复挂载（已修复）

- **措施**：移除 compose 中单独的 `app.db` 文件挂载，仅依赖目录 bind mount `./mergeKit_beta` → 容器内项目根，避免 0-byte 宿主文件覆盖真实库。
- **备份**：治理前从容器导出 DB 至 `mergeKit_beta/app.db.exported`；compose 备份为 `Model_factory/docker-compose.yml.bak`。

### 8.4 单体 `app.py` 双轨（已归档）

- **措施**：根目录 `app.py` 已重命名为 **`app.py.legacy`**（只读归档，不删除）；运行时入口仍为 `start_app.sh` → `app/` 包。
- **说明**：`from app import app` 解析的是 `app/` 包，与 `app.py.legacy` 无冲突。

---

## 9. 待开发 / 未完善功能

### 9.1 近期（已有计划但未完成）

| 项 | 来源计划 |
|---|---|
| 前端 VLM 测试集标签 + 兼容性检查 UI | vlm_eval_pipeline |
| 容器内 accelerate 多卡验证 | 4-gpu_parallel_acceleration |
| 用户侧真实评测任务验证 | 系统健康检查 |
| 限时 3D 汇报图（模型仓库内嵌进化过程可视化） | 模型仓库3d汇报图 |

### 9.2 中期（架构优化）

- `app.py.legacy` 中若有仍被外部脚本依赖的路由，再按需迁到 `app/routes.py`（当前前端未依赖）
- `evaluation_results` 表数据稀疏（仅 3 行），评测结果双写 DB 路径需验证
- `evolution_steps` 表为空，CSV → DB 同步未触发或未使用

### 9.3 长期（产品方向）

- 真正的随机采样评测（当前 lm_eval CLI 限制）
- 分布式 Ray 进化融合
- 评测任务优先级队列可视化
- 模型版本管理与对比

---

## 10. 文件约定

| 目录/文件 | 说明 |
|---|---|
| `.cursor/rules/` | AI 协作规则（5 个文件），见 `RULES_INDEX.md` |
| `.cursor/rules/SYSTEM_STATUS.md` | **本文件**：系统现状（Agent 必读） |
| `DEVELOPMENT.md` | 开发指南（技术背景、环境搭建、数据策略） |
| `docs/DEVELOPMENT_LOG.md` | 开发日志 |
| `.omm/` | `oh-my-mermaid (omm)` 架构镜像目录（架构元素与 diagram 字段） |
| `../model_factory_system_architecture.mmd` | 系统架构图（Mermaid 源文件，位于 workspace 根） |
| `scripts/verify_limit_plan_checklist.py` | limit 计划自查脚本 |
| `tests/test_eval_limit_resolution.py` | limit 解析单测 |
| `tests/test_limit_plan_smoke_cmds.py` | limit 命令级烟测 |

---

## 11. 操作速查

```bash
# 启动容器
cd /home/a/Workspace/Model_factory && docker-compose up -d mergekit-beta

# 进入容器
docker exec -it model_factory_mergekit-beta_1 bash -l

# 容器内用 mergenetic 环境
conda activate mergenetic
# 或
conda run -n mergenetic python ...

# 查看日志
docker logs -f model_factory_mergekit-beta_1

# 运行单测（宿主机即可，不依赖 torch）
cd /home/a/Workspace/Model_factory/mergeKit_beta
python3 -m unittest tests.test_eval_limit_resolution -v
python3 -m unittest tests.test_limit_plan_smoke_cmds -v

# 全量自查
python3 scripts/verify_limit_plan_checklist.py

# API 健康检查
curl http://localhost:5000/api/models
curl http://localhost:5000/api/model_repo/list
curl http://localhost:5000/api/fusion_history
curl http://localhost:5000/api/recipes
```
