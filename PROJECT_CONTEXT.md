# Model_factory 项目上下文

> 最后核验：2026-07-17（Asia/Shanghai）
> 当前源码分支：`checkpoint/20260715-platform-gateway`
> 发布流水线验收提交：`61547bf`，已由 merge commit `2fc0406` 接入当前分支
> 用途：为接手本仓库的开发者或 AI Agent 提供可验证的项目事实、当前状态和后续方向。

## 1. 项目概览

`Model_factory` 是一个以 `mergeKit_beta/` 为主工程的 LLM/VLM 模型融合、评测、资产管理和服务发布平台。系统同时包含两条业务主线：

1. 模型工厂：标准融合、进化融合、配方复现、LLM/VLM 评测、模型与测试集管理、历史和 3D 可视化。
2. Model Gateway：管理员控制 vLLM 服务和 API Key，用户通过 OpenAI-compatible API 进行对话，并可提交短期文档或网页资料执行带引用的研究任务。

当前仓库没有语义化版本号。Git checkpoint、分支和验收文档是现有版本边界。

### 2026-07-18 上下文安全默认值修复

- Gateway 文本服务的 `max_model_len` 首发默认值为 65536，VLM 首发默认值为 16384；服务创建 API 即使遗漏该字段也应用对应安全值。
- `gpu_memory_utilization=0.85` 下不再让文本 vLLM 继承模型声明的 128K 上下文，避免单卡 KV Cache 初始化失败。
- 主工作树当前修复尚待受控重建和真实 7B 对话验收；源码通过不等于运行容器已重载。

### 已确认事实

- Git 主工作区当前分支为 `checkpoint/20260715-platform-gateway`；模型发布流水线的源码合并点为 `2fc0406`。当前 HEAD 和远端差异应在每次新会话中重新核验。
- 远端默认分支 `main` 停在 `8728077`，明显早于当前 checkpoint；`master` 停在 `fc96235`。
- merge commit `2fc0406` 的两个父提交是 `79de55a` 和 `61547bf`；原 `feature/model-publication-pipeline` 的提交已全部进入当前 checkpoint，本地 feature 分支和 worktree 已清理。
- 主应用入口是 `mergeKit_beta/start_app.sh` -> `app/__init__.py:create_app()`；`app.py.legacy` 仅作历史参考。
- 2026-07-17 只读检查时，Compose 中 6 个服务正在运行；带健康检查的服务均为 healthy，`/healthz` 与 `/readyz` 返回 HTTP 200。
- 当前合并后分支包含 37 个 Python 测试文件、358 个测试方法、2 个 JavaScript harness，以及 1 个 Java 测试文件中的 2 个 JUnit 测试。
- `ACCEPTANCE_20260715_INTERNAL_PILOT.md` 历史记录 173 个容器测试通过；`ACCEPTANCE_20260717_MODEL_PUBLICATION.md` 记录 feature 完整容器测试 `358/358` 通过。合并后又在 merge commit `2fc0406` 上运行完整 `unittest`，结果同为 `358/358`。
- 发布流水线 Task 7 已在隔离 Compose project `mergekit_publication_task7`、回环端口 5057 上完成真实文本/VLM/GPU 验收，最终提交为 `61547bf`。
- 当前常驻 `model_factory-mergekit-beta-1` 启动于 merge 前。它虽 bind mount 已更新的主工作区源码，但 Python 进程未重启，且现有 mount 清单缺少新 Compose 定义的 `/data/PublishedModels`；当前 5000 运行实例不能视为已完成发布管线部署。
- 仓库没有 GitHub Actions、GitLab CI、Jenkins、tox、pytest 或前端包管理配置。

### 待负责人确认的假设

- 本文把当前 checkpoint 视为已完成源码集成的基线；当前常驻容器仍是 merge 前启动的运行实例，需重建和合并后回归后才能视为部署基线。
- 本文假设 5000 端口只应暴露在可信网络；若实际直接暴露公网，现有核心 API 和 Flask-Admin 的无鉴权状态属于紧急风险。
- 核心 SQLite 中较多 `failed`/`error` 任务可能包含历史回填或失败试验，不能直接等同于近期线上失败率。
- `.cursor/rules/SYSTEM_STATUS.md` 最后更新于 2026-04-16，其中数据规模和部分待办已过时；本文优先采用 2026-07 源码、Git 和运行实例证据。

## 2. 技术栈

| 范畴 | 技术与版本/形式 | 事实来源 |
| --- | --- | --- |
| 主语言 | Python 3.11.14 | `environment.yml` |
| Web 后端 | Flask 3.1.2、Werkzeug、Jinja2 | `environment.yml`、`app/` |
| ORM/迁移 | SQLAlchemy 2.0.46、Flask-SQLAlchemy、Flask-Migrate、Alembic | `environment.yml`、`app/extensions.py` |
| 模型融合 | mergenetic 0.1.1、mergekit 0.1.4 | `environment.yml`、`merge_manager.py` |
| 模型推理 | PyTorch 2.5.1、Transformers 5.3.0、vLLM 0.7.0、Ray 2.9.0 | `environment.yml` |
| 评测 | lm-eval 0.4.11、lmms-eval 相关调用、本地 CMMMU/MMBench 路径 | `merge_manager.py` |
| 数据处理 | Hugging Face datasets 2.20.0、pandas、NumPy | `environment.yml` |
| 检索 | sentence-transformers、FAISS CPU、ONNX Runtime、BGE-M3 路径 | `environment.yml`、`app/model_gateway/` |
| 主数据库 | SQLite 默认，可由 `DATABASE_URL` 切换 | `config.py` |
| Gateway 数据库 | 独立 bind，可配置 PostgreSQL 16 | `config.py`、`docker-compose.yml` |
| 队列 | 模型工厂使用进程内 `PriorityQueue`；研究链路生产配置使用 Redis Streams 7.4 | `app/state.py`、`app/model_gateway/queue.py` |
| 文件安全 | ClamAV 1.5.3、下载 SSRF/重定向校验 | Compose、`scanner.py`、`web_sources.py` |
| 旧 Office 解析 | Java 17、Maven、Apache POI、JUnit 5 | `model_gateway_legacy_parser/` |
| 前端 | Jinja2 HTML、原生 JavaScript/CSS | `templates/`、`static/` |
| 浏览器依赖 | Plotly、Chart.js、GSAP、Remix Icon、MiSans CDN | 模板文件 |
| 包管理 | Conda + pip；Java 子项目使用 Maven | `environment.yml`、`pom.xml` |
| 构建/运行 | Docker、Docker Compose、NVIDIA Container Runtime | Dockerfile、Compose |

`environment.yml` 同时包含大量精确固定的直接和传递依赖。Dockerfile 会先移除部分冲突行，再单独安装指定版本，因此 `environment.yml` 与镜像实际构建步骤需要一起阅读，不能只看其中一个文件。

## 3. 架构概览

```text
Browser / API Client
        |
        v
Flask application (:5000)
  |-- Core routes (no unified auth)
  |     -> Services / in-memory priority worker
  |     -> repositories / SQLAlchemy
  |     -> merge_manager / evolution.runner / subprocesses
  |     -> merges/, recipes/, testset_repo/ and model directories
  |
  `-- Model Gateway routes
        |-- Admin token -> model/API-key lifecycle
        |-- User API key -> /v1/* and research APIs
        |-- loopback HTTP -> managed vLLM processes
        |-- PostgreSQL -> durable gateway/research state
        `-- Redis Streams -> file worker / research worker delivery
                               |-- ClamAV
                               |-- Java legacy parser
                               |-- text parsing/chunking/vector retrieval
                               `-- citation-validated vLLM response
```

### 3.1 模型工厂分层

项目已声明的目标分层是：

```text
HTTP -> app/routes.py -> app/services.py
                         |-> app/repositories/ / app/db_read_layer.py
                         |-> merge_manager.py
                         `-> evolution.runner / subprocess
```

- `routes.py`：HTTP 参数解析、校验、响应和用例编排。
- `services.py`：任务队列、Worker、历史、模型仓库、测试集和配方等业务编排。
- `repositories/`：核心 ORM 写入和查询。
- `db_read_layer.py`：复杂只读组装和文件回退。
- `merge_manager.py`：标准融合、配方应用、lm-eval/lmms-eval 和文件产物处理。
- `evolution/runner.py`：进化融合入口，负责显存裁剪、子进程和进度契约。

存量代码仍有路由直连 ORM、超大模块和文件/DB 双写等历史设计；新增代码不得继续扩大这些问题。

### 3.2 Gateway 与研究链路

- 管理员通过 `/api/model-gateway/admin/*` 创建模型服务、启动/停止 vLLM、创建或撤销用户 API Key。
- 用户通过 Bearer Key 调用 `/v1/models`、`/v1/chat/completions`、`/v1/requests/*`。
- 文件或网页来源先写 `research_files`，再进入 Redis file stream。
- 独立 no-GPU file worker 执行下载、隔离、ClamAV 扫描、解析、分块和向量准备。
- 研究任务写 `research_jobs`，由主容器研究线程消费 Redis research stream；PostgreSQL 是持久状态真相，Redis 只负责交付。
- 检索证据经过上下文预算后送入本机回环 vLLM；只有引用校验通过的结果才提交。
- 源文件、块、payload 和结果受 TTL 清理；使用量和审计元数据保留但不应包含源正文。

### 3.3 数据权威边界

- 单任务细节：`merges/<task_id>/metadata.json` 和 `progress.json` 是兼容期的重要任务事实。
- 全局列表与聚合：核心 DB 优先，必要时从文件补全或回填。
- 配方：`recipes/*.json` 是主数据源。
- 测试集：DB 与 `testset_repo/data/testsets.json` 双写。
- 榜单：`evaluation_results` 与 `testset_repo/data/leaderboard.json` 双写。
- Gateway 生产状态：专用 PostgreSQL。
- Redis：交付媒介，不是研究任务最终真相。

### 3.4 正式模型发布层（已验收并合并，尚待运行部署）

当前分支已包含独立于 `merges/` 的正式资产发布边界：

```text
existing model / recipe / evolution result
        -> publication Task
        -> /data/PublishedModels/.staging/<publication_id>
        -> explicit UUID-gated GPU validation
        -> atomic rename + publication_manifest.json
        -> core Model(source=published)
        -> Gateway compatibility check
        -> administrator-created vLLM service (ready only)
```

- 正式资产内容：`/data/PublishedModels/<publication_id>/`。
- 资产契约：`publication_manifest.json`；新发布使用 schema 2，合法 schema 1 资产保持兼容。
- 发布状态：复用核心 `Task`；正式模型注册：复用核心 `Model`，`source=published`，没有新增数据库表或列。
- schema 2 配方发布绑定 recipe SHA、完整 snapshot、有序父模型、权重 shard/index 指纹和 VLM 基座指纹；来源原地替换会 fail closed。
- Gateway 只接受正式 published asset；只有 serving 状态 `ready` 的资产可创建服务。`blocked` 不可服务，`stale` 必须重新验证。
- 验收中的 Qwen2.5-VL 资产已完成发布和 Transformers 图像推理，但在 vLLM 0.7.0 下为 `blocked/unsupported_architecture`，不会出现在用户模型列表或研究门户。

## 4. 目录结构

| 路径 | 职责 |
| --- | --- |
| `docker-compose.yml` | 主应用和可选 research profile 的部署编排 |
| `.env` | 本机密钥和路径覆盖，已忽略且权限受限；禁止提交或记录值 |
| `.worktrees/` | 本地 Git worktree，已忽略 |
| `mergeKit_beta/` | 主工程 |
| `mergeKit_beta/app/` | Flask 应用、路由、服务、ORM 和数据访问 |
| `mergeKit_beta/app/model_gateway/` | Gateway、研究、鉴权、队列、解析和生命周期子系统 |
| `mergeKit_beta/core/` | GPU 拓扑/锁、路径和进程管理 |
| `mergeKit_beta/evolution/` | 进化融合 runner、契约和内置算法适配 |
| `mergeKit_beta/model_gateway_legacy_parser/` | 隔离的 Java 旧 Office 解析服务 |
| `mergeKit_beta/templates/` | 服务端 HTML 页面 |
| `mergeKit_beta/static/` | 原生 JS/CSS 和静态资源 |
| `mergeKit_beta/tests/` | Python `unittest` 测试 |
| `mergeKit_beta/gateway_migrations/` | Gateway 专用 Alembic 迁移链 |
| `mergeKit_beta/scripts/` | 数据回填、检查、清理、验证和运维脚本 |
| `mergeKit_beta/docs/` | API、数据库、部署、验收和设计文档 |
| `mergeKit_beta/recipes/` | 当前 checkpoint 跟踪 48 份配方，包含已验收配方 `4424f954.json` |
| `mergeKit_beta/testset_repo/yaml/` | 已跟踪的测试集配置/说明；当前 19 份 |
| `mergeKit_beta/merges/` | 运行时任务产物，Git 忽略 |
| `mergeKit_beta/runtime/` | Gateway 运行时文件和数据库卷，Git 忽略 |
| `hf_datasets_cache/`、`eval_datasets_cache/` | 宿主机数据集缓存，非项目源码 |
| `logs/` | 本地验收/运行日志，非项目源码 |

## 5. 重要文件

| 文件 | 作用与注意事项 |
| --- | --- |
| `mergeKit_beta/README.md` | 模型工厂快速入口；未完整覆盖 2026-07 Gateway 现状 |
| `mergeKit_beta/DEVELOPMENT.md` | 环境变量、运行和历史开发说明；部分状态已过时 |
| `mergeKit_beta/.cursor/rules/*.md` | 现有协作纪律和架构边界 |
| `mergeKit_beta/config.py` | 路径、数据库、GPU、Gateway 和研究配置 |
| `mergeKit_beta/app/__init__.py` | 应用工厂、建表/迁移、恢复、Worker 启动 |
| `mergeKit_beta/app/routes.py` | 核心页面和 API；约 1900 行 |
| `mergeKit_beta/app/services.py` | 核心业务编排；约 2950 行 |
| `mergeKit_beta/merge_manager.py` | 融合和评测引擎；约 3500 行 |
| `mergeKit_beta/app/models.py` | 核心 ORM 表 |
| `mergeKit_beta/app/model_gateway/models.py` | Gateway/研究 ORM 表 |
| `mergeKit_beta/app/model_gateway/routes.py` | Gateway 管理、用户和研究 API |
| `mergeKit_beta/app/model_inspection.py` | 结构化模型、权重和 processor 检查 |
| `mergeKit_beta/app/model_publication.py` | manifest、原子发布、恢复和删除保护 |
| `mergeKit_beta/app/model_publication_tasks.py` | 发布任务、物化和显式 GPU 验证 |
| `mergeKit_beta/docs/API.md` | 核心 HTTP API 清单 |
| `mergeKit_beta/docs/DATABASE.md` | 核心 DB 和文件双写规范 |
| `mergeKit_beta/docs/model_gateway/ARCHITECTURE.md` | Gateway 详细架构 |
| `mergeKit_beta/docs/model_gateway/ACCEPTANCE_20260715_INTERNAL_PILOT.md` | 当前 Gateway 内测验收证据 |
| `mergeKit_beta/docs/model_gateway/MODEL_PUBLICATION_PIPELINE_*.md` | 发布流水线设计与实施计划 |
| `mergeKit_beta/docs/model_gateway/ACCEPTANCE_20260717_MODEL_PUBLICATION.md` | Task 7 真实文本/VLM 发布验收证据 |
| `mergeKit_beta/docs/model_gateway/OPERATIONS.md` | 正式资产、GPU、恢复和删除运维规则 |
| `mergeKit_beta/environment.yml` | Conda/pip 依赖快照 |
| `mergeKit_beta/Dockerfile` | CUDA 12.4.1 主镜像构建 |

## 6. 运行、配置与外部服务

### 6.1 启动方式

```bash
cd mergeKit_beta
./start_app.sh
```

Docker 主服务：

```bash
docker compose up -d mergekit-beta
```

研究 profile：

```bash
docker compose --profile research up -d
```

`start_app.sh` 当前使用 Flask 开发服务器，监听 `0.0.0.0`，`debug=True`，关闭 reloader。生产部署前必须改为受控 WSGI/反向代理方案并关闭 debug；不能把当前启动方式视为公网生产配置。

### 6.2 关键环境变量

只记录名称和职责，禁止把实际值写入文档或日志。

| 类别 | 变量 |
| --- | --- |
| 核心路径/端口 | `PORT`、`LOCAL_MODELS_PATH`、`MERGEKIT_MODEL_POOL`、`MERGEKIT_MERGE_DIR` |
| 核心数据库 | `DATABASE_URL` |
| Python/算法 | `MERGENETIC_PYTHON`、`VLM_SEARCH_DIR`、`MERGEKIT_EVOLUTION_LEGACY_BRIDGE` |
| HF/离线缓存 | `HF_ENDPOINT`、`HF_TOKEN`、`HF_DATASETS_CACHE`、`MERGEKIT_EVAL_HF_CACHE`、`HF_*_OFFLINE` |
| GPU/进化 | `CUDA_VISIBLE_DEVICES`、`NVIDIA_VISIBLE_DEVICES`、`MERGEKIT_EVAL_*`、`MERGEKIT_EVOLUTION_*`、`MERGEKIT_VLLM_*`、`MERGEKIT_RAY_*` |
| 正式发布 | `MERGEKIT_PUBLISHED_MODELS_PATH`、`MERGEKIT_PROTECTED_GPU_UUIDS`、`MERGEKIT_PUBLICATION_ALLOWED_GPU_UUIDS` |
| Gateway | `MERGEKIT_MODEL_GATEWAY_ENABLED`、`MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN`、`MERGEKIT_MODEL_GATEWAY_DATABASE_URL`、`MERGEKIT_MODEL_GATEWAY_VLLM_*` |
| 研究队列 | `MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND`、`MERGEKIT_MODEL_GATEWAY_REDIS_URL`、`MERGEKIT_MODEL_GATEWAY_WORKER_TOKEN` |
| 研究安全/限制 | `MERGEKIT_MODEL_GATEWAY_CLAMAV_*`、`MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_*`、配额、大小、TTL 和清理周期变量 |
| Compose 宿主路径 | `HOST_MODELS`、`HOST_MODEL_POOL`、`HOST_PACKAGES`、`HOST_MERGES`、`HOST_HF_DATASETS`、`HOST_EVAL_DATASETS`、`HOST_PUBLISHED_MODELS` |

本机 `.env` 已包含部署所需的一部分敏感变量，文件被 `.gitignore`/`.dockerignore` 排除。本次核验只读取变量名，没有记录值。

### 6.3 外部和基础设施依赖

- NVIDIA GPU、驱动和 `nvidia-container-toolkit`。
- Hugging Face Hub 或配置的镜像端点；Compose 当前默认离线优先。
- 本地模型目录、模型池和 Packages 挂载。
- PostgreSQL、Redis、ClamAV、Java legacy parser（research profile）。
- vLLM 回环服务，由 Gateway 管理员显式启动和停止。
- 用户提交的公开 HTTP(S) 网页；下载逻辑必须保持 SSRF 和重定向校验。
- 前端 CDN；完全离线部署时字体、图表、图标和动画资源可能不可用。

## 7. API、用户流和数据流

### 7.1 核心页面

- `/`：模型融合。
- `/evaluation`：模型评测。
- `/testsets`：测试集仓库。
- `/test_history`：评测历史。
- `/model_repo`：模型仓库。
- `/fusion_history`：进化历史与 3D。
- `/model-gateway`：Gateway 管理员控制台。
- `/research`：用户对话/研究门户。
- `/admin`：Flask-Admin；当前没有认证保护。

### 7.2 模型融合流

1. 前端读取 `/api/models`，可调用 `/api/check_compatibility`。
2. 用户提交 `/api/merge` 或 `/api/merge_evolutionary`。
3. 路由生成 task ID，写入内存状态、优先队列、核心 DB 和任务 metadata。
4. 单 Worker 调用 `merge_manager` 或 `python -m evolution.runner`。
5. 子进程写任务目录、进度、结果和日志；服务同步 DB 和模型注册。
6. 前端轮询 `/api/status/<task_id>`，终态以磁盘和 DB 状态协调。

### 7.3 评测流

1. 用户从模型和测试集列表选择目标，提交 `/api/evaluate`。
2. Worker 根据模型/数据集选择 lm-eval、lmms-eval 或本地 VLM 路径。
3. GPU 拓扑和显存逻辑可能降级并行度或设备数量。
4. 成功结果双写任务目录、`evaluation_results` 和 leaderboard 文件。
5. 前端通过状态、测试历史和测试集详情展示结果。

### 7.4 Gateway 对话流

1. 管理员用 Admin Token 注册模型服务、选择 GPU、启动 vLLM、创建用户 Key。
2. 用户 Key 仅能看到 allowlist 中正在运行的模型。
3. `/v1/chat/completions` 将请求代理到回环 vLLM，并记录请求与 token usage。
4. 非流式请求可能返回 202 和 request ID；用户可查询或取消自己的请求。
5. 重启不会自动重放在途生成；服务状态恢复为需管理员控制。

### 7.5 研究流

1. 用户上传文件或提交 URL，后端进行 owner、大小、类型和配额校验。
2. PostgreSQL 保存来源记录，Redis Streams 负责交付给 no-GPU file worker。
3. 文件/网页经过安全下载、隔离、ClamAV、解析、分块和向量化。
4. 用户提交研究 job；系统按 API Key 和 file IDs 强制所有权范围。
5. 研究 worker 检索证据、构造提示、调用正在运行的 vLLM。
6. 输出通过引用验证后才写结果；失败、暂停、取消和 TTL 由 DB 状态机管理。

### 7.6 正式发布流

1. 管理员从已有模型或配方调用 `POST /api/model-publications`，幂等创建 publication Task。
2. Worker 将候选物化到同文件系统 `.staging`，生成 manifest 和来源指纹。
3. 管理员调用 `/api/model-publications/<task_id>/validate`，显式提交当前容器 GPU index。
4. 后端把 index 解析为 UUID/PCI bus ID，并同时执行 allowed/protected UUID 门禁；验证子进程使用 UUID，而非可变 index。
5. 验证通过后原子 rename，注册唯一 `Model(source=published)`；rename 后崩溃可从 `registration_pending` 幂等恢复。
6. Gateway 管理员候选接口读取 manifest：`ready` 可创建服务，`blocked` 返回明确原因，用户端只看可服务模型。
7. 删除必须调用正式 publication API；任何未软删除服务引用都会返回 `409 asset_in_use`。

## 8. 数据库结构与运行快照

### 8.1 核心数据库

默认 SQLite：`mergeKit_beta/app.db`；可由 `DATABASE_URL` 切换。

| 表 | 作用 | 2026-07-17 只读快照 |
| --- | --- | ---: |
| `tasks` | 融合、进化、评测、配方任务 | 154 |
| `models` | 基座/融合/微调模型注册 | 48 |
| `testsets` | 测试集定义和 HF 缓存元数据 | 25 |
| `evaluation_results` | 评测聚合结果 | 3 |
| `evolution_steps` | 进化搜索步骤 | 0 |
| `tags` | 模型标签 | 1 |
| `model_tags` | 模型-标签多对多关联 | 未统计 |

任务状态快照：67 `completed`、79 `failed`、7 `error`、1 `interrupted`。这些是历史数据事实，不应在缺少时间分布和任务语义分析时解释为当前故障率。

核心 schema 没有完整 Alembic 版本链。SQLite 本地模式会执行 `db.create_all()`，并针对 `testsets` 的两个字段执行手工 `ALTER TABLE`。

### 8.2 Gateway 数据库

Gateway bind 可指向独立 PostgreSQL，使用 `gateway_alembic.ini` 和 baseline revision `fb3e39249ef0`。

| 表 | 作用 | 2026-07-17 只读快照 |
| --- | --- | ---: |
| `serving_model_services` | vLLM 服务定义和进程状态 | 4 |
| `serving_api_keys` | 哈希 API Key、owner 和 allowlist | 13 |
| `serving_requests` | 对话请求状态 | 15 |
| `serving_usage_records` | token usage | 14 |
| `serving_events` | 服务审计事件 | 0 |
| `gateway_quota_buckets` | 配额窗口 | 19 |
| `research_files` | 短期资料来源 | 0 |
| `research_jobs` | 研究任务 | 0 |
| `research_chunks` | 资料分块与定位 | 0 |

服务状态为 3 个 `stopped`、1 个 `failed`。研究内容记录为 0，与验收后清理和 TTL 策略一致，但这里只陈述快照，不推断清理过程。

## 9. 当前开发状态

### 9.1 已完成

| 功能 | 描述 | 相关文件 | 状态证据 |
| --- | --- | --- | --- |
| 标准融合 | Linear、TIES-DARE 等任务编排和产物注册 | `routes.py`、`services.py`、`merge_manager.py` | 已实现并长期使用 |
| 进化融合 | 内置 runner、Ray/vLLM、显存裁剪、进度契约 | `evolution/`、`core/gpu_*` | 已实现；历史任务存在 |
| 配方 | 保存、读取、应用和模型路径归一化 | `recipes/`、`services.py` | 已实现，48 份已跟踪配方 |
| LLM/VLM 评测 | lm-eval、lmms-eval、本地 CMMMU/MMBench、limit 修复 | `merge_manager.py`、tests | 当前分支有专项测试 |
| 模型仓库 | 扫描、DB 同步、标签、详情、删除、3D deep link | `services.py`、`model_repo.html` | 已实现 |
| 测试集仓库 | HF 搜索、创建、缓存、CMMMU 扫描、去重 | `routes.py`、`dataset_info.py` | 已实现 |
| 历史与 3D | 任务状态、历史、CSV/DB/progress 回退 | routes、services、templates | 已实现 |
| 核心 DB | ORM、后台、文件回退和回填脚本 | models、repositories、scripts | 已实现；一致性仍需治理 |
| Gateway 管理 | 服务生命周期、GPU/端口、Admin Token、API Key | `app/model_gateway/` | 真实模型验收文档存在 |
| OpenAI-compatible API | 模型列表、对话、异步查询、取消、usage | Gateway routes/runtime | 已实现并有测试 |
| 研究 Stage 1 | 文件/URL、扫描、解析、检索、引用、配额、TTL | Gateway research modules | 2026-07-15 内测通过 |
| 旧 Office 解析 | 隔离 Java/POI 服务和安全容器限制 | legacy parser | 已实现并有 JUnit/Python 测试 |
| Gateway 迁移 | 独立 Alembic baseline、完整表校验和 stamp | gateway migrations | 已实现 |

### 9.2 已验收并合并，尚待部署回归

#### 模型资产发布与 VLM 物化流水线

- 集成：`61547bf` 已通过 merge commit `2fc0406` 接入当前 checkpoint；原 feature worktree 和本地 feature 分支已清理，原始验收证据迁移到仓库根 `logs/model_gateway/acceptance/`。
- 规模：该 merge 引入 48 个文件变化，约 10152 行新增、355 行删除；当前分支 Python 测试方法为 358。
- 实施计划 Task 1-7 已完成。除原有结构化检查、VLM provenance、原子发布、恢复、API、删除保护和 UI 外，Task 7 期间继续修复了来源权重指纹、manifest schema 2、best genotype 物化、recipe metadata 隔离和受控崩溃恢复。
- 真实验收：文本资产成功发布、启动 vLLM 并记录非零 token usage；标准 VLM 进化、schema 2 发布、Transformers 图像推理、真实 CMMMU 功能样本、幂等恢复和删除保护均已执行。
- 自动化证据：feature 验收和 merge commit `2fc0406` 上的完整 `unittest` 均为 `358/358`；合并后 8 个 HTTP smoke、Compose 配置、两个前端脚本的 `node --check`、`git diff --check` 和 GPU/进程门禁均通过。
- 资源证据：保护 GPU UUID/PCI bus ID 未变化，最终无 vLLM、Ray 或 headless browser 进程，正式 `.staging`、`.trash`、`.quarantine` 为空。
- 保留结果：一个文本资产、一个 schema 2 VLM 资产和配方 `4424f954.json`；旧 schema 1 VLM 资产已归档而非删除。
- 已知边界：Qwen2.5-VL 在 vLLM 0.7.0 下仍为 `blocked/unsupported_architecture`。VLM 资产发布完成不等于 VLM 在线服务完成。
- 部署状态：当前 5000 容器启动于 merge 前，Python 进程未重新加载代码，且缺少 `/data/PublishedModels` mount。下一阶段是部署配置核对、受控容器重建和部署后验收，而不是重复 Task 7 或重复合并后单元回归。

### 9.3 已规划但未完整落地

- `docs/PLAN_SYSTEM_HEALTH.md`：系统健康监控和自动发现仍是 pending 规划。
- 自动化 CI/CD：没有配置。
- 生产入口：没有受控 WSGI、TLS/反向代理和核心管理面鉴权配置。
- 核心数据库的完整版本化迁移链：没有建立。
- 支持 Qwen2.5-VL 的受控 vLLM 版本或替代服务后端：设计为发布流水线之后的独立工作。
- `.cursor/rules/SYSTEM_STATUS.md` 中列出的 VLM 标签 UI、accelerate 多卡验证、用户侧真实评测等旧待办需要逐项重新核验，不能直接视为仍未实现。

## 10. 已知问题和风险

### 严重

1. 核心 API 无统一鉴权。模型删除、历史删除、停止全部任务、测试集写入等接口可被能访问 5000 端口的客户端调用。
2. Flask-Admin 无认证，并允许创建、编辑和删除多类核心记录。
3. `start_app.sh` 使用 `debug=True` 且监听 `0.0.0.0`。该模式不应直接暴露公网或作为正式生产 WSGI。

### 高

1. 无 CI，当前质量依赖人工执行测试、Compose 校验和验收文档。
2. 核心 schema 迁移依赖 `create_all()` 和手工列变更，生产升级、回滚和多数据库兼容性不足。
3. 核心 DB 只有 3 条 `evaluation_results`、0 条 `evolution_steps`，与 154 个任务和历史文件规模不匹配，双写/回填链需要验证。
4. 模型工厂任务队列和运行状态保存在单进程内存中，不支持多 Web 进程安全共享；扩展到 Gunicorn 多 worker 前必须拆分调度所有权。
5. 发布 feature 已合并但当前容器尚未重建；源码、已加载 Python 进程和容器 mounts 处于不同版本边界，部署前必须审查并通过合并后回归。

### 中

1. `app/routes.py` 重复注册 GET `/api/testset/<testset_id>`，两个处理器的数据源和响应结构不同，行为依赖路由匹配顺序。
2. `routes.py`、`services.py`、`merge_manager.py` 体积过大，职责混杂，回归面广。
3. 数据库与文件双写、内存状态和磁盘终态之间存在多套状态映射，故障恢复复杂。
4. 前端依赖公共 CDN，而 Compose 又强调离线运行；离线时 UI 资源可能缺失。
5. 大量异常路径静默 `pass` 或仅记录 warning，运维可观测性有限。
6. `environment.yml` 很大，Docker 又对关键依赖执行二次安装；依赖可复现性需要通过实际镜像构建验证。

### 测试缺口

- 合并后已在主容器的独立测试进程中运行 358 个测试并全部通过；该验证没有重启常驻 Flask 进程，也不能替代部署后验收。
- 没有持续执行的 CI。
- merge commit `2fc0406` 已完成完整非 GPU 回归；尚未完成的是主容器受控重建后的发布目录挂载、运行进程和正式发布链路验收。
- 核心危险 API 的鉴权测试不存在，因为核心鉴权本身不存在。
- 数据库双写一致性、迁移升级/回滚和长期恢复需要更系统的集成测试。

## 11. Git 历史与开发方向

### 分支关系

- `origin/main` / `main`：`8728077`，2026-03 规则文档阶段。
- `origin/master` / `master`：`fc96235`，2026-04 模型工厂合并基线。
- `origin/checkpoint/20260715-platform-gateway`：`010104f`。
- 当前本地 checkpoint 已包含 merge commit `2fc0406` 及本交接文档；具体 HEAD 与领先数量每次会话重新执行 Git 检查。
- 原 `feature/model-publication-pipeline`：`61547bf`，已由 merge commit 接入；本地 feature 分支和 worktree 已在验收证据迁移后删除。

### 近期方向

| 提交/阶段 | 方向 |
| --- | --- |
| `fc96235` | 汇总模型工厂、进化 runner、Compose、路由/服务/UI 和文档 |
| `010104f` | 平台与研究 Gateway Stage 1 checkpoint |
| `cb2c6de` | 内测加固：配额、TTL、网页来源、向量、Alembic 和验收 |
| `ad75a8f`、`41d83e1` | 模型发布流水线设计 |
| `c0bb2dd` | 模型发布流水线详细实施计划 |
| `79de55a` | 忽略本地 worktree |
| feature 前 23 commits | 实施发布流水线 Task 1-6，并多轮修复竞态、生命周期和 UI |
| `d4adfdb` 至 `adb5f43` | Task 7 实跑发现后的 worker 隔离、VLM 物化、来源指纹、schema 2 和恢复加固 |
| `61547bf` | 提交文本/VLM/GPU 真实验收、运维文档和正式 VLM 配方 |
| `2fc0406` | 将已验收模型发布管线合并进当前 checkpoint |

最近一个有完整真实验收记录的里程碑是 2026-07-17 模型发布流水线 Task 7；随后 `2fc0406` 已完成源码合并和合并后完整非 GPU 回归。当前未完成的主要工作是 GPU UUID/正式目录部署配置、容器重建和部署后验收。

## 12. 建议路线图

### P0：发布流水线部署治理

1. 部署前配置非空且不重叠的 protected/allowed GPU UUID，并确认 `/data/PublishedModels` 挂载与保留策略。
2. 在批准窗口重建主容器，使 Python 进程、Compose environment 和 mounts 与合并后源码一致。
3. 受控部署后重新检查健康端点、正式资产、服务引用、GPU 和进程残留。

### P1：生产安全边界

1. 为核心管理页面、Flask-Admin 和危险 API 建立统一认证与授权。
2. 将生产入口改为受控 WSGI/单调度 Worker 架构，关闭 Flask debug。
3. 通过反向代理提供 TLS、请求大小限制、网络 ACL 和审计。
4. 在认证落地前确保 5000 端口只对可信网络开放。

### P1：持续验证

1. 建立最小 CI：Python unittest、Java Maven test、Compose config、JS 语法、`git diff --check`。
2. 将 GPU/真实模型验收保留为人工批准的独立 gate，不放进普通 CI。
3. 记录镜像依赖解析结果和构建 smoke test。

### P2：数据与架构治理

1. 只读运行一致性脚本，确认 3 条 evaluation result 和 0 条 evolution step 的原因。
2. 为核心 schema 建立版本化迁移，停止依赖启动时手工 ALTER。
3. 消除重复测试集路由并固定响应契约。
4. 按用例逐步拆分超大模块；禁止一次性重写。
5. 明确任务状态的单一权威和恢复状态机。

### P3：产品扩展

1. 独立评估支持 Qwen2.5-VL 的 vLLM/后端升级。
2. 实施健康监控和自动发现计划。
3. 评估真正分布式任务队列、版本管理和对比能力。

## 13. 推荐下一步

1. 审查当前 Compose 的 published models mount 和 GPU UUID 门禁配置，再安排 5000 服务的受控容器重建。
2. 部署前确认发布目录挂载、两个 GPU UUID 门禁变量和已保留正式资产的备份/回滚策略。
3. 将 Qwen2.5-VL 在线服务后端升级作为独立项目；当前继续保持 blocked，不绕过兼容性门禁。
4. 随后推进核心鉴权、最小 CI 和 DB/文件一致性审计。

## 14. 需要人工澄清

- merge 后的 checkpoint 是否作为正式集成分支，后续如何同步到 `master`/`main`？
- `origin/checkpoint/20260715-platform-gateway` 是否应接收当前本地 checkpoint 的新增提交？
- 5000 端口当前是否可能从公网或不可信局域网访问？
- 何时批准重建当前主容器，使发布目录 mount 和合并后 Python 代码真正生效？
- 正式部署是否继续使用验收中的 published models 宿主目录和已保留的两个资产？
- Qwen2.5-VL 在线服务后端升级何时启动，是否继续以 vLLM 为首选？
- 历史任务中的 `failed`/`error` 是否有业务上应保留的失败样本或清理要求？
