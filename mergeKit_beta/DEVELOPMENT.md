# mergeKit_beta 开发进度与注意事项

## 范围说明

本文件仅描述 `Workspaces/mergeKit_beta` 的现状与约定，不覆盖仓库内其他子项目。

## 当前开发进度（已核验）

### 1) 融合与评估主链路

- 标准融合：`/api/merge` -> `app/services.py` -> `merge_manager.py:run_merge_task()`
- 进化融合：`/api/merge_evolutionary` -> `app/services.py` -> `scripts/run_vlm_search_bridge.py`
- 评估任务：`/api/evaluate` -> `app/services.py` -> `merge_manager.py:run_eval_only_task()`
- 配方复现：`/api/recipes/apply` -> `merge_manager.py:run_recipe_apply_task()`

### 2) 数据层能力

- 已接入 ORM 与迁移：Flask-SQLAlchemy + Flask-Migrate
- 已接入管理后台：`/admin`
- 已有核心表：`tasks`、`models`、`testsets`、`evaluation_results`、`evolution_steps`、`tags`
- 当前策略：数据库优先读取，文件回退读取保留（历史兼容）
- **测试集补全**：若 DB 中某条测试集缺少 `hf_dataset`、`lm_eval_task` 等（例如由评测结果自动创建时未带完整参数），在**读取测试集列表**时会自动从 `testsets.json` 或该测试集下的某条 `EvaluationResult` 补全并回写 DB；**写入新评测结果**时若需新建测试集，会优先从 `testsets.json` 取完整信息再写入，避免产生空记录。Navicat 中若仍见部分测试集字段为空，可刷新前端「测试集」列表或重新执行 `scripts/backfill_db_from_files.py` 触发补全。
- **脚本**：`scripts/check_db_file_consistency.py` 对比 DB 与文件一致性；`scripts/backfill_db_from_files.py` 从 `testsets.json` 与 `merges/*/metadata.json` 回填 TestSet/Task 到 DB（可选 `--compare` 回填后跑一致性检查）。

### 3) 前端与可视化

- 任务状态轮询与进度展示可用
- 进化任务支持 3D 数据读取（CSV -> DB -> progress 回退）
- 测试集榜单与基础对比可用

### 4) 回报与刷新机制（已核验并补全）

- **任务回报**：融合/进化/评估均通过轮询 `GET /api/status/<task_id>`（约 1s 间隔）更新进度与结果；完成后会刷新对应列表或展示结果。
- **记忆**：当前进行中的任务 ID 存于 `sessionStorage`（`mergenetic_active_task` / `mergekit_eval_active_task`），刷新页面或从其他页返回时会恢复轮询。
- **列表刷新**：
  - 主页：打开侧栏时拉取 `/api/history`；页面重新可见（visibilitychange）时也会刷新侧栏数据。
  - 评估页：打开侧栏时拉取 `/api/test_history`；visibility 可见时刷新测试历史列表。
  - 融合历史 / 模型仓库 / 测试历史：每页均有「刷新」按钮，且页面从隐藏变为可见时会自动重新拉取列表，避免多 tab 或 bfcache 导致看到旧数据。

## 统一运行规范

- 主入口：`./start_app.sh`
- 重启入口：`./restart_app.sh`
- 标准端口：`5000`
- 模块化应用入口：`app/__init__.py:create_app()`
- `app.py`：仅兼容回退，不作为主入口继续扩展

## 应用启动与外网访问

- **启动方式**：`start_app.sh` 内联 Python 加载 `app/__init__.py` 得到 `app`，执行 `app.run(host="0.0.0.0", debug=True, port=PORT, use_reloader=False)`。绑定 `0.0.0.0` 表示监听本机所有网卡；端口由环境变量 `PORT` 决定，默认 `5000`。前端与 API 由同一 Flask 应用提供，模板在 `templates/`，静态在 `static/`，前端所有请求使用相对路径（如 `/api/...`、`/static/...`），无硬编码 localhost 或固定域名。
- **外网用户能否进入**：可以，应用层已就绪。只要用户能访问到「服务器 IP 或域名:5000」，即可打开前端并正常使用。前置条件：本机防火墙需放行 5000（如 `sudo ufw allow 5000/tcp`）；服务器在内网时需在路由器做端口转发或通过内网穿透（如 natapp）将外网端口映射到本机 5000；对外暴露建议使用反向代理（如 nginx）将 80/443 转发到 5000，并配置域名与 HTTPS。

## 使用注意事项

### 1) 外部脚本依赖

进化融合依赖外部 `run_vlm_search.py`。当前阶段不内嵌重构，相关路径与参数必须保持可用，否则进化任务会失败。

### 2) 数据一致性约定

- 主数据源以 DB 为准（管理、查询、统计都优先 DB）
- 文件数据（如 `metadata.json`、`leaderboard.json`）保留为兼容与回退通道
- 新功能默认先写 DB，再保证文件回退不破坏现有行为

### 3) 风险控制（当前优先级）

- 文档漂移：通过精简文档与单一规范修复
- 双写偏差：以 DB 为主源治理，文件作为回退
- 入口分叉：统一到 `start_app.sh` + `create_app()`
- 扩展边界：新增或修改功能仅进入 `app/` 模块体系（routes/services/repositories 等），禁止在 `app.py` 增加新路由或业务逻辑；`app.py` 仅保留兼容回退

### 4) 模型兼容性说明（融合前）

- 当前兼容性校验以 `hidden_size` 与 `num_hidden_layers` 为核心依据
- 任一模型无法读取上述架构字段时，任务会被拒绝
- 架构字段不一致时，任务会被拒绝

### 5) 评测依赖版本（当前环境）

- **lm_eval**: `0.4.11`（安装命令需用 `pip install 'lm_eval[hf]'` 以包含 HuggingFace 后端）
- **transformers**: `5.3.0`
- lm_eval 0.4.11 已原生使用 `AutoModelForImageTextToText`，兼容 transformers 5.x，**不再需要** `scripts/patch_lm_eval_transformers5.py` 补丁
- 升级/恢复命令：`conda activate mergenetic && pip install --upgrade 'lm_eval[hf]' transformers`

### 6) 依赖冲突说明

- mergenetic 在 `pyproject.toml` 中固定 `transformers==4.45.2` 和 `lm-eval==0.4.8`，升级后 pip 会报冲突警告，属预期行为
- 不重装 mergenetic（`pip install -e .`）则当前环境保持升级后版本，融合与评测可照常使用
- 若执行 `pip install -e .` 重装 mergenetic 会拉回旧版本，需重新运行升级命令

### 7) 融合与评测的架构支持

- **标准融合**（mergekit/mergenetic）：不支持 Qwen3_5 架构（报 Unsupported architecture），仅支持旧架构模型间融合
- **评测**（lm_eval 0.4.11 + transformers 5.3.0）：支持旧架构和新架构（含 qwen3_5 VLM）
- 结论：「旧架构模型融合 + 旧/新架构模型评测」均可做，仅「新架构模型融合」暂不可用

### 8) 可追踪与日志

- 异常/失败日志应尽量携带 `task_id`（任务相关时必选）、`model_id`（涉及模型时）、`testset_id`（涉及评测时）。
- 推荐在 logger 的 message 中直接写出或使用 `extra={"task_id": task_id}`，便于 grep 与后续集中日志。

## 开发步骤细化（参考）

以下为短期/中期/后期开发的操作级参考，具体以 ROADMAP 与当前分支为准。

- **短期 1.1（/api/status DB 优先）**：在 `app/repositories/__init__.py` 新增 `task_get_by_id(task_id)`；在 `app/routes.py` 的 `get_status` 中当任务不在内存时先查 DB，用 Task 组 JSON 并可选以 `status_from_disk` 补 result/evolution_progress/eval_progress。
- **短期 1.2（不在 app.py 扩展）**：见上文风险控制「扩展边界」；可选在 `app.py` 文件头注释注明仅兼容回退。
- **短期 3.1/3.2（可追踪与日志）**：见上文「可追踪与日志」；在 `app/services.py` 的 worker 相关 except 与 logger 中补全 task_id/model_id/testset_id。
- **中期（单机双 worker / 双集群）**：见下节「单机双 worker 与双集群」；实现 `task_claim_next(worker_task_types)`、tasks 表 `priority_order`、提交时必写 DB、Worker 按 `WORKER_TASK_TYPES` 从 DB 消费，单机可起两线程分别跑融合类型与 eval_only。
- **后期（新服务器迁移）**：见下节「新服务器迁移与部署」。

## 单机双 worker 与双集群

- **目标**：融合 worker 只处理 merge/merge_evolutionary/recipe_apply；评估 worker 只处理 eval_only。单机可为一进程两线程，双机为两进程共享同一 DB 与存储。
- **配置**：环境变量 `WORKER_TASK_TYPES`，逗号分隔，如 `merge,merge_evolutionary,recipe_apply` 或 `eval_only`；不设或空则保持现有单机全类型（从内存队列消费）。
- **实现要点**：`app/repositories` 中 `task_claim_next(worker_task_types)` 从 DB 抢占一条 queued 任务并更新为 running；tasks 表需有 `priority_order`（或等价）用于排序；提交接口在入队时写入 DB 且 status=queued；worker 循环按 WORKER_TASK_TYPES 调用 `task_claim_next`，抢到后执行现有 run_merge_task/run_eval_only_task/run_recipe_apply_task 并写回完成状态。双机时两进程共享 DATABASE_URL、merges/、模型目录等。在新服务器 **4×RTX 3090（两两 NVLink）+ 128GB 内存** 场景下，推荐通过 `CUDA_VISIBLE_DEVICES` 或进程启动参数，将不同 worker 绑定到不同 GPU 组（如融合使用 0,1，评估使用 2,3）；eval_only 并发数可按负载上调（128GB 比 64GB 更宽裕），但仍建议设上限并观察 `htop`/`nvidia-smi`，避免 lm_eval/datasets 与多进程把内存或某张卡打满。

## 新服务器迁移与部署

- **代码获取**：Git clone 或 rsync 同步 `Workspaces/mergeKit_beta`（不含虚拟环境与大型模型目录）。
- **环境**：在旧机使用 `conda env export -n mergenetic --no-builds > environment.yml` 导出当前运行环境；将 `environment.yml` 拷贝到新机仓库根，在新机执行 `conda env create -n mergenetic -f environment.yml`（已存在时可用 `conda env update -n mergenetic -f environment.yml`），确保新机与旧机依赖版本一致；`config.py` 中路径通过环境变量覆盖（见下）。
- **配置清单**：`DATABASE_URL`、`LOCAL_MODELS_PATH`、`MERGEKIT_MODEL_POOL`、`PORT`、`MERGENETIC_PYTHON`、`MERGEKIT_EVAL_HF_CACHE`、`HF_DATASETS_CACHE`、`WORKER_TASK_TYPES`（双集群时使用）；项目内路径如 `MERGE_DIR`、`LOGS_DIR`、`RECIPES_DIR`、`TESTSET_REPO` 等默认相对 `PROJECT_ROOT`。
- **数据迁移顺序**：DB（拷贝 app.db 或迁至 PostgreSQL）→ merges/ → 模型目录（LOCAL_MODELS_PATH、MODEL_POOL_PATH）→ testset_repo（testsets.json 及已下载数据集）→ recipes/。
- **部署命令**：单机 `./start_app.sh`；生产可用 gunicorn/uWSGI + systemd 或 supervisor；双集群时两台机分别设不同 `WORKER_TASK_TYPES`，共享同一 `DATABASE_URL` 与存储。
- **迁移后校验**：执行 `scripts/backfill_db_from_files.py --compare`。
- **回滚要点**：保留旧机 DB 与关键目录备份；若新机出问题可切回旧机或从备份恢复。

## Docker 构建与迁移到新服务器

- **目标机硬件（当前规划）**：CPU AMD EPYC 7543；GPU 4×RTX 3090（两两 NVLink）；内存 **128GB** DDR4；与本节多卡、`CUDA_VISIBLE_DEVICES` 及并发建议一致。
- **目标**：在不改变现有开发方式的前提下，提供一套基于 Docker 的「整仓 + GPU」运行环境，用于新服务器迁移与按需复现，且容器内环境与本机 `mergenetic` conda 环境保持一致。
- **环境标准**：以旧机导出的 `environment.yml` 为唯一环境真相；Docker 镜像与新服务器宿主机都通过这份文件创建/更新 `mergenetic` 环境，避免因依赖差异导致无法运行。
- **Dockerfile 位置与构建**：
  - 文件在 `Workspaces/mergeKit_beta/Dockerfile`，**构建上下文为仓库根 `ServiceEndFiles`**（与 Git 仅跟踪 `mergeKit_beta` 不矛盾）。
  - 命令：`cd ServiceEndFiles && docker build -f Workspaces/mergeKit_beta/Dockerfile -t mergekit-beta .`
  - 可选：`docker compose build`（使用仓库根 `docker-compose.yml`）。
  - 基础镜像：`nvidia/cuda:12.4.1-runtime-ubuntu22.04`；镜像内安装 Miniconda，用本目录 `environment.yml` 执行 `conda env create -n mergenetic`（Dockerfile 内已含 `conda tos accept`，以通过新版 Conda 对 `defaults` 频道的服务条款校验）。
  - **Docker 与裸机差异**：`conda env create` 会严格解析 pip 依赖，无法复现「本机已手动升级 lm-eval/transformers 后与 mergenetic 元数据冲突但仍可跑」的状态；Dockerfile 在创建环境前会从副本里**去掉**若干与 ray/mergenetic 冲突的显式钉版本（如 `lm-eval`、`transformers`、`huggingface-hub`、`tokenizers`、`virtualenv`），创建成功后再 `pip install` 升级到与上文「评测依赖版本」一致的栈；`pip` 可能对 `mergenetic` 报版本不兼容警告，与裸机说明一致，可忽略除非重装 mergenetic。
  - 镜像内路径：`WORKDIR=/app/ServiceEndFiles/Workspaces/mergeKit_beta`；`Packages/` 复制到 `/app/ServiceEndFiles/Packages/`；`modelmerge_visual` 若本机不存在则仅创建空目录，**进化融合**需挂载含 `run_vlm_search.py` 的树或设置 `VLM_SEARCH_DIR`。
  - 启动：`CMD ./start_app.sh`（与宿主机一致）；通过环境变量与卷传入 `LOCAL_MODELS_PATH`、`MERGEKIT_MODEL_POOL`、`DATABASE_URL`、`PORT`、`VLM_SEARCH_DIR`、模型与 merges 等。
- **.dockerignore 建议**：
  - 排除 `Models/`、`Models-local_dir/`、`mergeKit/models_pool/` 等大模型目录，以及 `Datasets/`（若体量较大）。
  - 排除 `.git/`（可选）、`logs/`、`__pycache__/`、`*.pyc`、`.cursor/`、`.env` 等不应进镜像的内容。
  - 保证 `Workspaces/mergeKit_beta/`、`Workspaces/modelmerge_visual/`（含 VLM_merge_total/VLM_merge）、`Packages/` 等代码被打包入镜像。
- **运行与迁移流程（Docker 视角）**：
  - **建议先在当前开发机**完成 `environment.yml` 导出、`Dockerfile` / `.dockerignore`（及可选 compose）编写，并在本机 `docker build` 做一次冒烟，再迁到新服务器；详见 Cursor 计划「Docker 整仓迁移方案」§0（当前设备先行与 Git）。
  - 在新机安装 Docker 与 nvidia-container-toolkit，在 `ServiceEndFiles` 根目录构建：`docker build -f Workspaces/mergeKit_beta/Dockerfile -t mergekit-beta .` 或 `docker compose build`。
  - 同步 `Models/`、`merges/`、`app.db`、`testset_repo/`、`recipes/` 至新机指定路径，通过 `-v` 或 docker-compose `volumes` 挂载到容器内。
  - 使用 `docker run --gpus all -p 5000:5000 ... mergekit-beta` 或 `docker compose up` 启动，验证 Web 与融合/评估任务是否正常。
  - 日常开发仍按照上文「新服务器迁移与部署」在宿主机激活 `mergenetic` 环境并运行 `./start_app.sh`；需要对比或复现时，使用基于同一 `environment.yml` 构建的 Docker 镜像启动服务。
- **Git 版本管理**：强烈建议将代码与 `environment.yml`、Docker 相关文件纳入 Git，大目录（Models、大体量 Datasets、日志、缓存、`.env` 等）用 `.gitignore` 排除；便于新机 `git clone` 与回滚。Docker 构建不依赖 Git，但迁移与协作强烈依赖版本管理。

## 近期已完成的治理动作

- 端口规范统一为 `5000`
- 启动/重启脚本与测试脚本端口同步
- 文档收敛为三份，移除过时与重复说明
- lm_eval 升级至 0.4.11 + transformers 升级至 5.3.0，代码已适配新 API

## 快速核验清单

```bash
./restart_app.sh
curl -s http://127.0.0.1:5000/api/models
curl -s http://127.0.0.1:5000/api/history
```

若三个请求可用，说明基础服务链路正常。

## 文档变更历史

| 日期       | 变更摘要 |
|------------|----------|
| 2025-03-04 | DB 主源收敛：Phase A–E 落地（TestSet/EvaluationResult/Task/Model 双写消除或补全，metadata 统一写入，基座模型启动扫描，一致性校验与回填脚本）；测试集补全逻辑（读列表/写评测时从文件或 EvaluationResult 补全并回写）；数据层能力中补充脚本 `check_db_file_consistency.py`、`backfill_db_from_files.py` 说明；Navicat 连接文档新增外网映射、Y: 盘、Samba、方案二及变更历史。 |
| 2025-03-04 | 新增「应用启动与外网访问」「可追踪与日志」「开发步骤细化（参考）」「单机双 worker 与双集群」「新服务器迁移与部署」「Docker 构建与迁移到新服务器」；风险控制中增加扩展边界（不在 app.py 扩展）；迁移与 Docker 章节强调以 environment.yml 作为统一环境标准。 |
| 2025-03-10 | 目标新服务器内存更正为 128GB；Docker 小节补充目标机硬件摘要；双 worker 场景下 eval 并发建议改为按负载与内存观察调整。 |
| 2025-03-10 | Docker 小节补充：建议当前设备先行构建冒烟、Git 与 `.gitignore` 约定；与 Docker 整仓迁移方案 §0 对齐。 |
| 2025-03-10 | Docker：实际 `Dockerfile` 位于 `mergeKit_beta/`，构建上下文为 `ServiceEndFiles`；仓库根增加 `.dockerignore`；`modelmerge_visual` 缺失时镜像内仅占位目录，进化融合需挂载或 `VLM_SEARCH_DIR`。 |
| 此前       | 端口统一 5000；文档收敛为 README/DEVELOPMENT/ROADMAP；lm_eval 0.4.11 + transformers 5.3.0 升级与适配。 |
