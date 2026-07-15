# 数据库与数据存放规范（mergeKit_beta）

本文说明 **SQLAlchemy ORM 表结构**、与**磁盘文件的双写关系**，以及读写时的约定。模型定义：`app/models.py`；写入：`app/repositories/__init__.py`；DB 优先只读组装：`app/db_read_layer.py`。

## 数据存放位置与权威来源

- **任务目录（文件）**：`merges/<task_id>/` 是每个任务的“落地包”，里面的 `metadata.json` 记录任务参数与终态；进化任务还有 `progress.json`、`bridge.log`、`vlm_search_results/` 等。
- **数据库（索引与列表）**：DB 表（如 `tasks`、`models`、`testsets`、`evaluation_results`）主要用于**列表查询、后台管理、跨任务聚合**；历史兼容阶段仍保留“文件回退”与“从文件回填 DB”的能力。

一句话：**看单个任务细节以 `merges/<task_id>/metadata.json` 为准；看全局列表与聚合以 DB 为主（必要时从文件回填）。**

## 连接配置

| 项 | 说明 |
|----|------|
| 环境变量 | `DATABASE_URL`（Flask `SQLALCHEMY_DATABASE_URI`） |
| 默认 | `sqlite:///<PROJECT_ROOT>/app.db`（`PROJECT_ROOT` 为 `mergeKit_beta` 根目录，见 `config.py`） |
| Gateway PostgreSQL 迁移 | 专用 Alembic 链：`gateway_alembic.ini`、`gateway_migrations/`；不管理核心 SQLite 表 |

生产可切换 PostgreSQL 等；**时间字段在模型层使用 UTC**（`datetime.utcnow`），便于多时区与迁移。

Gateway 已有完整表首次纳管时会先校验表完整性并 `stamp head`，不会重建或删除数据；缺少任一 Gateway 表时启动失败，禁止用 `create_all()` 掩盖 schema 漂移。

---

## 表与职责

### `tasks`

| 字段（摘要） | 说明 |
|--------------|------|
| `id` | 主键，与 **`merges/<task_id>`** 目录名一致（通常 8 位） |
| `status` | `pending` / `running` / `completed` / `failed` / `queued` 等（与磁盘 `metadata.json` 的 `success`/`error` 在回填时映射） |
| `task_type` | `merge`、`merge_evolutionary`、`eval_only`、`recipe_apply` 等 |
| `config` | JSON，任务参数快照（与 metadata 大量重叠） |
| `custom_name`、`error`、`duration_seconds`、`model_path` | 展示与结果 |
| `gen_1_duration`、`avg_merge_time`、`final_eval_duration` | 进化/融合时序（可选） |

**规范**：创建任务时 `repositories.task_upsert`；结束更新 `task_update_after_completion`；可从 `metadata.json` 回填 `task_backfill_from_metadata`（迁移/修复脚本）。

### `models`

| 字段（摘要） | 说明 |
|--------------|------|
| `path` | 唯一，模型目录绝对路径 |
| `source` | `base` / `merged` / `fine_tuned` |
| `task_id` | 产出该模型的融合任务（可选） |
| `parent_model_ids` | JSON 列表 |
| `tags` | 多对多 → `tags` |

### `testsets`

| 字段（摘要） | 说明 |
|--------------|------|
| `id` | 与业务 `testset_id` 一致 |
| `hf_dataset`、`hf_subset`、`hf_split`、`lm_eval_task` | 评测绑定 |
| `cached_configs`、`cached_splits` | Hub 探测缓存 |
| `benchmark_config`、`yaml_template_path` | 配置与模板路径 |

**规范**：**读**以 DB 为主，`services.testset_list()` 会结合 `testset_repo/data/testsets.json` 与评测历史补全缺失字段并回写；**写**新测试集时双写 DB + `testsets.json`（见 `DEVELOPMENT.md` 测试集补全说明）。

### `evaluation_results`

| 字段（摘要） | 说明 |
|--------------|------|
| `model_id`、`testset_id`、`task_id` | 外键/索引 |
| `accuracy`、`metrics` 等 | 指标快照 |

**规范**：评估成功后在 `services` 中双写 **`evaluation_results`** 与 **`testset_repo/data/leaderboard.json`**（与 `merge_manager._update_leaderboard` 成对出现，保持榜单一致）。

### `evolution_steps`

进化搜索每步一条，供 3D 图与历史查询；`task_id` 外键关联 `tasks.id`，级联删除。

### `tags` / `model_tags`

模型标签多对多，用于分类与生命周期标记。

---

## 磁盘文件与双写矩阵

| 路径 | 内容 | 与 DB 关系 |
|------|------|------------|
| `merges/<task_id>/metadata.json` | 任务参数、状态、进化 Ray 裁剪字段等 | `tasks.config` / 状态回填应对齐；API `GET /api/status` 以磁盘终态修正内存 |
| `merges/<task_id>/progress.json` | 进化/任务进度（Runner 单写者约定见 `evolution/contracts.md`） | 不整表替代 ORM；进化步可同步至 `evolution_steps` |
| `merges/<task_id>/bridge.log` | Runner/子进程日志 | 非 DB |
| `merges/<task_id>/vlm_search_results/` | CSV、中间结果 | 非 DB |
| `testset_repo/data/testsets.json` | 测试集注册表 | 与 `testsets` 双写/补全 |
| `testset_repo/data/leaderboard.json` | 按测试集分组的榜单 | 与 `evaluation_results` 双写 |
| `recipes/<id>.json` | 配方 | 主要文件源；列表 API 直接扫目录 |

---

## 使用规范（给开发者与 Agent）

1. **新增持久化逻辑**：禁止在 `routes.py` 直接 `db.session`；应经 `services` → `repositories`。
2. **任务 ID**：全链路使用同一 `task_id` 字符串，与 `merges/` 目录名一致。
3. **状态对齐**：`metadata.json` 使用 `success`/`error` 等；入库时按 `task_backfill_from_metadata` 的映射转为 `completed`/`failed`。
4. **进化任务**：显存裁剪结果写入 metadata（如 `ray_num_gpus_effective`、`evolution_cuda_visible_devices`）；勿在业务层再维护一份并行度真相。
5. **一致性工具**：`scripts/check_db_file_consistency.py`、`scripts/backfill_db_from_files.py`（见 `DEVELOPMENT.md`）。

---

## 相关文档

- HTTP 接口：[`docs/API.md`](API.md)
- 进化侧契约：[`evolution/contracts.md`](../evolution/contracts.md)
- 开发与运维：[`DEVELOPMENT.md`](../DEVELOPMENT.md)
