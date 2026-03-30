# 架构与分层边界（mergeKit_beta）

目的：减少「路由里写 SQL、双轨 Worker、任务逻辑散落在多处」带来的回归成本。**存量代码**允许逐步收敛；**新增与修改**应遵守下文。

## 应用入口

- **唯一推荐**：`./start_app.sh` → `app/__init__.py:create_app()` → 注册 `routes`、启动 `Services` 内 Worker。
- **`app.py`**：仅保留历史兼容；**禁止**在其中新增业务路由、Worker 分支或与 `app/services.py` 并行的任务执行逻辑。新功能一律进入 `app/` 包（`routes` / `services` / `repositories` 等）。

## 接口层（HTTP）

- **`app/routes.py`**：负责请求解析、校验、调用 services、`jsonify`、HTTP 状态码。
- **新增代码**：不要在路由中直接使用 `db.session` 做查询或写入；应通过 `services` → `repositories`（或既有 `db_read_layer` 模式）。**既有**路由内的 `db.session` 仅为历史遗留，修改该文件时优先顺带迁出，非强制一次性大扫除。

## 业务编排层

- **`app/services.py`**：用例编排、任务队列 Worker、调用 `merge_manager`、子进程（如进化桥接）、与 `state` 交互。
- 与 HTTP 无关的可复用逻辑：优先沉淀为 `services` 内方法或独立模块，避免堆在 `routes`。

## 数据访问层

- **ORM 模型**：`app/models.py`。
- **读写封装**：`app/repositories/__init__.py`（及同类模块）。
- **复杂只读**：`app/db_read_layer.py`。
- **数据策略**：与 [DEVELOPMENT.md](../../DEVELOPMENT.md) 一致——**数据库优先**，文件（如 `metadata.json`、`testsets.json`）作为兼容与回退。

## 任务引擎与文件落地

- **`merge_manager.py`**：标准融合、评测主流程、配方应用、与 `lm_eval`/文件系统交互。
- **`merges/<task_id>/`**：`metadata.json`、`progress.json`、产物目录等——面向单次任务的**文件型真相**；与 DB 中 `tasks` 等表的同步应在 `services` 与既有 DB 更新路径中完成，避免在路由层直接改文件。
- **`scripts/`**（如进化桥接）：视为**外部编排或适配层**，职责是衔接第三方脚本；业务上仍由 `services` 触发，规则文件不鼓励从 `routes` 直接调用脚本。

## 小结

```text
HTTP 请求 → routes → services → repositories / merge_manager / 子进程
                ↘（避免新增）db.session 直连
```
