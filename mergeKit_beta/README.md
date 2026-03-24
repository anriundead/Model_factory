# mergeKit_beta

`mergeKit_beta` 是当前模型融合工厂主工程（仅围绕本目录维护），提供 Web 化的融合、评估、历史追踪和基础可视化能力。

## 统一入口与端口（当前规范）

- 统一启动入口：`./start_app.sh`
- 统一重启入口：`./restart_app.sh`
- 统一端口：`5000`（可用环境变量 `PORT` 覆盖）
- 推荐访问：`http://localhost:5000`

说明：
- `start_app.sh` 走模块化入口 `app/__init__.py:create_app()`，这是当前最完整入口。
- `app.py` 仅保留兼容回退能力，不作为主入口文档对外推荐。

## 快速开始

1) 在可用环境中安装依赖（至少包含 Flask、SQLAlchemy、Flask-Migrate、Flask-Admin、mergenetic、lm_eval、datasets 等）。

2) 启动服务：

```bash
./start_app.sh
```

3) 打开：

```text
http://localhost:5000
```

外网访问：服务已监听 `0.0.0.0:5000`，前端使用相对路径，外网用户只要能访问到服务器 5000 端口即可使用；需放行本机防火墙，内网时需端口转发或内网穿透，生产建议 nginx 反向代理与 HTTPS。详见 DEVELOPMENT.md「应用启动与外网访问」。

## 已实现能力（核验后）

- 标准融合（Linear / TIES-DARE）
- 进化融合（桥接外部搜索脚本）
- 配方保存与配方复现融合
- 评估任务（lm_eval / YAML）
- 进化步骤 CSV 与 `evolution_steps` 数据库同步
- 历史与榜单（数据库优先，文件回退）
- 测试集列表：DB 中缺失的测试集字段（如 `hf_dataset`、`lm_eval_task`）会在读取时从文件或评估历史自动补全并回写
- 管理后台（`/admin`）

## 当前架构约定

- 数据主源：数据库（SQLite 默认，支持 `DATABASE_URL` 切换）
- 文件策略：保留读取回退能力（`metadata.json` / `leaderboard.json` 等）
- 运行目录：`merges/`、`recipes/`、`logs/`
- 外部依赖：进化融合仍依赖外部 `run_vlm_search.py`，暂不内嵌重构
- 维护脚本：`scripts/check_db_file_consistency.py`（DB 与文件一致性）、`scripts/backfill_db_from_files.py`（从文件回填 TestSet/Task）

## 文档

| 文档 | 说明 |
|------|------|
| `DEVELOPMENT.md` | 开发进度、注意事项、运行规范、依赖与兼容性；外网访问、可追踪与日志、开发步骤细化、单机双 worker/双集群、新服务器迁移与部署 |
| `ROADMAP.md` | 后续需求与开发路径（短期/中期/长期） |
| `docs/NAVICAT_连接数据库.md` | 外网环境下用 Navicat 连接 SQLite（映射 Y: 或下载 app.db）及 Samba 配置 |

## 文档变更历史

| 日期       | 变更摘要 |
|------------|----------|
| 2025-03-04 | 与 DEVELOPMENT/ROADMAP 同步：数据主源 DB、测试集自动补全、文档索引与变更历史；架构约定中补充维护脚本说明；快速开始后增加外网访问说明，文档表补充 DEVELOPMENT 中迁移与部署等章节说明。 |
| 此前       | 统一入口与端口；已实现能力与架构约定收敛。 |
