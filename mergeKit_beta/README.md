# mergeKit_beta

`mergeKit_beta` 是当前模型融合工厂主工程（仅围绕本目录维护），提供 Web 化的融合、评估、历史追踪和基础可视化能力。

## 统一入口与端口（当前规范）

- 统一启动入口：`./start_app.sh`
- 统一重启入口：`./restart_app.sh`
- 统一端口：`5000`（可用环境变量 `PORT` 覆盖）
- 推荐访问：`http://localhost:5000`

说明：
- `start_app.sh` 走模块化入口 `app/__init__.py:create_app()`，这是当前最完整入口。
- `app.py.legacy`（已归档的旧单体入口）不作为主入口文档对外推荐。

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
- 进化融合算法：默认使用仓内 `evolution/vendor/vlm_merge/run_vlm_search.py`；如需对比调试/临时回滚，可通过环境变量 `VLM_SEARCH_DIR` 覆盖算法目录（见 `evolution/runner.py` 说明）。
- 维护脚本：`scripts/check_db_file_consistency.py`（DB 与文件一致性）、`scripts/backfill_db_from_files.py`（从文件回填 TestSet/Task）

## 架构图（oh-my-mermaid / omm）

- Mermaid 源文件：`/home/a/Workspace/model_factory_system_architecture.mmd`
- omm 元素：`overall-architecture`（存储于工作区 `.omm/`）

导出为图片（SVG/PNG）：

```bash
cd /home/a/Workspace
npx -y @mermaid-js/mermaid-cli -i model_factory_system_architecture.mmd -o model_factory_system_architecture.svg
npx -y @mermaid-js/mermaid-cli -i model_factory_system_architecture.mmd -o model_factory_system_architecture.png
```

## 文档

| 文档 | 说明 |
|------|------|
| `DEVELOPMENT.md` | 开发进度、注意事项、运行规范、依赖与兼容性；外网访问、可追踪与日志、开发步骤细化、单机双 worker/双集群、新服务器迁移与部署 |
| `docs/DEVELOPMENT_LOG.md` | 阶段性开发记录（VLM 评测接入、LLM 评测修复等） |
| `docs/PLAN_SYSTEM_HEALTH.md` | 系统健康监控/自动发现的规划文档（未落地，作为需求草案） |

## 进化融合（text + vLLM TP + Ray）使用注意

- **可以正常使用**：标准融合、配方、lm_eval 评测、进化融合 API（`/api/merge_evolutionary` 等）在更新后仍按原流程工作；text 进化在 **TP>1** 时默认走 **子进程 vLLM**，单次任务总时间可能变长，但应避免此前「第二代评测 TCPStore 长时间卡住」类问题。
- **单机 Docker（推荐）**：在 `docker-compose` 或环境中设置 `MERGEKIT_RAY_SINGLE_NODE_LOOPBACK=1`；若仍有网络解析问题，可再加 `MERGEKIT_DOCKER_HOSTS_LOOPBACK_FIX=1`（见 `DEVELOPMENT.md`）。
- **多机 / 可能多节点 Ray**：**不要**开启 `MERGEKIT_RAY_SINGLE_NODE_LOOPBACK`；按需配置 `MERGEKIT_RAY_NODE_IP_ADDRESS`。
- **回滚子进程**：`MERGEKIT_VLLM_TP_SUBPROCESS=0` 恢复进程内 vLLM（存在第二轮 c10d 风险，仅建议对比排障）。
- **兜底**：仍可用 `MERGEKIT_VLLM_TP_SERIALIZE=1`、`MERGEKIT_VLLM_ENABLE=0`、`MERGEKIT_EVOLUTION_TP2=0` 等（详见 `DEVELOPMENT.md` 环境表）。

## 文档变更历史

| 日期       | 变更摘要 |
|------------|----------|
| 2026-04-09 | 补充进化融合 vLLM TP=2 + Ray 子进程隔离、`MERGEKIT_RAY_SINGLE_NODE_LOOPBACK` / hosts 门禁等运维说明；与 `DEVELOPMENT.md`、`docs/DEVELOPMENT_LOG.md` 同步。 |
| 2025-03-04 | 与 DEVELOPMENT/ROADMAP 同步：数据主源 DB、测试集自动补全、文档索引与变更历史；架构约定中补充维护脚本说明；快速开始后增加外网访问说明，文档表补充 DEVELOPMENT 中迁移与部署等章节说明。 |
| 此前       | 统一入口与端口；已实现能力与架构约定收敛。 |
