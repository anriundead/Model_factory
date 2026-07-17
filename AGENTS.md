# AGENTS.md

本文件适用于仓库根目录及全部子目录。未来 AI Agent 和开发者在开始工作前必须先读本文件，再读 `PROJECT_CONTEXT.md`、`mergeKit_beta/.cursor/rules/RULES_INDEX.md` 及任务相关文档。

## 项目规则

### 1. 修改授权

- 对任何文件写入、代码修改、配置修改、数据库变更、运行时数据清理、Git 提交或外部发布，必须先向用户说明范围并取得明确确认。
- 只读分析不等于修改授权。用户要求“分析”“诊断”“评审”时，不得顺带修复。
- 删除、移动、批量清理文件，以及不可逆数据库操作，必须在一般修改确认之外再次列出清单并获得二次确认。
- 不得修改、覆盖或回退用户已有的未提交变更。

### 2. 事实与假设

- 在提出方案前检查实际源码、配置、依赖、运行入口、Git 状态和相关历史。
- 明确标注“已确认事实”“推断”“待负责人确认”。不要把旧文档、计划或测试名称当作功能已完成的证据。
- 当前 checkpoint 已通过 `2fc0406` 合并 publication feature，但当前 5000 容器启动于 merge 前；必须分开描述“源码已合并”“进程已重载”“Compose mounts 已更新”和“部署验收通过”。
- 不得输出 `.env`、数据库连接串、API Key、Token、模型私有路径或日志中的敏感值。

### 3. 编码约定

- Python 以 3.11 为基线，遵循现有模块、命名、类型提示和 `unittest` 风格。
- 保持最小 diff；不夹带格式化、重命名、依赖升级或无关清理。
- 优先复用现有标准库、模块和依赖，不为单一用例添加抽象或新依赖。
- 新代码应有清楚的错误边界；禁止新增无日志、无说明的宽泛 `except Exception: pass`。
- 注释说明原因、约束或恢复语义，不重复代码表面行为。
- 用户沟通、维护文档和提交说明优先使用简体中文；已有英文技术文档可保持原语言一致性。

### 4. 架构边界

- 唯一推荐入口：`mergeKit_beta/start_app.sh` -> `app/__init__.py:create_app()`。
- `mergeKit_beta/app.py.legacy` 只读参考，禁止新增路由、Worker 或业务逻辑。
- HTTP 层：`app/routes.py` 或 `app/model_gateway/routes.py` 只处理请求、校验、编排和响应。
- 业务编排：进入 `app/services.py` 或对应 Gateway 领域模块。
- 核心数据库访问：通过 `app/repositories/` 或 `app/db_read_layer.py`；不要在新路由中直接使用 `db.session`。
- 融合/评测核心：沿用 `merge_manager.py`；进化任务沿用 `evolution.runner` 契约。
- Gateway 和模型工厂必须保持边界：Gateway 不得绕过正式模型注册和发布规则读取任意融合中间目录。
- 在 publication feature 或其集成版本中，正式资产只存在于 `/data/PublishedModels/<publication_id>/`，以 `publication_manifest.json` 和核心 `Model(source=published)` 为准；Gateway 不得从 `merges/` 直接创建服务。
- 新发布 manifest 使用 schema 2；不得删除 schema 1 兼容读取，或把 `_publication_recipe_metadata.json` 改回会被历史 backfill 读取的标准 `metadata.json`。
- PostgreSQL 是生产研究状态真相，Redis 只负责交付；不得把 Redis pending 状态当作最终业务状态。
- 核心任务现为单进程调度。未经专门设计，不得通过增加多个 Web worker 来“扩容”，否则可能产生重复 Worker 和状态竞争。
- 不得在没有迁移和回滚设计的情况下修改数据库 schema。

### 5. 文件与数据规则

- `merges/`、`runtime/`、`logs/`、缓存、模型文件和 `.env` 是运行时数据，不得提交。
- `recipes/*.json` 是业务配方，不得当作临时文件批量删除。
- 单任务文件、核心 DB、Gateway PostgreSQL 和 Redis 有不同权威边界；修改同步逻辑前先读 `docs/DATABASE.md` 和 Gateway 架构文档。
- 路径删除必须使用既有允许目录和真实路径校验；不得扩大删除根目录。
- 正式 published asset 不得手工移动或删除；必须调用 publication 删除 API，并先处理所有未软删除的 Gateway service 引用。
- 不得提交本机绝对路径、API Key、管理员 Token、数据库密码、HF Token、模型权重或原始用户资料。

### 6. 测试要求

- Python 测试框架是标准库 `unittest`，主命令：

```bash
cd mergeKit_beta
python -m unittest discover -s tests
```

- Java legacy parser：

```bash
cd mergeKit_beta/model_gateway_legacy_parser
mvn test
```

- Compose 配置：

```bash
docker compose config --quiet
docker compose --profile research config --quiet
```

- 前端 JavaScript 至少执行 `node --check <changed-file.js>`；涉及交互时按现有验收模式做桌面/移动浏览器检查。
- 所有改动完成前执行 `git diff --check` 和与改动范围对应的测试。
- 不得声称“测试通过”，除非在当前会话中运行完整命令并看到退出码 0。
- GPU、真实模型、数据库迁移或研究链路的高风险改动必须执行专门验收；单元测试不能替代真实运行验证。
- 测试可能创建目录、DB、缓存或任务记录。运行前确认其副作用和目标环境，必要时先征求用户批准。

### 7. 部署要求

- Docker 和 Compose 是当前部署基线；不要假定宿主机裸跑与容器完全一致。
- GPU 操作前记录 UUID、bus ID、显存和进程；不得仅按可变 GPU index 识别受保护设备。
- 发布验证必须配置非空且互不重叠的 `MERGEKIT_PROTECTED_GPU_UUIDS` 与 `MERGEKIT_PUBLICATION_ALLOWED_GPU_UUIDS`；API index 必须解析并校验为 UUID/PCI bus ID，CUDA 子进程使用 UUID。
- 不得终止未由当前任务创建或无法确认归属的 vLLM、Ray、Python、浏览器或 GPU 进程。
- Gateway vLLM 只应绑定回环地址，并由管理员生命周期 API 管理。
- Research worker 必须保持 no-GPU、独立扫描/解析职责和 TTL 清理边界。
- 当前 Flask `debug=True` 启动方式不是公网生产方案。任何生产入口改造需单独设计认证、WSGI、反向代理、TLS 和单调度所有权。
- 部署变更必须提供回滚命令、保留数据清单和健康检查结果。

## 开发工作流

修改代码前必须执行以下流程：

1. 分析现有实现。
2. 向用户解释拟议变更、依据、风险和不做什么。
3. 列出将创建、修改或删除的准确文件；取得明确授权。
4. 实施最小范围变更。
5. 运行与风险相匹配的测试和验证。
6. 汇总修改、验证证据、未验证项、风险和回滚方式。

推荐的具体步骤：

1. 运行 `git status --short --branch`，确认分支和用户改动。
2. 阅读本文件、`PROJECT_CONTEXT.md`、`.cursor/rules/` 和任务相关设计/验收文档。
3. 用 `find`/`grep`/`rg` 检查所有调用方和数据边界；不要只读报告中提到的单一函数。
4. publication Task 1-7 已合并到当前 checkpoint；不要重复实现，并先确认当前运行容器是否已重建到合并后版本。
5. 给出受影响文件、接口、DB、运行时和测试清单，等待授权。
6. 对功能/bugfix 优先先写或确认失败测试，再做最小实现。
7. 只运行获准且副作用明确的命令。
8. 检查 `git diff --stat`、`git diff --check`、`git status`，确认无额外文件。
9. 未经用户要求，不提交、不推送、不创建 PR、不重启生产服务。

## Git 与分支规则

- 当前已知基线、远端和 worktree 状态见 `PROJECT_CONTEXT.md`；每次工作仍需重新核验。
- 不得假设 `main`、`master` 或当前 checkpoint 自动是正确目标分支。
- 模型发布流水线已在 `61547bf` 完成 Task 7 验收，并由 `2fc0406` 合并到当前 checkpoint；原 feature worktree 和本地 feature 分支已在证据迁移后清理。后续不得从历史分支重复实现，应以当前 checkpoint 为集成基线，并先核对运行容器是否已完成受控重建。
- 禁止 `git reset --hard`、强制 checkout、强推和历史重写，除非用户明确批准。
- 提交应小而可审查，说明“做了什么、为什么”；遵循项目现有中文提交约定，除非目标分支已有明确英文规范。
- 不提交 `.worktrees/`、运行时日志、截图原件、数据库、缓存或秘密。

## AI Agent 应避免的事项

- 不经批准重写架构、拆分大文件或迁移框架。
- 不在 `app.py.legacy` 开发新功能。
- 不在路由中新增直接 SQL/ORM 写入。
- 不随意修改数据库 schema、Alembic baseline、表名或双写契约。
- 不删除现有功能、兼容回退、历史文件或配方。
- 不引入不必要的依赖、队列、前端框架、ORM 层或抽象。
- 不把规划文档中的代码当作已合并实现。
- 不把测试数量、代码存在或 UI 可见当作真实模型验收成功。
- 不把“VLM 正式资产已发布”写成“VLM 在线服务已完成”；Qwen2.5-VL 在当前 vLLM 0.7.0 下必须保持 `blocked/unsupported_architecture`。
- 不绕过 HTTP/任务注册直接调用 vendor 脚本来宣称端到端成功。
- 不自动选择或占用 GPU 2/受保护 GPU；具体保护范围每次由用户确认。
- 不使用 `kill -9`、`pkill`、容器全量重启或目录清理作为默认排障手段。
- 不访问或回显秘密值。
- 不对无关文件做批量格式化。
- 不在没有验证证据时声称完成、修复或通过。

## 文档维护

- 架构、分支或开发状态发生实质变化时，同步更新 `PROJECT_CONTEXT.md`。
- AI 辅助阶段性工作记录到 `CHANGELOG_AI.md`，但正式产品版本仍应使用团队认可的 release/changelog 流程。
- `AGENTS.md` 只放长期规则；一次性任务细节放设计、计划或验收文档。
- 更新文档时保留“事实”和“假设”的区别，附核验日期、分支和关键证据。
