# AI 辅助开发记录

> 本文件记录 AI 辅助分析、设计、实现和验收状态，不替代正式产品 CHANGELOG、Git 历史或发布说明。
> 最后更新：2026-07-18（Asia/Shanghai）

## 2026-07-18：Gateway 上下文上限与 vLLM 启动失败修复

- 根因：文本服务创建时前端未提交 `max_model_len`，vLLM 继承 Qwen 模型的 128K 默认值，单卡 KV Cache 只能容纳约 102K tokens，启动因此退出。
- 修复：管理员服务表单提交文本 64K/VLM 16K；后端对遗漏参数应用安全默认值并拒绝超过首发上限的值。
- 修复：已知 KV Cache 启动失败写入脱敏、可操作的 `last_error`，不把 vLLM 日志正文写入数据库。
- 回归：主工作树容器 `unittest` 为 `361` 项通过；Gateway runtime 重点测试、前端 harness、JS 语法和 `git diff --check` 通过。
- 待完成：主容器重建后使用 GPU 0 做一次真实 7B 启动和 OpenAI-compatible 对话验收；测试结束停止服务并撤销临时 Key。

## 当前版本状态

- 仓库没有语义化版本号。
- 当前主工作区：`checkpoint/20260715-platform-gateway`；模型发布流水线源码合并点为 `2fc0406`。
- 远端同名 checkpoint：`010104f`；当前 HEAD 和领先数量应在每次新会话中重新核验。
- 最近一个有完整真实运行验收记录的里程碑：Model Publication Real Acceptance，2026-07-17。
- `feature/model-publication-pipeline` HEAD `61547bf` 已由 merge commit `2fc0406` 接入当前 checkpoint；本地 feature 分支和 worktree 已在验收证据迁移后清理。
- 发布计划 Task 1-7 已完成并合并；当前常驻 5000 容器启动于 merge 前，尚未重建到新的 Python 代码和 `/data/PublishedModels` mount。

## 2026-07-17：项目接管文档

### 新增

- `PROJECT_CONTEXT.md`：仓库概览、技术栈、架构、目录、数据流、数据库、开发状态、问题、Git 历史和路线图。
- `AGENTS.md`：根目录级 Agent 规则、修改授权、架构边界、测试和部署要求。
- `CHANGELOG_AI.md`：AI 辅助开发状态和后续计划。

### 核验范围

- 检查仓库结构、现有 README/开发/架构/数据库/API/验收文档。
- 检查 Conda、pip、Maven、Dockerfile、Compose、环境变量名和忽略规则。
- 检查核心 Flask、ORM、任务 Worker、融合/评测、Gateway 和研究源码。
- 检查测试文件、迁移文件、CI 配置和 Git 分支/提交历史。
- 只读检查运行中的 Compose 服务、健康端点和数据库计数。

### 运行快照

- 6 个 Compose 服务运行中；健康检查正常。
- `/healthz` 和 `/readyz` 返回 HTTP 200。
- 当前合并后分支包含 358 个 Python 测试方法、2 个 JavaScript harness 和 2 个 Java/JUnit 测试。
- 2026-07-15 Gateway 验收历史记录 173 个容器测试通过；2026-07-17 publication feature 验收记录完整容器测试 `358/358` 通过。
- merge commit `2fc0406` 上已重新运行完整 `unittest`，结果为 `358/358`；Compose、HTTP smoke、JS 语法、Git diff 和 GPU/进程检查通过。
- 没有发现 CI/CD workflow。

### 识别出的主要风险

- 核心 API 和 Flask-Admin 没有统一鉴权。
- Flask 开发服务器监听 `0.0.0.0` 且 `debug=True`。
- 核心 schema 缺少完整版本化迁移。
- 核心 DB/文件双写结果稀疏，需要一致性审计。
- `/api/testset/<testset_id>` 存在重复 GET 路由。
- 核心模块体积过大且任务调度依赖单进程内存。
- publication feature 已合并并完成 merge commit 非 GPU 回归，但当前 5000 容器尚未重建；仍需部署配置核对和部署后验收。

## 近期已完成方向

### 2026-07-15：Gateway Stage 1 内部试点

相关提交：`010104f`、`cb2c6de`。

- 建立管理员控制的 vLLM 服务生命周期和 OpenAI-compatible API。
- 建立 API Key、model allowlist、请求和 usage 记录。
- 建立上传文件和公开 URL 的研究流程。
- 增加 PostgreSQL 权威状态、Redis Streams 交付、ClamAV 和隔离 Java 解析器。
- 增加 SSRF 防护、配额、并发限制、每日导入限制和 24 小时 TTL。
- 增加 Gateway 专用 Alembic baseline。
- 真实 PDF、网页、BGE-M3 检索、7B vLLM 和引用验证验收通过。

### 2026-07-16 至 2026-07-17：模型发布流水线 Task 1-7

实现阶段位于 `feature/model-publication-pipeline`；该分支随后由 `2fc0406` 合并到当前 checkpoint。以下条目描述其实现内容。

- 结构化检测 text/VLM 模型及 Qwen processor 元数据。
- 在配方中持久化 VLM base provenance。
- 增加严格的 VLM language-weight composition。
- 增加独立发布目录、manifest、文件 hash、原子 rename 和启动恢复。
- 增加发布任务、API、幂等、取消和显式 GPU 验证。
- 增加核心模型删除保护和 Gateway 正式资产绑定。
- 增加管理员发布状态和兼容性 UI。
- 多轮修复停止/删除竞态、GPU 解析、生命周期和 Firefox/UI 溢出问题。

### 2026-07-17：模型发布流水线真实验收

验收提交：`61547bf`；证据文件：feature 中的 `ACCEPTANCE_20260717_MODEL_PUBLICATION.md`。

- 在隔离 Compose project 和回环端口 5057 上执行，不替换当前常驻 5000 服务。
- 文本模型完成正式发布、Gateway service、真实 vLLM 调用和非零 token usage 记录。
- VLM 完成标准进化、schema 2 正式发布、Transformers 图像推理和真实 CMMMU 功能样本。
- 完成来源 shard/index 指纹、recipe SHA/snapshot、有序父模型和 VLM 基座 provenance 约束。
- 完成显式 allowed/protected GPU UUID 门禁，保护 GPU UUID、bus ID 和显存基线保持不变。
- 完成 rename 后受控崩溃恢复、幂等注册和 `asset_in_use` 删除保护。
- 完整容器测试 `358/358`、8 个 HTTP smoke、关键 import、JS 语法和 diff 检查通过。
- 最终无 vLLM、Ray 或 headless browser 进程，控制目录为空。
- 保留正式文本资产、schema 2 VLM 资产和配方 `4424f954.json`；旧 schema 1 VLM 资产归档而未删除。
- Qwen2.5-VL 在 vLLM 0.7.0 下仍为 `blocked/unsupported_architecture`，VLM 在线服务未宣称完成。

### 2026-07-17：模型发布流水线合并

- merge commit：`2fc0406`（父提交 `79de55a`、`61547bf`）。
- 当前 checkpoint 已包含验收过的 publication 源码、测试、运维文档和配方。
- merge commit 上完整 `unittest` 为 `358/358`，8 个 HTTP smoke、Compose、JS、Git diff 和 GPU/进程门禁通过。
- 原始 Task 6/7 验收资料已迁移到仓库根 `logs/model_gateway/acceptance/`；原 feature worktree 和本地分支已清理。
- 合并未自动重建已运行 34 小时的主容器；当前容器没有新的 `/data/PublishedModels` mount。
- 合并完成不等于部署完成，必须在批准窗口执行配置核对、容器重建和部署验收。

## 待处理工作

### 最高优先级

- 配置正式 published models 挂载以及互不重叠的 protected/allowed GPU UUID。
- 制定当前常驻 5000 服务的受控升级与回滚步骤。
- 明确 checkpoint 后续同步到 `master`/`main` 和远端同名分支的策略。
- 确认 5000 端口网络边界；若有不可信访问，优先隔离核心管理面。

### 安全与生产化

- 为核心 API、页面和 Flask-Admin 增加认证/授权。
- 关闭生产 debug，采用受控 WSGI、反向代理和 TLS。
- 为危险操作增加权限、确认、审计和回归测试。
- 保持 Gateway vLLM loopback-only 和管理员手动生命周期。

### 质量与数据

- 建立 Python、Java、Compose 和 JS 的最小 CI。
- 审计 `tasks`、`evaluation_results`、`evolution_steps` 和文件产物一致性。
- 建立核心数据库版本化迁移方案。
- 消除重复测试集路由并固定 API 契约。
- 按用例逐步收敛超大模块，不做一次性重写。

## 未来计划

- 独立评估支持 Qwen2.5-VL 的 vLLM 版本或替代服务后端。
- 实施系统健康监控和自动发现计划。
- 在明确单一调度所有权后评估多进程或分布式任务队列。
- 改善离线前端资源、可观测性、模型版本管理和结果对比。

## 维护规则

- 每条记录注明日期、分支/提交、事实证据和未验证项。
- 未真实运行的功能写“已实现代码”或“待验收”，不能写“已完成”。
- 不记录 secret、连接串、API Key、用户资料正文或私有模型内容。
- 正式发布版本、breaking change 和数据库迁移应另行维护团队认可的产品 CHANGELOG。
