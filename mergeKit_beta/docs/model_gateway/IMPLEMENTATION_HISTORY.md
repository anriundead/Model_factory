# 模型服务 v1 文本推理纵切实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不影响现有融合/评测系统稳定性的前提下，交付一个可运行的“融合模型文本推理服务纵切”：管理员能发布并启动一个融合后的 Qwen 文本模型，用户能用 API Key 通过 OpenAI-compatible API 调用，系统能记录 usage，管理员能停止服务，重启后状态恢复为手动管理。

**Architecture:** 在现有 Flask 应用内新增 `app/model-gateway/` 模块，作为 Gateway 管理用户鉴权、OpenAI-compatible API、vLLM 子进程、usage 记录和管理员 API。vLLM 只监听 `127.0.0.1`，用户永远访问 Gateway；系统重启时只恢复 DB 状态，不自动加载模型抢占 GPU。

**Tech Stack:** Flask + Flask-SQLAlchemy + SQLite；vLLM OpenAI-compatible server；Python stdlib `subprocess`/`socket`/`secrets`/`hashlib`；`requests` 仅用于 Gateway 转发；现有 Docker Compose 服务 `mergekit-beta`。

## Global Constraints

- 第一版只做文本模型推理；VLM、文件解析、RAG、队列、门户高级动画和支付不进入本轮实现。
- 不修改 `merge_manager.py` 的 VLM/LLM 评测分支。
- 不修改 `HF_DATASETS_TRUST_REMOTE_CODE` / `--trust_remote_code` 评测逻辑。
- 不修改 `evolution/vendor/vlm_merge/run_vlm_search.py` 的 vLLM TP 子进程隔离。
- 不修改 Ray/vLLM 进化融合环境变量白名单，除非后续另起计划。
- 不覆盖当前脏工作区；新增改动必须集中在 serving 相关文件、最小注册点、测试和文档。
- vLLM 监听地址固定 `127.0.0.1`；用户不能直连 vLLM。
- 系统重启后模型服务不自动启动，管理员必须手动恢复。
- API Key 第一版长期保存哈希，不做支付扣费，不做密钥加密。
- 默认验收不启动真实 GPU/vLLM 模型；如需真实模型 smoke，必须先说明目标 GPU、预计显存、预计时长和清理方式，并获得人工确认。
- 用户已授权：当需要真实用户模拟验收时，可使用模型工厂融合进化功能先融合一个 7B 模型，再纳入 serving 流程进行端到端检查；该验收仍必须显式检查 GPU 空闲、预计耗时、停止方式和回滚方式，不进入默认单元测试或普通 smoke。

---

## Hardening Batch: Runtime Safety And No-GPU Acceptance

**Scope:** 只修复 serving 后端稳定性缺口，不启动真实 vLLM/GPU 模型，不修改融合、评测、Ray、vLLM TP 子进程隔离逻辑。

**Files allowed:**

- `mergeKit_beta/app/model_gateway/runtime.py`
- `mergeKit_beta/app/model_gateway/routes.py`
- `mergeKit_beta/tests/model_gateway/test_gateway_runtime.py`
- `mergeKit_beta/tests/model_gateway/test_gateway_routes.py`
- `mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md`

**Rollback:**

```bash
git restore mergeKit_beta/app/model_gateway/runtime.py \
  mergeKit_beta/app/model_gateway/routes.py \
  mergeKit_beta/tests/model_gateway/test_gateway_runtime.py \
  mergeKit_beta/tests/model_gateway/test_gateway_routes.py \
  mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md
```

If these files are still untracked in the local worktree, inspect them before removal and roll back only this batch's hunks.

**Acceptance commands:**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python - <<'PY'
import merge_manager
from app import app
import evolution.runner
import app.model_gateway.runtime
print('import-ok')
PY
docker compose ps
curl -fsS http://127.0.0.1:5000/healthz
curl -fsS http://127.0.0.1:5000/readyz
curl -fsS http://127.0.0.1:5000/api/models
curl -fsS http://127.0.0.1:5000/api/testset/list
curl -fsS http://127.0.0.1:5000/api/history
git diff --check -- docker-compose.yml mergeKit_beta/app/__init__.py mergeKit_beta/config.py mergeKit_beta/app/model_gateway mergeKit_beta/docs/model_gateway/ARCHITECTURE.md mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md mergeKit_beta/tests/model_gateway/test_gateway_*.py
docker top model_factory-mergekit-beta-1 aux
pgrep -af 'app.model_gateway.vllm_entrypoint|vllm serve|MERGEKIT_MODEL_GATEWAY_SERVICE_ID' || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
```

**Stop condition:** 任一验收失败，停止继续开发；只允许回滚本批次或修复本批次直接导致的问题。

---

## Frontend Portal Batch: Serving Portal V2

**Scope:** 新增嵌入式模型服务门户 `/model-gateway`，同时服务管理员和用户调用者。只实现前端页面、页面路由和导航入口，不启动真实 vLLM/GPU 模型，不新增计费系统。

**Design constraints:**

- 使用 `frontend-design` 作为主前端设计约束。
- 使用 `design-taste-frontend` 做视觉质量和 anti-slop 预检。
- 使用 `gsap-core` / `gsap-performance` 约束动效：只动画 `transform` 与 `opacity`，支持 reduced motion。
- 使用 `ponytail` 控制复杂度：沿用 Flask 静态模板，不引入 React/Vue/npm 构建链。

**Files allowed:**

- `mergeKit_beta/app/routes.py`
- `mergeKit_beta/templates/model_gateway/console.html`
- `mergeKit_beta/static/model_gateway/console.css`
- `mergeKit_beta/static/model_gateway/console.js`
- 旧模板中的工程菜单入口
- `mergeKit_beta/tests/model_gateway/test_gateway_portal.py`
- `mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md`

**Rollback:**

```bash
git restore mergeKit_beta/app/routes.py \
  mergeKit_beta/templates/index.html \
  mergeKit_beta/templates/evaluation.html \
  mergeKit_beta/templates/model_repo.html \
  mergeKit_beta/templates/test_history.html \
  mergeKit_beta/templates/testsets.html \
  mergeKit_beta/templates/fusion_history.html \
  mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md
rm -f mergeKit_beta/templates/model_gateway/console.html \
  mergeKit_beta/static/model_gateway/console.css \
  mergeKit_beta/static/model_gateway/console.js \
  mergeKit_beta/tests/model_gateway/test_gateway_portal.py
```

**Acceptance commands:**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
curl -fsS http://127.0.0.1:5000/healthz
curl -fsS http://127.0.0.1:5000/readyz
curl -fsS http://127.0.0.1:5000/model-gateway
git diff --check -- mergeKit_beta/app/routes.py mergeKit_beta/templates/model_gateway/console.html mergeKit_beta/static/model_gateway/console.css mergeKit_beta/static/model_gateway/console.js mergeKit_beta/tests/model_gateway/test_gateway_portal.py
```

**Stop condition:** 任一验收失败，停止继续开发；不通过启动真实模型或占用 GPU 来掩盖前端或路由问题。

---

## Request Cancellation Batch: User-Owned Cancel API

**Scope:** 新增用户侧请求取消能力，用于队列/后台任务恢复前的状态语义铺底。该批次不启动真实 vLLM/GPU 模型，不终止真实上游进程，不实现完整异步队列。

**Behavior:**

- `POST /v1/requests/<request_id>/cancel` 使用用户 API Key 鉴权。
- API Key 只能取消自己创建的请求；其他用户的请求返回 `404 request_not_found`。
- `pending`、`queued`、`paused_model_offline`、`retrying` 立即变为 `canceled`。
- `running`、`streaming` 变为 `cancel_requested`，由后续 worker/stream 逻辑尽力停止。
- `success`、`completed`、`failed`、`dead_letter`、`expired`、`canceled` 返回 `409 request_not_cancelable`。
- 门户 `/model-gateway` 提供轻量请求控制表单，用户输入 API Key 和 `request_id` 后调用取消 API。

**Files allowed:**

- `mergeKit_beta/app/model_gateway/routes.py`
- `mergeKit_beta/templates/model_gateway/console.html`
- `mergeKit_beta/static/model_gateway/console.css`
- `mergeKit_beta/static/model_gateway/console.js`
- `mergeKit_beta/tests/model_gateway/test_gateway_routes.py`
- `mergeKit_beta/tests/model_gateway/test_gateway_portal.py`
- `mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md`

**Rollback:**

```bash
git restore mergeKit_beta/app/model_gateway/routes.py \
  mergeKit_beta/templates/model_gateway/console.html \
  mergeKit_beta/static/model_gateway/console.css \
  mergeKit_beta/static/model_gateway/console.js \
  mergeKit_beta/tests/model_gateway/test_gateway_routes.py \
  mergeKit_beta/tests/model_gateway/test_gateway_portal.py \
  mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md
```

**Acceptance commands:**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_routes
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal
node --check mergeKit_beta/static/model_gateway/console.js
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
git diff --check -- mergeKit_beta/app/model_gateway/routes.py mergeKit_beta/templates/model_gateway/console.html mergeKit_beta/static/model_gateway/console.css mergeKit_beta/static/model_gateway/console.js mergeKit_beta/tests/model_gateway/test_gateway_routes.py mergeKit_beta/tests/model_gateway/test_gateway_portal.py mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md
```

**Stop condition:** 任一验收失败，停止继续开发；不通过杀进程、重启 Docker 或启动 GPU 服务来掩盖取消 API 的状态语义问题。

---

## Request Status Batch: User-Owned Status API

**Scope:** 新增用户侧请求状态查询能力，为 60 秒异步返回、取消请求和后续队列恢复提供最小闭环。该批次不实现队列 worker，不启动真实 vLLM/GPU 模型，不保存用户 prompt 正文。

**Behavior:**

- `GET /v1/requests/<request_id>` 使用用户 API Key 鉴权。
- API Key 只能查看自己创建的请求；其他用户的请求返回 `404 request_not_found`。
- 响应包含 `request` 状态摘要和最近一条 `usage` 摘要。
- 未记录 usage 时返回 `usage_source=not_recorded`，token 数为 0。
- 门户 `/model-gateway` 的请求控制区域提供轻量状态查询入口。

**Files allowed:**

- `mergeKit_beta/app/model_gateway/routes.py`
- `mergeKit_beta/templates/model_gateway/console.html`
- `mergeKit_beta/static/model_gateway/console.css`
- `mergeKit_beta/static/model_gateway/console.js`
- `mergeKit_beta/tests/model_gateway/test_gateway_routes.py`
- `mergeKit_beta/tests/model_gateway/test_gateway_portal.py`
- `mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md`
- `mergeKit_beta/docs/model_gateway/ARCHITECTURE.md`

**Rollback:**

```bash
git restore mergeKit_beta/app/model_gateway/routes.py \
  mergeKit_beta/templates/model_gateway/console.html \
  mergeKit_beta/static/model_gateway/console.css \
  mergeKit_beta/static/model_gateway/console.js \
  mergeKit_beta/tests/model_gateway/test_gateway_routes.py \
  mergeKit_beta/tests/model_gateway/test_gateway_portal.py \
  mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md \
  mergeKit_beta/docs/model_gateway/ARCHITECTURE.md
```

**Acceptance commands:**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_routes
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal
node --check mergeKit_beta/static/model_gateway/console.js
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
git diff --check -- mergeKit_beta/app/model_gateway/routes.py mergeKit_beta/templates/model_gateway/console.html mergeKit_beta/static/model_gateway/console.css mergeKit_beta/static/model_gateway/console.js mergeKit_beta/tests/model_gateway/test_gateway_routes.py mergeKit_beta/tests/model_gateway/test_gateway_portal.py mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md mergeKit_beta/docs/model_gateway/ARCHITECTURE.md
```

**Stop condition:** 任一验收失败，停止继续开发；不通过新增队列、启动 GPU 服务或扩大数据持久化范围来掩盖状态查询接口问题。

---

## Async Wait Batch: Non-Streaming 202 Request Lifecycle

**Scope:** 为 `stream=false` 的 Chat Completions 请求增加“最多等待 60 秒，否则返回 `202 + request_id`”的第一版语义。该批次使用轻量后台线程完成已提交请求，不引入 Redis Streams/worker，不保存用户 prompt 到 DB，不启动真实 vLLM/GPU 模型。

**Behavior:**

- 非流式请求创建 `serving_requests` 记录后进入 `running`。
- Gateway 默认最多等待 `MERGEKIT_MODEL_GATEWAY_SYNC_WAIT_SECONDS=60` 秒。
- 等待窗口内完成：同步返回 vLLM 原始 JSON，并写入 usage。
- 超过等待窗口：返回 `202 Accepted`，响应包含 `request_id`、`status_url`、`cancel_url`。
- 后台线程继续执行该请求，完成后写 `success`/`failed` 和 usage。
- 如果用户在后台执行期间取消请求，后台完成后把请求写为 `canceled`，并保留已产生的 usage。
- 测试使用极短等待配置模拟超时，不等待真实 60 秒。

**Files allowed:**

- `mergeKit_beta/app/model_gateway/routes.py`
- `mergeKit_beta/config.py`
- `mergeKit_beta/static/model_gateway/console.js`
- `mergeKit_beta/tests/model_gateway/test_gateway_routes.py`
- `mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md`
- `mergeKit_beta/docs/model_gateway/ARCHITECTURE.md`

**Rollback:**

```bash
git restore mergeKit_beta/app/model_gateway/routes.py \
  mergeKit_beta/config.py \
  mergeKit_beta/static/model_gateway/console.js \
  mergeKit_beta/tests/model_gateway/test_gateway_routes.py \
  mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md \
  mergeKit_beta/docs/model_gateway/ARCHITECTURE.md
```

**Acceptance commands:**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_routes
node --check mergeKit_beta/static/model_gateway/console.js
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
git diff --check -- mergeKit_beta/app/model_gateway/routes.py mergeKit_beta/config.py mergeKit_beta/static/model_gateway/console.js mergeKit_beta/tests/model_gateway/test_gateway_routes.py mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md mergeKit_beta/docs/model_gateway/ARCHITECTURE.md
```

**Stop condition:** 任一验收失败，停止继续开发；不通过启动真实模型或引入 Redis/队列系统来掩盖第一版等待语义问题。

---

## Restart Recovery Batch: In-Flight Background Requests

**Scope:** 补齐第一版内存后台线程在 Flask/Docker 重启后的状态收敛。该批次不实现 durable queue，也不保存或重放用户 prompt。

**Behavior:**

- `running` -> `failed`, `error_code=request_interrupted_by_restart`。
- `streaming` -> `failed`, `error_code=stream_interrupted_by_restart`。
- `cancel_requested` -> `canceled`, `error_code=canceled_by_restart`。
- `queued` 保持不变，因为当前纵切没有持久 worker 可以安全取回或执行它。
- 这与模型服务的“重启后管理员手动启动”策略并行执行，绝不自动占用 GPU。

**Files allowed:**

- `mergeKit_beta/app/model_gateway/runtime.py`
- `mergeKit_beta/app/__init__.py`
- `mergeKit_beta/tests/model_gateway/test_gateway_runtime.py`
- `mergeKit_beta/docs/model_gateway/ARCHITECTURE.md`
- `mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md`

**Acceptance commands:**

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_runtime.TestRestartRecovery
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
```

**Rollback:**

```bash
git restore mergeKit_beta/app/model_gateway/runtime.py mergeKit_beta/app/__init__.py \
  mergeKit_beta/tests/model_gateway/test_gateway_runtime.py \
  mergeKit_beta/docs/model_gateway/ARCHITECTURE.md \
  mergeKit_beta/docs/model_gateway/IMPLEMENTATION_HISTORY.md
```

---

## Runtime Acceptance Notes: 2026-07-12

- Docker Compose service `mergekit-beta` uses `init: true`. Tini reaps orphaned Ray descendants after a stopped evolution task; this is required because the Flask process is otherwise the container PID 1 and is not a process reaper.
- A real two-GPU 7B evolution smoke task (`54f979c8`) verified local MMLU loading, Ray worker binding to GPU `0,1`, model merging, and inference. It was intentionally stopped after 10 candidate evaluations.
- Do not use `n_iter=1` as a resource-bound smoke guarantee. In the current CMA-ES wrapper, `("n_iter", 1)` did not terminate after one generation. `max_evals` is the existing hard cap, but the runner deliberately serializes it to one GPU to prevent parallel overscheduling.
- Before the next two-GPU end-to-end serving smoke, implement and test a bounded parallel evaluation mode or explicitly approve the existing single-GPU `max_evals` path. Do not claim a bounded two-GPU smoke from `n_iter` alone.

### Natural-Termination Reproduction

- Reproduced on `2026-07-12` with task `cc2ce3b7`, using the same two 7B Qwen2 text models, cached `cais/mmlu` `college_medicine` validation split, `pop_size=2`, `n_iter=1`, `max_samples=4`, `ray_num_gpus=2`, and `skip_final_eval=true`.
- The runner correctly constrained Ray workers to GPU `0,1`; GPU `2` remained assigned to the external vLLM process and GPU `3` remained unused.
- Runtime introspection confirms pymoo maps `("n_iter", 1)` to `MaximumGenerationTermination(n_max_gen=1)`. Despite that, CMA-ES continued beyond 16 candidate evaluations in this integration. The task was manually stopped to avoid unbounded GPU and disk consumption.
- Both stopped smoke tasks left 9.3 GiB of candidate checkpoints under `vlm_search_results/merged_models`. Those checkpoints were removed; metadata, progress, config CSV, and logs were retained for diagnosis.
- This is a reproducible integration defect, not a model-quality result. A successful bounded two-GPU smoke requires a dedicated termination fix and an automated test before it may be used as a release gate.

### Natural Completion Confirmation

- A third identical run, task `eec970f3`, was allowed to finish without manual interruption. It completed successfully after `n_eval=20` / `step_count=20` in about 553 seconds.
- The task used GPU `1,3` because the availability selector chose the two cards with the most free memory. GPU `2` remained reserved for the external vLLM service; GPU `0` retained only the Flask CUDA baseline.
- The runner wrote `final_vlm`, copied it to `Qwen2.5-VL-7B-TextOnly_HuatuoGPT-Vision-TextOnly_20260712-132400`, created `output` as a symlink, wrote the recipe, and marked metadata `status=success`.
- The generated model is a 15 GiB `Qwen2ForCausalLM` directory and passes the serving model-path validator. Container vLLM version is `0.7.0`.
- Correct operational wording: `n_iter=1` starts a small CMA-ES optimization run, not a two-candidate limit. It naturally converged here at 20 evaluations, but it is not a deterministic resource cap. Keep `max_evals` for deterministic caps and use a separate acceptance profile when exact GPU parallelism is required.

---

## 当前代码事实

- 应用入口是 `mergeKit_beta/app/__init__.py:create_app()`。
- DB 使用 `app.extensions.db = SQLAlchemy()`，启动时优先 `flask_migrate.upgrade()`，迁移不可用时回退 `db.create_all()`。
- 当前仓库没有 `mergeKit_beta/migrations/` 目录，因此 v1 可先采用“新增 ORM 模型 + create_all 回退”的方式，后续再补 Alembic。
- 现有主路由集中在 `mergeKit_beta/app/routes.py:register_routes()`，没有 Blueprint 注册模式；v1 为降低主文件膨胀，应新增 serving Blueprint，并在 `create_app()` 中注册。
- 现有模型来源可复用：
  - `/api/merged_models`
  - `/api/model_repo/list`
  - `app.models.Model`
  - `Services.sync_models_db_from_disk()`
- 现有 GPU 快照可复用 `core.gpu_topology.query_gpus()`。
- 现有进程组启动参数可复用 `core.process_manager.ProcessManager.create_process_group_kwargs()`，但停止 vLLM 时必须校验服务标记，不能粗暴按名称杀进程。
- 当前工作区已有大量未提交变更和删除项；本计划不回滚、不清理这些变更。

## File Structure

Create:

- `mergeKit_beta/app/model_gateway/__init__.py`：提供 `register_model_gateway(app)`。
- `mergeKit_beta/app/model_gateway/models.py`：serving 相关 ORM 模型。
- `mergeKit_beta/app/model_gateway/auth.py`：API Key/Admin Token 认证与哈希。
- `mergeKit_beta/app/model_gateway/repository.py`：DB 查询和状态变更。
- `mergeKit_beta/app/model_gateway/runtime.py`：vLLM 命令构建、预检、启动、停止、健康检查、重启恢复。
- `mergeKit_beta/app/model_gateway/routes.py`：管理员 API 与 `/v1/*` OpenAI-compatible API。
- `mergeKit_beta/tests/model_gateway/test_gateway_auth.py`
- `mergeKit_beta/tests/model_gateway/test_gateway_runtime.py`
- `mergeKit_beta/tests/model_gateway/test_gateway_routes.py`

Modify:

- `mergeKit_beta/app/__init__.py`：导入 serving ORM 并注册 serving Blueprint；启动时执行 serving 状态恢复。
- `mergeKit_beta/config.py`：增加 serving 安全默认配置。
- `mergeKit_beta/docs/model_gateway/ARCHITECTURE.md`：补充 v1 实施状态和本计划链接。

Do not modify:

- `mergeKit_beta/merge_manager.py`
- `mergeKit_beta/evolution/vendor/vlm_merge/run_vlm_search.py`
- `mergeKit_beta/evolution/runner.py`
- `mergeKit_beta/app/services.py`，除非后续发现模型路径解析必须复用且无替代入口。

---

## Task 1: Serving 配置与 ORM

**Files:**
- Create: `mergeKit_beta/app/model_gateway/models.py`
- Modify: `mergeKit_beta/config.py`
- Modify: `mergeKit_beta/app/__init__.py`
- Test: `mergeKit_beta/tests/model_gateway/test_gateway_auth.py`

**Interfaces:**
- Produces ORM classes:
  - `ServingModelService`
  - `ServingApiKey`
  - `ServingRequest`
  - `ServingUsageRecord`
  - `ServingEvent`
- Produces config values:
  - `MERGEKIT_MODEL_GATEWAY_ENABLED: bool`
  - `MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN: str`
  - `MERGEKIT_MODEL_GATEWAY_VLLM_BIN: str`
  - `MERGEKIT_MODEL_GATEWAY_PORT_START: int`
  - `MERGEKIT_MODEL_GATEWAY_PORT_END: int`
  - `MERGEKIT_MODEL_GATEWAY_LOG_DIR: str`

- [ ] Add config defaults in `config.py`.

```python
MERGEKIT_MODEL_GATEWAY_ENABLED = (os.environ.get("MERGEKIT_MODEL_GATEWAY_ENABLED") or "1").strip().lower() not in ("0", "false", "no", "off")
MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN = (os.environ.get("MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN") or "").strip()
MERGEKIT_MODEL_GATEWAY_VLLM_BIN = os.environ.get("MERGEKIT_MODEL_GATEWAY_VLLM_BIN", "/opt/conda/envs/mergenetic/bin/vllm")
MERGEKIT_MODEL_GATEWAY_PORT_START = int(float(os.environ.get("MERGEKIT_MODEL_GATEWAY_PORT_START", "18000") or 18000))
MERGEKIT_MODEL_GATEWAY_PORT_END = int(float(os.environ.get("MERGEKIT_MODEL_GATEWAY_PORT_END", "18999") or 18999))
MERGEKIT_MODEL_GATEWAY_LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "serving")
```

- [ ] Ensure `Config.setup_environment()` creates `MERGEKIT_MODEL_GATEWAY_LOG_DIR`.

```python
os.makedirs(cls.MERGEKIT_MODEL_GATEWAY_LOG_DIR, exist_ok=True)
```

- [ ] Create serving ORM models with additive tables only.

Required columns:

```text
serving_model_services:
  id, model_id, model_path, display_name, served_model_name, model_type,
  backend_type, status, vllm_host, vllm_port, vllm_pid, vllm_pgid,
  gpu_ids, gpu_uuids, tensor_parallel_size, gpu_memory_utilization,
  dtype, max_model_len, max_num_seqs, max_num_batched_tokens,
  trust_remote_code, internal_api_key_hash, last_error, last_exit_reason,
  created_at, updated_at, started_at, stopped_at

serving_api_keys:
  id, key_hash, prefix, last4, owner_label, status, model_allowlist,
  notes, last_used_at, expires_at, created_at, updated_at

serving_requests:
  id, api_key_id, model_service_id, served_model_name, request_type,
  status, stream, idempotency_key, error_code, error_message,
  created_at, finished_at

serving_usage_records:
  id, request_id, api_key_id, model_service_id, served_model_name,
  prompt_tokens, completion_tokens, total_tokens, usage_source, created_at

serving_events:
  id, model_service_id, event_type, message, payload, created_at
```

- [ ] Import `app.model_gateway.models` inside `create_app()` before migrations/create_all.

```python
from .serving import models as serving_models  # noqa: F401
```

- [ ] Run existing test discovery.

```bash
cd /home/a/Workspace/Model_factory/mergeKit_beta
python -m unittest discover -s tests
```

Expected: existing tests pass, or failures match pre-change baseline.

---

## Task 2: API Key 与 Admin Token 认证

**Files:**
- Create: `mergeKit_beta/app/model_gateway/auth.py`
- Test: `mergeKit_beta/tests/model_gateway/test_gateway_auth.py`

**Interfaces:**
- Produces:
  - `generate_api_key() -> tuple[str, str, str, str]`
  - `hash_secret(secret: str) -> str`
  - `extract_bearer_token(header_value: str | None) -> str | None`
  - `require_admin_token(config) -> None`
  - `find_active_api_key(raw_key: str) -> ServingApiKey | None`

- [ ] Implement key format.

```text
User-facing key: mk_live_<32+ urlsafe chars>
Stored fields: sha256 hash, prefix="mk_live", last4
Admin token: Bearer token from MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN
```

- [ ] Test key hashing is deterministic and plaintext is never stored.

```python
def test_hash_secret_is_deterministic_and_not_plaintext():
    from app.model_gateway.auth import hash_secret
    digest = hash_secret("mk_live_example")
    assert digest == hash_secret("mk_live_example")
    assert digest != "mk_live_example"
    assert len(digest) == 64
```

- [ ] Test bearer parsing.

```python
def test_extract_bearer_token():
    from app.model_gateway.auth import extract_bearer_token
    assert extract_bearer_token("Bearer abc") == "abc"
    assert extract_bearer_token("bearer abc") == "abc"
    assert extract_bearer_token("abc") is None
    assert extract_bearer_token(None) is None
```

- [ ] Run focused tests.

```bash
cd /home/a/Workspace/Model_factory/mergeKit_beta
python -m unittest tests.model_gateway.test_gateway_auth
```

Expected: pass.

---

## Task 3: Runtime 预检、命令构建与重启恢复

**Files:**
- Create: `mergeKit_beta/app/model_gateway/runtime.py`
- Create: `mergeKit_beta/app/model_gateway/repository.py`
- Test: `mergeKit_beta/tests/model_gateway/test_gateway_runtime.py`

**Interfaces:**
- Produces:
  - `build_vllm_command(service: ServingModelService, config) -> list[str]`
  - `find_free_port(start: int, end: int, reserved: set[int]) -> int`
  - `validate_model_path(path: str, allowed_roots: list[str]) -> None`
  - `mark_services_stopped_after_restart(db_session) -> int`
  - `start_service(service_id: str) -> ServingModelService`
  - `stop_service(service_id: str, timeout_s: int = 30) -> ServingModelService`

- [ ] Implement model path validation.

Rules:

```text
path must exist
path must be a directory
path must contain config.json
path must contain at least one tokenizer file: tokenizer.json, tokenizer.model, vocab.json, merges.txt
path must contain at least one weight file: *.safetensors, *.bin
realpath must be under Config.MODEL_POOL_PATH, Config.LOCAL_MODELS_PATH, Config.MERGE_DIR, or Config.LOCAL_MODELS_EXTRA_PATHS
```

- [ ] Implement vLLM command builder.

Minimum command shape:

```bash
<MERGEKIT_MODEL_GATEWAY_VLLM_BIN> serve <model_path> \
  --host 127.0.0.1 \
  --port <port> \
  --served-model-name <served_model_name> \
  --api-key <internal_key> \
  --tensor-parallel-size <tensor_parallel_size> \
  --gpu-memory-utilization <gpu_memory_utilization> \
  --dtype <dtype> \
  --disable-log-requests
```

Only include optional arguments when non-empty:

```text
--max-model-len
--max-num-seqs
--max-num-batched-tokens
--trust-remote-code
```

- [ ] Start vLLM using a new process group.

Required behavior:

```text
env CUDA_VISIBLE_DEVICES=<comma-separated gpu_ids>
env MERGEKIT_MODEL_GATEWAY_SERVICE_ID=<service_id>
stdout/stderr append to logs/model-gateway/<service_id>.log
store vllm_pid and vllm_pgid
poll http://127.0.0.1:<port>/v1/models until healthy or timeout
on success: status=running
on failure: status=failed, last_error set, process group terminated
```

- [ ] Stop only the stored process group after service marker check.

Rules:

```text
If PID is missing or not alive: mark stopped.
If /proc/<pid>/environ exists and MERGEKIT_MODEL_GATEWAY_SERVICE_ID does not match: refuse stop and mark failed with manual_action_required.
If marker matches: terminate process group, wait up to timeout_s, then SIGKILL process group if needed.
Never kill by process name, port, or model path.
```

- [ ] Implement restart recovery.

At Flask startup:

```text
starting -> stopped, last_exit_reason=system_restarted_manual_recovery_required
running -> stopped, last_exit_reason=system_restarted_manual_recovery_required
stopping -> stopped, last_exit_reason=system_restarted_manual_recovery_required
failed/stopped/deleted unchanged
```

- [ ] Unit-test command builder and restart recovery without launching vLLM.

```python
def test_restart_recovery_marks_running_stopped(app):
    from app.model_gateway.runtime import mark_services_stopped_after_restart
    # create a running service, call recovery, assert status stopped
```

- [ ] Run focused tests.

```bash
cd /home/a/Workspace/Model_factory/mergeKit_beta
python -m unittest tests.model_gateway.test_gateway_runtime
```

Expected: pass.

---

## Task 4: 管理员 API

**Files:**
- Create: `mergeKit_beta/app/model_gateway/routes.py`
- Create: `mergeKit_beta/app/model_gateway/__init__.py`
- Modify: `mergeKit_beta/app/__init__.py`
- Test: `mergeKit_beta/tests/model_gateway/test_gateway_routes.py`

**Interfaces:**
- Produces endpoints:
  - `POST /api/model-gateway/admin/model-services`
  - `GET /api/model-gateway/admin/model-services`
  - `GET /api/model-gateway/admin/model-services/<service_id>`
  - `POST /api/model-gateway/admin/model-services/<service_id>/start`
  - `POST /api/model-gateway/admin/model-services/<service_id>/stop`
  - `POST /api/model-gateway/admin/api-keys`
  - `GET /api/model-gateway/admin/api-keys`

- [ ] Register serving Blueprint in app startup.

```python
from .serving import register_model_gateway
register_model_gateway(app)
```

- [ ] Require admin token for all `/api/model-gateway/admin/*` routes.

Behavior:

```text
If MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN is empty: return 503 serving_admin_token_not_configured.
If Authorization header is missing/wrong: return 401 unauthorized.
```

- [ ] Implement create model service.

Request body:

```json
{
  "model_id": "optional-existing-model-id",
  "model_path": "/absolute/path/to/model",
  "display_name": "Qwen merged demo",
  "served_model_name": "qwen-merged-demo",
  "gpu_ids": [0],
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.85,
  "dtype": "auto",
  "max_model_len": null,
  "max_num_seqs": 8,
  "trust_remote_code": false
}
```

Validation:

```text
served_model_name unique among non-deleted services
model_type fixed to text in v1
backend_type fixed to vllm
vllm_host fixed to 127.0.0.1
gpu_memory_utilization between 0.50 and 0.92
tensor_parallel_size <= len(gpu_ids)
```

- [ ] Implement create API key.

Request body:

```json
{
  "owner_label": "demo-user",
  "model_allowlist": ["qwen-merged-demo"],
  "notes": "temporary test key"
}
```

Response includes plaintext key once:

```json
{
  "status": "success",
  "api_key": "mk_live_...",
  "last4": "abcd"
}
```

- [ ] Test admin auth and service creation with Flask test client.

```bash
cd /home/a/Workspace/Model_factory/mergeKit_beta
python -m unittest tests.model_gateway.test_gateway_routes
```

Expected: pass.

---

## Task 5: OpenAI-compatible 用户 API 与 usage 记录

**Files:**
- Modify: `mergeKit_beta/app/model_gateway/routes.py`
- Modify: `mergeKit_beta/app/model_gateway/repository.py`
- Test: `mergeKit_beta/tests/model_gateway/test_gateway_routes.py`

**Interfaces:**
- Produces endpoints:
  - `GET /v1/models`
  - `POST /v1/chat/completions`

- [ ] Implement `/v1/models`.

Behavior:

```text
Requires valid user API key.
Returns only running model services allowed by that key.
Response follows OpenAI-style shape:
{"object":"list","data":[{"id":"qwen-merged-demo","object":"model","owned_by":"mergekit-beta"}]}
```

- [ ] Implement `/v1/chat/completions`.

Supported request fields in v1:

```text
model
messages
temperature
top_p
max_tokens
stream
stop
presence_penalty
frequency_penalty
```

Rejected fields:

```text
tools
tool_choice
response_format
audio
modalities
file_ids
images outside OpenAI vision content format
```

Routing:

```text
validate user API key
validate key allowlist contains model
find running service by served_model_name
forward request to http://127.0.0.1:<vllm_port>/v1/chat/completions
use internal vLLM key, never user key
```

- [ ] Record request and usage.

Non-streaming:

```text
create serving_requests row before forwarding
on vLLM success, copy response JSON unchanged
read response.usage.prompt_tokens/completion_tokens/total_tokens
create serving_usage_records with usage_source=vllm_response
mark request success
```

Streaming:

```text
proxy stream chunks as-is
create serving_requests row
after generator finishes, mark request success
if vLLM does not provide final usage, create usage row with zeros and usage_source=stream_usage_unavailable
```

- [ ] Test non-streaming proxy with mocked vLLM HTTP response.

Expected DB result:

```text
serving_requests.status == success
serving_usage_records.total_tokens == mocked total_tokens
ServingApiKey.last_used_at is not null
```

---

## Task 6: Startup Recovery 与系统验收

**Files:**
- Modify: `mergeKit_beta/app/__init__.py`
- Test: `mergeKit_beta/tests/model_gateway/test_gateway_runtime.py`

**Interfaces:**
- Consumes: `mark_services_stopped_after_restart(db.session)`

- [ ] Call restart recovery during `create_app()` after DB tables exist.

Required behavior:

```python
try:
    from .serving.runtime import mark_services_stopped_after_restart
    mark_services_stopped_after_restart(db.session)
except Exception as e:
    logging.getLogger("mergeKit_beta").warning("serving 状态恢复跳过: %s", e)
```

- [ ] Run full local unit tests.

```bash
cd /home/a/Workspace/Model_factory/mergeKit_beta
python -m unittest discover -s tests
```

Expected: pass, or only pre-existing baseline failures.

- [ ] Run import check.

```bash
cd /home/a/Workspace/Model_factory/mergeKit_beta
python -c "import merge_manager; from app import app; import evolution.runner; import app.model_gateway.runtime"
```

Expected: exit code 0.

---

## Task 7: Docker Smoke 与可选真实模型验证

**Files:**
- No source changes unless validation finds a defect.

- [ ] Rebuild only if Python dependencies changed. This v1 should not require new dependencies.

```bash
cd /home/a/Workspace/Model_factory
docker compose config --quiet
docker compose up -d --force-recreate mergekit-beta
```

- [ ] Verify runtime path and health.

```bash
docker compose exec -T mergekit-beta pwd
curl -fsS http://127.0.0.1:5000/healthz
curl -fsS http://127.0.0.1:5000/readyz
curl -fsS http://127.0.0.1:5000/api/models
curl -fsS http://127.0.0.1:5000/api/model_repo/list
```

Expected:

```text
pwd == /app/ServiceEndFiles/Workspaces/mergeKit_beta
all curl commands return 2xx
```

- [ ] Run container tests.

```bash
docker compose exec -T mergekit-beta python -m unittest discover -s tests
docker compose exec -T mergekit-beta python -c "import merge_manager; from app import app; import evolution.runner; import app.model_gateway.runtime"
```

Expected: pass, or only pre-existing baseline failures.

- [ ] Optional real GPU smoke, only when GPU is idle and a small local Qwen-compatible model is available.

Commands:

```bash
nvidia-smi
curl -fsS -H "Authorization: Bearer $MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d @/tmp/model-gateway-create-service.json \
  http://127.0.0.1:5000/api/model-gateway/admin/model-services
curl -fsS -H "Authorization: Bearer $MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN" \
  -X POST http://127.0.0.1:5000/api/model-gateway/admin/model-services/<service_id>/start
curl -fsS -H "Authorization: Bearer $USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"<served_model_name>","messages":[{"role":"user","content":"用一句话说明你是谁。"}],"max_tokens":32}' \
  http://127.0.0.1:5000/v1/chat/completions
curl -fsS -H "Authorization: Bearer $MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN" \
  -X POST http://127.0.0.1:5000/api/model-gateway/admin/model-services/<service_id>/stop
```

Expected:

```text
service reaches running
/v1/chat/completions returns valid JSON
usage row is created
stop returns service status stopped
no unrelated container exits
```

---

## Rollback

- New serving code only:

```bash
git restore mergeKit_beta/app/__init__.py mergeKit_beta/config.py mergeKit_beta/docs/model_gateway/ARCHITECTURE.md
rm -rf mergeKit_beta/app/model_gateway
rm -f mergeKit_beta/tests/model_gateway/test_gateway_auth.py mergeKit_beta/tests/model_gateway/test_gateway_runtime.py mergeKit_beta/tests/model_gateway/test_gateway_routes.py
```

- If a vLLM process remains after failed start:

```bash
ps -ef | grep MERGEKIT_MODEL_GATEWAY_SERVICE_ID
docker compose logs --tail=300 mergekit-beta
```

Only terminate a process after confirming its stored service marker or matching PID/PGID from `serving_model_services`.

- If DB additive tables need removal in local SQLite during development:

```sql
DROP TABLE IF EXISTS serving_usage_records;
DROP TABLE IF EXISTS serving_requests;
DROP TABLE IF EXISTS serving_events;
DROP TABLE IF EXISTS serving_api_keys;
DROP TABLE IF EXISTS serving_model_services;
```

Do not drop existing `models`, `tasks`, `testsets`, or evaluation tables.

## Acceptance Criteria

- Existing `/healthz`, `/readyz`, `/api/models`, `/api/model_repo/list` continue to pass.
- Admin can create a text model service record through `/api/model-gateway/admin/model-services`.
- Admin can create a long-lived user API key and receives plaintext only once.
- Admin can start the service; vLLM listens on `127.0.0.1:<auto-port>`.
- `/v1/models` lists only running allowed models.
- `/v1/chat/completions` supports a non-streaming OpenAI-style text request and returns vLLM output.
- Usage is recorded from vLLM response usage fields.
- Admin can stop the service without killing unrelated processes.
- After Flask/container restart, previously running services are marked `stopped` with manual recovery reason and do not auto-start.
- Unit tests and Docker smoke pass, or any failure is proven pre-existing and documented.

## Implementation Notes

- This plan intentionally does not add Redis. Synchronous `/v1/chat/completions` is enough for the first vertical slice.
- This plan intentionally does not add frontend. The backend contract must be stable before building the new portal.
- This plan reserves NewAPI-like fields (`owner_label`, `model_allowlist`, usage records) without implementing payment or quota deduction.
- This plan uses vLLM's OpenAI-compatible API instead of building tokenization and sampling logic in Flask.

## Self-Review

- Spec coverage: v1 objective is covered by Tasks 1-7. Broader design items such as VLM, file parsing, RAG, queues, billing and portal are intentionally excluded from v1.
- 占位扫描：未发现未解决的占位标记。
- Type consistency: service, key, request and usage names are consistent across tasks.
- Risk check: only additive DB tables and contained serving module are introduced; high-risk merge/eval/evolution files remain untouched.

---

## 2026-07-17: Model Publication Real Acceptance

- Added formal text/VLM publication acceptance through real HTTP, Transformers, CMMMU,
  vLLM, restart recovery, idempotency and deletion paths.
- Preserved the full recipe snapshot, recipe hash, ordered parents and resolved VLM base
  fingerprint in `publication_manifest.json`; recipe manifests now fail closed when these
  fields are absent.
- Reused the evolution pipeline's local CMMMU helpers for publication smoke validation,
  avoiding an undeclared runtime dependency on `lmms_eval`.
- Added a double-gated, default-off after-rename crash hook and verified restart recovery
  from `registration_pending` to one core model row and a completed Task.
- Verified Qwen text serving and token usage. Qwen2.5-VL publication is accepted, while
  service creation remains blocked under vLLM 0.7.0 with `unsupported_architecture`.
- Full evidence and retained asset paths are in
  [`ACCEPTANCE_20260717_MODEL_PUBLICATION.md`](ACCEPTANCE_20260717_MODEL_PUBLICATION.md).
- Final review follow-up introduced manifest schema 2 while preserving schema 1 assets,
  bound evolution, recipe publication and existing-model copies to shard/index hashes,
  and moved publication recipe diagnostics out of standard `metadata.json`.
- Publication GPU validation now resolves requested indexes to UUID and PCI bus ID,
  enforces disjoint allowed/protected UUID sets, and passes UUIDs to CUDA children.
- A second real VLM task (`4424f954`) and publication
  (`24c69e3ef8d549e3ae3e72e6e1a0a4bd`) verified the strengthened contract.
