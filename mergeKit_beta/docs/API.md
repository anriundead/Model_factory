# HTTP 接口层说明（mergeKit_beta）

本文描述 Flask 应用对外 HTTP 接口的**路径、方法与用途**，便于前端、自动化脚本与外部系统集成。实现入口：`app/routes.py` 中 `register_routes`。

## 约定

| 项 | 说明 |
|----|------|
| Base URL | 默认 `http://<host>:5000`，端口由环境变量 `PORT` 控制 |
| 内容类型 | 未特别说明的 JSON 接口使用 `Content-Type: application/json` |
| 鉴权 | 当前版本**无**统一 Token；生产环境建议由反向代理层做认证 |
| 响应习惯 | 多数接口返回 `{"status": "success"|"error", ...}`；错误时常伴 HTTP 4xx/5xx |
| 任务 ID | 创建类接口返回 8 位十六进制 `task_id`，与目录 `merges/<task_id>/` 一致 |

**分层约定**（见 `DEVELOPMENT.md`）：路由只做参数校验与编排；业务与持久化经 `app/services.py`，数据库写入经 `app/repositories/`。

## 进化融合并行与用卡对账

- **你提交的** `ray_num_gpus`：只是“希望并行用几张卡”。
- **系统实际用的**：以任务目录 `merges/<task_id>/metadata.json` 为准：
  - `ray_num_gpus_effective`：最终并行 worker 数（TP=1 时≈实际用卡数）。
  - `evolution_cuda_visible_devices`：子进程实际可见的 GPU 子集（需要裁剪时才会写）。
  - `ray_cap_reason`：为什么被裁剪（例如显存不足）。

一句话：**看 `ray_num_gpus_effective` + `evolution_cuda_visible_devices`，就能知道“实际用了几张卡、是哪几张卡”。**

---

## 健康与就绪

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/healthz` | 进程存活，不做重依赖检查 |
| GET | `/readyz` | 就绪探针：DB `SELECT 1` + `merges` 目录可写 |

---

## 页面（HTML）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 主页 |
| GET | `/evaluation` | 评估页 |
| GET | `/testsets` | 测试集页 |
| GET | `/test_history` | 测试历史页 |
| GET | `/model_repo` | 模型仓库页 |
| GET | `/fusion_history` | 进化融合历史页 |
| GET | `/static/<path:filename>` | 静态资源 |

管理后台：`/admin`（Flask-Admin）。

---

## 模型与路径

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/models` | 基座模型列表（扫描 `LOCAL_MODELS_PATH` 等） |
| POST | `/api/models/delete` | 按路径删除模型目录（危险操作，需确认业务含义） |
| GET | `/api/models_pool` | 融合模型池列表 |
| GET | `/api/merged_models` | 已融合模型列表 |
| POST | `/api/check_compatibility` | 提交多路径，检查是否可融合 |
| GET | `/api/resolve_model_path?name=` 或 `path=` | 名称/相对路径 → 绝对路径 |
| GET | `/api/model_is_vlm` | 查询某路径是否为 VLM（参数见实现） |

---

## 融合与评估任务

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/merge` | 标准融合；body：`custom_name` 必填，`model_paths` / `models` / `items` 三选一；可选 `dataset_type`、`dataset_subset`、`priority` 等 |
| POST | `/api/merge_evolutionary` | 进化融合；至少两模型路径或 `items`；字段含 `eval_mode`、`hf_dataset`、`hf_subsets`、`pop_size`、`n_iter`、`ray_num_gpus`、`max_evals`、`skip_final_eval` 等，详见 `merge_manager` 写入的 `metadata.json` |
| POST | `/api/merge_evolutionary_check` | 提交路径或 `items`，返回 `compatible` / `reason` / `model_types` |
| POST | `/api/evaluate` | 评估任务；`model_path` 必填，可选 `testset_id`、`hf_dataset`、`hf_subset`、`hf_split` 等 |

成功时通常返回：`{"status":"success","task_id":"xxxxxxxx"}`。

**进化任务 Ray 并行**：Runner 可能根据显存将请求的 `ray_num_gpus` 降为 `ray_num_gpus_effective` 并设置 `evolution_cuda_visible_devices`，见 `evolution/contracts.md` 与任务目录 `metadata.json`。

---

## 任务状态与历史

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/history` | 融合侧历史列表（DB 优先，文件回退，逻辑在 `services`） |
| GET | `/api/history/<task_id>` | 读取 `merges/<task_id>/metadata.json` |
| DELETE | `/api/history/<task_id>` | 删除整个任务目录 |
| GET | `/api/status/<task_id>` | **核心轮询接口**：内存队列状态；进化任务合并 `progress.json`；磁盘 `metadata.json` 成功/失败时覆盖内存态 |
| POST | `/api/stop/<task_id>` | 停止任务（含杀子进程树） |
| POST | `/api/resume/<task_id>` | 恢复被中断任务（`status==interrupted`） |

---

## 模型仓库 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/model_repo/list` | `base_models` + `merged_models` |
| GET | `/api/model_repo/sync` | 触发同步视角下的统计（列表条数 vs DB `models` 条数） |
| GET | `/api/model_repo/<model_id>/path` | 解析模型 ID 对应路径 |
| DELETE | `/api/model_repo/<model_id>` | 删除仓库条目及关联（实现见 routes） |

---

## 测试集

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/testset/list?refresh=0|1` | 测试集列表 |
| GET | `/api/testset/search?q=&limit=` | 搜索 |
| GET | `/api/testset/<testset_id>?refresh=0|1` | **当前生效行为**以 `routes.py` 中**后注册**的处理器为准：返回 `get_testset_by_id` + `load_leaderboard()` 中该测试集榜单（文件中同路径存在两处注册，后者覆盖前者） |
| POST | `/api/testset/create` | 从 HF 数据集创建测试集，双写 DB + `testset_repo/data/testsets.json` |
| POST | `/api/testset/cmmmu/scan` | 批量补全 CMMMU 子集条目 |
| POST | `/api/testset/dedup` | 测试集去重（body 见实现） |
| GET | `/api/mmlu_subset_groups` | MMLU 分组定义 |
| GET | `/api/cmmmu_subset_groups` | CMMMU 分组定义 |
| GET/POST | `/api/hf/datasets/search` | Hub 数据集搜索，`q`、`limit` |
| POST | `/api/dataset/hf_info` | 数据集元信息探测 |
| POST | `/api/dataset/refresh_cache` | 刷新数据集相关缓存 |

---

## 测试历史与评估展示

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/test_history` | 评估历史列表 |

---

## 配方

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/recipes` | `recipes/*.json` 列表 |
| GET | `/api/recipes/<recipe_id>` | 单个配方 |
| POST | `/api/recipes/apply` | body：`recipe_id`，可选 `custom_name`、`priority` |

---

## 搜索与进化可视化

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/search?q=` | 搜索配方、任务 metadata、模型仓库 |
| GET | `/api/fusion_history` | 仅 `type=merge_evolutionary` 的任务列表（扫磁盘 metadata） |
| GET | `/api/fusion_3d_data/<task_id>` | 3D 图用进化数据 |
| GET | `/api/evolution_steps/<task_id>` | DB/API 进化步数据 |
| PUT | `/api/evolution_steps/<task_id>/<int:step_id>` | 更新单步 |
| POST | `/api/evolution_steps/<task_id>/sync` | 与 CSV/文件同步 |

---

## 与数据文档的关系

- 任务参数与结果字段以 **`merges/<task_id>/metadata.json`** 为任务级真相来源之一；DB 表 `tasks` 为索引与列表加速，见 [`DATABASE.md`](DATABASE.md)。
- 进化契约（进度文件、Ray 裁剪）：[`evolution/contracts.md`](../evolution/contracts.md)。

---

## 代码锚点

- 路由注册：`app/routes.py` → `register_routes`
- 应用工厂：`app/__init__.py` → `create_app`
