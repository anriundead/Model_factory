# 模型资产发布与 VLM 物化管线设计

> 日期：2026-07-16
> 状态：设计已确认，等待实施计划
> 范围：融合配方、模型物化、正式发布资产、Gateway 服务创建、管理员与用户可见性
> 不包含：vLLM 版本升级、支付计费、自动启动模型服务、现有历史模型批量迁移

## 1. 背景与事实基线

当前系统已经具备模型融合、CMMMU 多模态评测、模型仓库和 Gateway 服务管理，但“融合完成”和“可供用户调用”之间缺少可信的正式模型资产层。

实际代码与数据检查得到以下结论：

- VLM 搜索会加载完整 VLM，并把融合后的语言权重注入其中执行 CMMMU，因此现有 CMMMU 分数来自真实多模态推理。
- 当前搜索最终保存的仍是文本模型；视觉塔没有被写入长期输出目录。
- 现有配方没有稳定记录视觉基座、revision 和文件指纹。
- 现有 45 个融合输出均为文本模型，即使部分目录名包含 `VL` 或 `Vision`。
- 目录名启发式会把 TextOnly 模型误判为 VLM，正式发布不能复用该判断顺序。
- Gateway 创建服务时允许客户端提交模型路径，并硬编码 `model_type="text"`。
- Gateway 运行时允许目录尚未包含独立发布目录。
- 现有模型删除接口没有检查 Gateway 服务引用，可能删除仍被服务使用的模型。
- 当前 Transformers `5.3.0` 可以加载 Qwen2.5-VL；当前 vLLM `0.7.0` 不支持 `Qwen2_5_VLForConditionalGeneration`。

因此，完整 Qwen2.5-VL 可以被融合、评测和发布为模型资产，但在当前运行时下不能声明为可提供 vLLM 服务。

## 2. 目标与非目标

### 2.1 目标

1. 文本模型和 VLM 使用同一套正式发布机制。
2. VLM 发布资产保留完整视觉塔、processor、图像 token 和融合后的语言权重。
3. 搜索阶段继续以配方为主要长期成果，管理员明确发布时才占用长期模型空间。
4. 正式资产可验证、可追溯、可恢复，不依赖父模型软链接。
5. 模型资产发布与 Gateway 服务上线严格分离。
6. 管理员能看到不兼容原因，用户只看到实际运行且有权限使用的服务。
7. 删除、崩溃恢复和真实验收不影响 GPU 2 或设备上的外部任务。

### 2.2 非目标

- 本轮不升级 vLLM、CUDA、Transformers 或 NVIDIA 驱动。
- 本轮不让 Qwen2.5-VL 绕过 vLLM 兼容检查进入用户门户。
- 本轮不批量复制或迁移现有 45 个历史融合输出。
- 本轮不自动启动或自动恢复模型服务。
- 本轮不实现任意外部模型供应商或 One API 上游。
- 本轮不补齐与发布闭环无关的 Gateway PATCH、restart 和 force-stop 接口。

## 3. 方案选择

### 3.1 采用：独立通用发布层

新增统一发布目录：

```text
宿主机：/home/a/Model_factory_data/published_models/
容器内：/data/PublishedModels/
```

Compose 增加独立 bind mount，配置增加明确的发布目录变量。文本和 VLM 都发布到此目录，每个正式资产都包含标准 Hugging Face 文件和 `publication_manifest.json`。

采用该方案的原因：

- 不把长期服务资产与搜索临时输出混在 `merges/`。
- 不依赖 Gateway 在创建服务时临时复制或改造模型。
- 文本和 VLM 共用一套注册、校验、删除和恢复逻辑。
- 回滚应用代码时，正式模型资产仍保留在独立数据目录。

### 3.2 不采用：继续使用 `merges/`

`merges/` 当前同时包含任务目录、软链接、历史产物和临时结果，不适合作为对外服务资产的稳定边界。

### 3.3 不采用：由 Gateway 创建服务时物化模型

Gateway 应只管理服务，不应承担融合、视觉塔组装或大型文件复制。否则一次“创建服务”会同时改变 GPU、数据库和几十 GB 文件，失败边界不可控。

## 4. 权威数据边界

| 层级 | 权威内容 | 不负责 |
|---|---|---|
| 融合配方 | 如何重新生成模型、父模型顺序、融合参数、评测结果 | 服务运行状态 |
| `publication_manifest.json` | 正式资产来源、文件完整性、能力、验证和运行时兼容性 | GPU、端口和用户权限 |
| 核心 `models` 表 | 模型资产索引、ID、路径、架构、来源 | 详细配方和 vLLM 进程 |
| Gateway `serving_model_services` | 服务名称、GPU、端口、vLLM 参数和运行状态 | 模型物化和配方执行 |
| API Key allowlist | 允许用户调用的 `served_model_name` | 模型资产完整性 |

模型资产发布和模型服务上线是两个不同操作：

1. 模型工厂执行“物化并发布模型资产”。
2. Gateway 管理员执行“创建模型服务”。
3. 管理员显式启动服务后，用户才可能看到模型。

## 5. 状态模型

### 5.1 发布任务状态

```text
materializing -> validating -> registration_pending -> published
       |              |                  |
       +-----------> failed <------------+
```

- `materializing`：生成或复制标准模型文件。
- `validating`：执行结构、文件和真实功能验证。
- `registration_pending`：完整目录已原子发布，核心模型表尚未确认注册。
- `published`：资产验证和核心注册均完成。
- `failed`：任务失败；保留小型诊断记录，清理大型 staging 文件。

发布任务复用核心 SQLite 中现有的 `Task` 模型，新增
`task_type="model_publication"`，不新建发布任务表。任务执行期间的状态、错误码、
`publication_id` 和幂等键记录在现有任务配置/结果中；只有进入原子提交阶段后才生成
正式 manifest。

### 5.2 服务兼容性

```text
ready | blocked | stale
```

- `ready`：当前记录的服务后端版本通过架构预检，可以创建服务。
- `blocked`：明确不支持，必须记录稳定错误码和原因。
- `stale`：运行时版本与上次验证版本不同，必须重新检查。

兼容性不是发布生命周期。一个模型可以是 `published`，同时是 `blocked`。

### 5.3 Gateway 服务状态

继续沿用：

```text
stopped -> starting -> running -> stopping -> stopped
                    \-> failed
stopped/failed -> deleted
```

系统重启后不自动启动模型服务。原 `starting`、`running`、`stopping` 服务恢复为 `stopped`。

## 6. 配方契约

### 6.1 向后兼容原则

- 现有配方字段不删除、不重命名。
- 当前 runner 使用的 `vlm_path` 保留。
- 新配方增加结构化 `vlm_base`，用于准确追踪视觉外壳。
- 旧配方成功解析视觉基座后，只做原子、增量写回，保留全部原字段。
- 写回失败不影响已经产生的正式 manifest 快照，但必须记录告警。

### 6.2 新增字段

```json
{
  "recipe_schema_version": 2,
  "artifact_type": "vlm",
  "capabilities": ["text_generation", "vision_language"],
  "vlm_path": "/data/Models/Qwen2.5-VL-7B-Instruct",
  "vlm_base": {
    "parent_index": 0,
    "model_id": "source-model-id",
    "source_path": "/data/Models/Qwen2.5-VL-7B-Instruct",
    "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "config_sha256": "...",
    "model_type": "qwen2_5_vl",
    "architectures": ["Qwen2_5_VLForConditionalGeneration"],
    "processor_class": "Qwen2_5_VLProcessor"
  }
}
```

文本配方使用 `artifact_type="text"`，不要求 `vlm_base`。

### 6.3 VLM 基座选择

按以下顺序选择：

1. 管理员显式指定的完整 VLM 父模型。
2. 配方中已经记录且指纹一致的 `vlm_base`。
3. 按有序 `model_paths` 选择第一个结构完整的 VLM。
4. 没有完整 VLM 时，在任务启动前以 `vlm_base_missing` 失败。

结构判断必须先读 `config.json` 和权重索引，再参考名称。完整 VLM 至少具备：

- 视觉架构配置；
- 视觉塔权重；
- processor 和 tokenizer；
- 图像 token 定义；
- 与待融合语言模型兼容的层数、隐藏维度、词表和目标 tensor shape。

禁止因为路径名包含 `VL`、`Vision` 或类似字符串就判定为 VLM。

## 7. 发布清单契约

每个正式目录必须包含一个 `publication_manifest.json`：

```json
{
  "schema_version": 1,
  "publication_id": "...",
  "display_name": "...",
  "artifact_type": "vlm",
  "capabilities": ["text_generation", "vision_language"],
  "publication_state": "published",
  "provenance": {
    "task_id": "...",
    "recipe_path": "...",
    "recipe_sha256": "...",
    "recipe_snapshot": {},
    "parents": [],
    "vlm_base": {}
  },
  "model": {
    "model_type": "qwen2_5_vl",
    "architectures": ["Qwen2_5_VLForConditionalGeneration"],
    "tokenizer_class": "...",
    "processor_class": "Qwen2_5_VLProcessor",
    "dtype": "bfloat16"
  },
  "files": {
    "hash_algorithm": "sha256",
    "total_bytes": 0,
    "entries": []
  },
  "validation": {
    "structural": {},
    "functional": {},
    "evaluation": {}
  },
  "compatibility": {
    "transformers": {},
    "lm_eval": {},
    "lmms_eval": {},
    "serving": {
      "backend": "vllm",
      "tested_version": "0.7.0",
      "status": "blocked",
      "reason_code": "unsupported_architecture",
      "reason": "Qwen2_5_VLForConditionalGeneration is not supported by vLLM 0.7.0"
    }
  },
  "timestamps": {
    "created_at": "...",
    "validated_at": "...",
    "published_at": "..."
  }
}
```

规则：

- 文件哈希覆盖所有模型文件，不包含 manifest 本身。
- manifest 更新使用同目录临时文件、`fsync` 和原子替换。
- 日志引用使用相对任务路径，不把用户输入或密钥写入 manifest。
- `recipe_snapshot` 保存发布时实际使用的完整配方，避免源配方后续变化破坏追溯。
- 运行时版本变化后，服务兼容性变为 `stale`，重新验证后再原子更新。

## 8. 物化与原子发布

### 8.1 目录

```text
/data/PublishedModels/.staging/<publication_id>/
/data/PublishedModels/<publication_id>/
/data/PublishedModels/.trash/<publication_id>/
```

staging 和正式目录必须位于同一文件系统，以保证目录重命名原子性。

### 8.2 预检

任务启动前完成：

1. 配方字段和父模型存在性检查。
2. 父模型真实路径和指纹检查。
3. VLM 视觉外壳结构检查。
4. 语言权重层数、hidden size、词表和 tensor shape 检查。
5. 磁盘空间检查。

最低可用空间：

```text
预计正式模型大小 + max(5 GiB, 预计大小的 10%)
```

原子重命名不会再复制一份 staging，因此不按两倍模型大小计算。

### 8.3 文本模型

沿用现有配方应用逻辑，将结果写入 staging。输出必须包含 config、tokenizer、权重和模型加载所需的标准文件。

“发布已有模型”会复制完整模型到 staging，不直接把历史目录登记为正式资产。

### 8.4 VLM

1. 按配方生成融合后的语言权重。
2. 复制选定 VLM 的完整视觉外壳到 staging。
3. 复用当前 CMMMU 推理已经验证的权重映射规则，将语言权重写入完整 VLM。
4. 保留视觉塔、processor、图像 token 和多模态配置。
5. 保存为不依赖父模型软链接的标准 Hugging Face 模型。

不能按简单字符串前缀盲目替换权重。每个目标 tensor 必须校验名称、shape 和 dtype。缺失、多余或尺寸不一致均以 `architecture_mismatch` 失败。

### 8.5 验证

结构验证必须检查：

- `AutoConfig`、tokenizer 和 processor 可加载；
- safetensors 索引引用的分片全部存在；
- 文件清单和 SHA-256 完整；
- VLM 同时存在语言和视觉权重；
- 没有指向 staging 或父模型的软链接。

功能验证必须走真实模型路径：

- 文本模型执行一次短文本推理；
- VLM 使用真实图片执行一次 Transformers 多模态推理；
- VLM 再执行一个极小规模的真实 CMMMU 样本。

没有获准使用的空闲 GPU 时，任务保持 `validating`，不降低标准、不使用 GPU 2、不注册为正式资产。

### 8.6 原子提交

1. staging 完成写入和验证。
2. 生成状态为 `registration_pending` 的 manifest。
3. `fsync` 关键文件和目录。
4. 原子重命名为正式目录。
5. 注册核心模型表，使用 `source="published"`。
6. 原子更新 manifest 为 `published`。

数据库注册失败时保留完整正式目录。启动扫描器重新验证并补注册，不能重新物化一份重复模型。

核心模型 SQLite 与 Gateway PostgreSQL 之间不建立跨库外键，也不伪造跨库事务。
以下短操作统一获取 `/data/PublishedModels/.publication.lock` 的排他 `flock`：

- 发布任务的幂等检查和最终目录提交；
- Gateway 创建服务前的资产复检与服务记录写入；
- 正式资产删除前的服务引用复检与移入 `.trash`。

模型复制、权重写入和真实推理验证不持有该锁。这样既消除“检查后立即被删除/创建”的
竞态，也不会让数十分钟的模型任务阻塞管理员读取或其他任务预检。

## 9. 启动恢复

启动扫描器只恢复资产注册，不启动模型服务：

- 无活跃任务的残留 staging：保存小型诊断记录后删除大型文件。
- `registration_pending` 且验证通过：补注册核心模型表。
- 正式目录已注册：幂等跳过。
- 数据库记录存在但目录缺失：标记不可用并告警，不静默删除记录。
- manifest 损坏或文件指纹不符：隔离资产，禁止创建或启动服务。
- 运行时版本变化：兼容性标记为 `stale`。

扫描器以 `publication_id` 和正式真实路径作为幂等键。

## 10. Gateway 对接

### 10.1 模型工厂发布 API

最小接口为：

```text
POST /api/model-publications
GET  /api/model-publications/<task_id>
POST /api/model-publications/<task_id>/cancel
GET  /api/model-publications/<publication_id>/manifest
DELETE /api/model-publications/<publication_id>
```

创建请求必须携带管理员生成的 `Idempotency-Key`，并选择以下一种来源：

```json
{
  "source_type": "recipe",
  "recipe_path": "recipes/example.json",
  "display_name": "qwen-research-vlm",
  "vlm_base_model_id": null
}
```

或：

```json
{
  "source_type": "existing_model",
  "model_id": "...",
  "display_name": "qwen-research-text"
}
```

同一个幂等键和相同请求返回原任务；同一个键配合不同请求返回
`409 idempotency_conflict`。发布 API 不接受任意绝对路径：recipe 必须来自受管理的
配方目录，existing model 必须来自核心模型表。

取消规则复用现有任务取消语义：

- `queued` 立即取消且不创建 staging；
- `materializing` 和 `validating` 在分片复制、物化和验证阶段边界协作取消；
- 已进入原子提交临界区时返回 `409 commit_in_progress`，提交完成后由管理员删除正式资产；
- 取消后保存小型诊断记录并清理大型 staging 文件。

### 10.2 新增管理员候选接口

```text
GET /api/model-gateway/admin/publishable-models
```

返回正式发布资产的：

- 核心模型 ID；
- 显示名称；
- 文本/VLM 类型；
- capabilities；
- publication state；
- serving compatibility；
- 管理员可读的阻断原因。

### 10.3 创建服务

新服务创建请求以 `model_id` 为资产引用：

```json
{
  "model_id": "...",
  "display_name": "...",
  "served_model_name": "...",
  "gpu_ids": [0, 1]
}
```

后端根据核心模型表和 manifest 解析 `model_path`、`model_type` 和能力。客户端不能提交任意路径，也不能手动修改模型类型。

创建和启动服务前都必须重新检查：

- 模型目录仍位于 `/data/PublishedModels`；
- manifest 与文件指纹有效；
- 服务兼容性不是 `blocked` 或 `stale`；
- 当前 backend 版本和 manifest 记录一致。

现有服务记录继续按历史服务运行，不强制迁移；新建服务必须来自正式发布资产。

### 10.4 命名对齐

- 模型工厂：`物化并发布模型资产`。
- Gateway：`创建模型服务`、`启动服务`、`停止服务`。
- 保留代码中的 `backend_type="vllm"`，修正文档中的 `local_vllm`，不做无收益的数据迁移。
- API Key allowlist 继续使用当前代码中的 `served_model_name`。

## 11. 前端状态映射

### 11.1 模型工厂资产页

- 基础模型：仅作为融合或 VLM 视觉基座来源。
- 历史融合产物：显示“历史产物”，允许“验证并发布”。
- 配方：允许“物化并发布”。
- 正式资产：展示类型、能力、发布状态、验证摘要和服务兼容性。

目录名含 `VL` 但结构为文本的历史模型必须显示为文本，不能继续误导管理员。

### 11.2 Gateway 管理员页

- `ready`：可选并创建服务。
- `blocked`：管理员可见但禁用，展示简短原因。
- `stale`：要求重新验证后才能创建或启动。
- 模型路径和类型只读。
- 当前“发布模型”文案改为“创建模型服务”。

Qwen2.5-VL 在当前环境中显示为“模型资产完整，当前 vLLM 0.7.0 不支持”。

### 11.3 用户门户

用户门户继续只读取 `/v1/models`，并只显示：

- Gateway 状态为 `running`；
- API Key allowlist 允许；
- 后端健康检查通过。

Qwen2.5-VL 当前不显示禁用卡片、说明或占位入口。

## 12. 错误契约

| 错误码 | 含义 |
|---|---|
| `invalid_recipe` | 配方缺失或字段非法 |
| `vlm_base_missing` | 没有结构完整的 VLM 父模型 |
| `source_fingerprint_mismatch` | 父模型内容与配方记录不一致 |
| `architecture_mismatch` | 语言权重与视觉外壳不兼容 |
| `insufficient_disk_space` | 发布目录空间不足 |
| `materialization_failed` | 模型生成或复制失败 |
| `validation_failed` | 文件或真实推理验证失败 |
| `registration_pending` | 文件完整，核心注册等待恢复 |
| `serving_backend_unsupported` | 当前服务后端不支持该架构 |
| `asset_in_use` | 模型仍被 Gateway 服务或任务引用 |

错误不得触发以下隐式行为：

- 更换父模型；
- 把 VLM 改为文本模型；
- 使用其他 GPU；
- 绕过 vLLM 兼容性；
- 删除已有正式资产。

## 13. 删除保护

所有删除模型目录的接口必须调用同一个引用保护函数，包括现有：

```text
POST /api/models/delete
DELETE /api/model_repo/<model_id>
```

正式资产删除条件：

1. 没有 `starting`、`running` 或 `stopping` 服务。
2. 没有任何未删除的服务引用其模型 ID 或真实路径。
3. 没有排队或运行中的发布任务。
4. 管理员输入模型显示名称确认。

Gateway 服务只允许在 `stopped` 或 `failed` 状态软删除为 `deleted`。`served_model_name` 永久保留，避免 API Key allowlist、usage 和历史请求出现同名歧义。

正式资产先原子移动到 `.trash`，再删除核心模型记录，最后清理磁盘。数据库操作失败时把目录移回原路径。配方和评测记录不删除，因此资产仍可重新物化。

正式发布资产以核心 ORM `models` 表和 manifest 为准，不写入
`model_repo/data/models.json` 作为第二权威来源。现有删除路由保留 URL 兼容性，但对正式
资产必须委托上述统一删除服务，不能继续直接操作 JSON 或磁盘目录。

## 14. 测试与验收门禁

### Gate 0：系统基线

- `docker compose config --quiet`。
- `/healthz`、`/readyz`、`/api/models`、`/api/testset/list`、`/api/history`。
- 使用 `/opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests`。
- 记录 Git 状态、容器状态、GPU UUID、显存和 GPU 进程。
- GPU 2 和外部容器状态在整个验收期间保持不变。

### Gate 1：配方与 manifest 契约

- 文本配方、新 VLM 配方和旧 VLM 配方解析。
- 旧配方按父模型顺序选择第一个完整 VLM。
- TextOnly 目录名含 `VL` 时仍判定为文本。
- VLM 缺失视觉权重、processor 或图像 token 时失败。
- 父模型指纹变化和 tensor shape 不兼容时失败。
- manifest schema、文件哈希、原子更新和幂等扫描测试。

### Gate 2：原子发布与崩溃恢复

通过真实发布任务路径受控终止：

- 写入中断：正式目录不可见。
- 重命名前中断：只留下可清理 staging。
- 重命名后、DB 注册前中断：启动后补注册。
- 注册后重启：核心模型表只出现一次。
- 重复提交同一幂等请求：不产生重复资产。

受控故障注入只验证崩溃位置，不能替代 Gate 3 和 Gate 4 的真实模型推理。

### Gate 3：真实文本模型闭环

1. 使用真实 Qwen 文本模型物化并发布。
2. 管理员资产库显示该模型。
3. 通过正式候选接口创建 Gateway 服务。
4. 管理员选择获准且空闲的非 GPU 2 设备并启动服务。
5. API Key 通过 `/v1/models` 看到模型。
6. 真实调用 `/v1/chat/completions`。
7. usage 正确记录。
8. 停止并删除服务后，正式模型资产仍保留。
9. vLLM 进程组退出，显存回落，无 Ray 残留。

### Gate 4：真实 VLM 融合与发布闭环

执行前验证至少两个语言父模型与视觉基座兼容。条件不足时停止并记录，不重复同一个父模型伪造融合。

1. 走现有标准 VLM 进化入口。
2. 使用极小但真实的 CMMMU 样本完成一次搜索。
3. 保存包含 `vlm_path` 和结构化 `vlm_base` 的配方。
4. 从配方重新物化独立完整 VLM。
5. 验证视觉权重、processor 和图像 token。
6. 使用真实图片完成 Transformers 推理。
7. 使用已发布模型再次执行真实 CMMMU 样本。
8. 验证发布前后的模型路径、输入和指标记录。
9. 当前 Qwen2.5-VL 被标记为 `serving_backend_unsupported`。
10. 管理员可见原因，用户门户和 `/v1/models` 不可见。

该 Gate 证明融合和正式 VLM 资产有效，不代表当前 vLLM 已完成 VLM 服务支持。vLLM 升级或适配必须另立设计、计划和验收。

### Gate 5：删除与权限

- 被未删除服务引用的模型不能从任何现有删除接口移除。
- 运行中的服务不能删除。
- 删除服务后，usage 和历史请求仍可查询。
- 已删除服务名称不能被新服务复用。
- 普通用户接口不能读取 manifest、宿主路径、配方或阻断详情。

### Gate 6：前端与回归

- 管理员可以区分基础模型、历史产物和正式资产。
- blocked/stale 状态不能提交服务创建表单。
- 用户门户不出现 blocked、stale 或 stopped 模型。
- `node --check`、浏览器桌面和移动验证、键盘焦点与 reduced-motion 通过。
- 既有模型工厂融合、评测、研究门户和 Gateway 测试没有新增失败。

## 15. GPU 与硬件保护

- 默认测试不启动模型、不使用 GPU。
- 真实验收前重新采集 GPU 快照，不能复用历史状态。
- 禁止使用 GPU 2。
- 发现外部 GPU 进程、UUID 变化、显存异常或其他容器异常时立即停止。
- 使用完成当前测试所需的最少 GPU，不因为更多卡空闲就全部占用。
- 每次启动前记录目标 GPU、预计显存、超时、PID、日志和清理命令。
- 完成后检查 vLLM/Ray 进程消失、显存回落、临时密钥撤销和其他容器状态不变。

若没有安全可用的 GPU，真实功能 Gate 保持未完成，不以模拟结果代替，也不宣称验收成功。

## 16. 实施批次与回滚

按以下独立批次实施和提交：

1. 配方扩展与结构化 VLM 检测。
2. 发布目录、manifest、原子物化和启动恢复。
3. 核心模型注册和统一删除保护。
4. Gateway 正式资产候选接口与服务创建约束。
5. 管理员前端状态展示与用户隐藏规则。
6. 真实文本和 VLM 验收记录。

每批次验收失败后停止后续工作，只允许修复或回滚当前批次。

回滚原则：

- 代码按独立提交 `git revert`。
- 新配方只增加字段，旧代码仍能读取原字段。
- `/data/PublishedModels` 独立保存，回滚代码不删除正式资产。
- Compose mount 和配置回滚后，发布目录暂时不可见但数据不丢失。
- 第一版不新增 Gateway 数据库列；若实现发现必须改 schema，停止并另行设计迁移和回退。

## 17. 完成标准

只有以下条件全部满足，才能标记本功能完成：

- 文本模型真实发布、服务启动和 OpenAI-compatible API 调用通过。
- VLM 真实融合、独立物化、图片推理和 CMMMU 验证通过。
- 原子发布、崩溃恢复和重复提交验证通过。
- 所有模型删除入口均无法绕过服务引用保护。
- Qwen2.5-VL 对管理员可解释，对用户完全隐藏。
- GPU 2、外部 GPU 进程和其他容器状态无变化。
- 完整测试输出、脱敏日志和 GPU 快照写入独立验收记录。
- 无失败 staging、大型临时输出、vLLM/Ray 进程、临时密钥和显存残留。

## 18. 后续独立工作

本设计完成后，Stage 2 仍有一个明确的独立阻断项：选择并验证支持 Qwen2.5-VL 的 vLLM 版本或其他受控服务后端。该工作涉及依赖升级、CUDA 兼容性、现有文本服务回归和 GPU 真实验收，不能混入本发布管线实施。
