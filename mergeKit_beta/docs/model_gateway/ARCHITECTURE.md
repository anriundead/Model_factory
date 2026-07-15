# 模型服务与科研推理门户设计决策

> 状态：设计已收敛，v1 文本推理纵切进入实施计划
> 范围：融合模型发布、vLLM 推理服务、API 调用、文档解析、科研任务、引用定位、队列恢复、前端门户
> 非范围：真实支付、账单扣费、联网搜索、长期知识库、图片生成、真实融合/评测任务改造
> v1 实施计划：[`IMPLEMENTATION_HISTORY.md`](IMPLEMENTATION_HISTORY.md)

---

## 1. 目标

系统需要让已经融合好的模型被用户实际使用。第一版重点是可交付、可观测、可恢复：

1. 管理员可以把融合模型发布为可调用的模型服务。
2. 用户可以通过 OpenAI-compatible API 或嵌入式门户使用模型。
3. 文本模型和 VLM 模型都要支持，VLM 第一版支持 base64 图片输入。
4. 文档类科研任务要支持 PDF、DOC、DOCX、PPT、PPTX。
5. 科研回答中涉及文档事实的内容必须支持引用定位。
6. 系统记录 token 使用量，支付和扣费模块后续再接入。
7. 队列任务在系统重启后可恢复状态，但模型服务由管理员手动恢复启动。

---

## 2. 核心架构决策

### 2.1 服务边界

第一版采用当前 Flask 系统内嵌门户，不拆独立 SPA，也不改现有融合和评测核心逻辑。

新增服务边界：

- **Inference Gateway**：统一接入用户 API、鉴权、参数校验、入队、转发 vLLM、记录 usage。
- **Model Runtime Manager**：管理发布模型的 vLLM 进程、端口、GPU、状态和能力。
- **Document Parser Worker**：解析上传文件，生成带来源定位的 chunk。
- **Retrieval Layer**：负责 FTS + FAISS 混合检索。
- **Serving Queue Worker**：执行异步科研任务和长请求。
- **Portal UI**：嵌入现有系统，供用户和管理员使用。

### 2.2 不触碰的高危区域

本功能不改以下逻辑：

- `merge_manager.py` 的 VLM/LLM 评测三路分支。
- `HF_DATASETS_TRUST_REMOTE_CODE` 和 `--trust_remote_code` 相关评测修复。
- `evolution/vendor/vlm_merge/run_vlm_search.py` 的 vLLM TP 子进程隔离。
- `config.py` 中 Ray/vLLM 环境变量白名单，除非后续实现明确需要并单独审批。

---

## 3. 模型推理服务

### 3.1 vLLM 作为第一版推理运行时

融合模型发布后，由管理员启动 vLLM OpenAI-compatible server。用户不直接访问 vLLM，而是访问系统的 Inference Gateway。

这类似本地 Ollama 的使用体验，但运行形态不同：

- Ollama 更偏本地单机模型运行器。
- 本系统是服务器上的多用户模型服务平台。
- Gateway 负责 API Key、队列、文件、引用、token usage 和后续计费。
- vLLM 只负责模型推理。

### 3.2 管理员手动恢复模型服务

系统重启后不自动启动模型服务。原因：

- 模型加载会占用 GPU。
- 多模型自动恢复可能抢占硬件。
- 管理员需要明确控制哪些模型上线。

重启后的状态规则：

- 队列和任务状态恢复。
- 已发布模型记录保留。
- vLLM 进程状态为 `stopped`。
- 新请求如果目标模型未运行，返回 `503 model_not_running`。
- 已排队任务可进入 `paused_model_offline`。

### 3.2.1 当前纵切的在途请求恢复

当前第一版只有内存后台线程，没有持久 worker、任务 lease 或可安全重放的请求正文。因此重启后不重放在途请求：

- `running` 变为 `failed`，错误码 `request_interrupted_by_restart`。
- `streaming` 变为 `failed`，错误码 `stream_interrupted_by_restart`。
- `cancel_requested` 变为 `canceled`，错误码 `canceled_by_restart`。
- `queued` 不由当前实现产生或调度，保持原状，等待后续 Redis/DB durable queue 实现。

这样不会重复模型推理或重复记录 token。后续持久队列上线后，才按第 9.8 节的 lease 和幂等规则恢复 `queued`、`retrying` 等可安全重入队的任务。

### 3.3 Qwen 兼容性

当前融合模型以千问系为主。第一版按以下原则兼容：

- Qwen2/Qwen2.5 文本模型：优先支持。
- Qwen2-VL/Qwen2.5-VL：按 vLLM 多模态能力支持 base64 图片输入。
- 模型目录必须是标准 HuggingFace/Transformers 导出结构，包含 config、tokenizer 和权重文件。
- 发布前要做轻量启动检查，不在用户请求时才发现模型无法加载。

### 3.4 管理员发布参数

第一版允许管理员调整 vLLM 高级参数，但使用“安全默认值 + 高级配置折叠区”。普通发布只需要填模型路径、展示名称、模型类型和 GPU；高风险参数必须显式展开后修改。

必填参数：

| 参数 | 默认/规则 | 原因 |
|------|-----------|------|
| 模型路径 | 从系统已知融合模型中选择 | 避免手填路径导致加载不存在或越权路径。 |
| 展示名称 | 默认使用模型目录名 | 方便 `/v1/models` 和门户展示。 |
| 模型类型 | `text`，管理员可改为 `vlm` | 文本是当前主线；VLM 需要额外多模态限制。 |
| GPU 选择 | 管理员显式选择 | 不自动抢占 GPU，避免影响评测、融合和其他容器。 |
| `tensor_parallel_size` | 默认为已选 GPU 数；单卡则为 `1` | TP 必须和 GPU 分配一致，减少 vLLM 分布式初始化失败。 |
| streaming | 默认开启 | 用户体验更好，长回答能尽早返回。 |

高级参数：

| 参数 | 安全默认值 | 允许范围/策略 | 原因 |
|------|------------|---------------|------|
| `served_model_name` | 展示名称的安全 slug | 只允许字母、数字、`.`、`_`、`-` | 避免模型名中含空格或特殊字符影响 API 路由和日志。 |
| `host` | `127.0.0.1` | 第一版不允许改 | vLLM 只暴露给 Gateway，避免用户绕过鉴权和 usage 统计。 |
| `port` | 系统自动分配本机空闲端口 | 不允许用户手填，冲突时换端口 | 避免端口冲突和误暴露。 |
| `dtype` | `auto` | `auto` / `bfloat16` / `float16` | `auto` 优先尊重模型配置；手动项用于排障和显存优化。 |
| `max_model_len` | 空值，使用模型默认 | 允许管理员设置上限；不得超过模型配置上限 | 不盲目拉长上下文，避免 KV cache 占满显存。 |
| `gpu_memory_utilization` | `0.85` | `0.50` - `0.92` | 给 CUDA、系统和并发留下余量，降低 OOM 风险。 |
| `trust_remote_code` | 默认关闭 | Qwen/VLM 或模型明确需要时手动开启 | 降低加载任意远程代码的风险。 |
| `api_key` | 系统生成内部 key | 不展示给用户 | 用户只访问 Gateway，不直接访问 vLLM。 |
| `limit_mm_per_prompt` | VLM 默认 `image=1` | 第一版只允许 `image=1` | 控制单请求图片数量，避免 VLM 请求占用过多显存。 |
| `max_num_seqs` | `8` | `1` - `32` | 限制并发序列数，避免首版服务因过量并发抖动。 |
| `max_num_batched_tokens` | 空值，使用 vLLM 默认 | 管理员可设置 | 保持 vLLM 默认调度策略，只有排障时调整。 |
| `disable_log_requests` | 开启 | 第一版固定开启 | 避免用户 prompt 进入 vLLM 明文日志。 |

发布前必须做参数校验：

- GPU 数量必须大于等于 `tensor_parallel_size`。
- `tensor_parallel_size` 必须能整除或匹配 vLLM 对该模型的要求。
- VLM 模型必须启用多模态限制，例如 `limit_mm_per_prompt`。
- `gpu_memory_utilization` 超过 `0.92` 时拒绝保存。
- `host` 必须保持 `127.0.0.1`。
- `trust_remote_code` 打开时，前端必须显示风险提示。

第一版不开放任意 vLLM CLI 参数透传。新增参数必须进入白名单，并写明默认值、风险和验收方式。

### 3.5 管理员发布流程与字段说明

管理员发布页必须让非核心开发者也能安全使用。每个可编辑字段旁边都要有短说明，说明内容直接来自系统配置，不依赖外部文档。

发布流程：

1. 选择模型：从系统已知模型和融合产物中选择，不允许第一版手填任意路径。
2. 填服务信息：展示名称、API 模型名、模型类型。
3. 选择 GPU：展示 GPU index、UUID、总显存、当前占用和本系统占用状态。
4. 配置 vLLM 参数：普通字段直接展示，高级字段放入折叠区。
5. 启动前预检：模型路径、端口、GPU、TP、VLM 限制、危险参数确认。
6. 启动模型：状态进入 `starting`，展示健康检查进度和日志尾部。
7. 验收结果：成功进入 `running`；失败进入 `failed` 并保存 `last_error`。

字段说明展示规则：

- 每个字段右侧提供一句短说明，必要时带“为什么这样默认”。
- 高风险字段使用 warning 样式，不用普通灰色 helper text 淹没风险。
- 校验错误显示在字段下方，不只用 toast。
- 高级字段默认折叠，但启动前摘要必须展示最终生效值。
- 参数说明必须和后端白名单、默认值、允许范围一致。

发布表单字段：

| 字段 | 控件 | 旁边说明 | 校验/风险 |
|------|------|----------|-----------|
| 模型 | 下拉/搜索选择 | 选择已经注册或融合完成的模型。第一版不手填任意路径，避免加载错误目录。 | 必填；模型路径必须存在且位于允许目录。 |
| 展示名称 | 文本输入 | 显示在门户里的名称，方便管理员识别。 | 必填；建议不超过 64 字符。 |
| API 模型名 `served_model_name` | 文本输入 | 用户在 `/v1/chat/completions` 的 `model` 字段里使用这个名称。 | 必填；只允许字母、数字、`.`、`_`、`-`；必须唯一。 |
| 模型类型 | 分段控件：text/vlm | 文本模型只接收文字；VLM 模型允许 base64 图片输入。 | VLM 必须启用 `limit_mm_per_prompt=image=1`。 |
| GPU 选择 | 多选 GPU 列表 | 明确选择模型服务使用哪些 GPU，系统不会自动抢占。 | 必填；不能和本系统正在运行的模型服务冲突。 |
| `tensor_parallel_size` | 数字输入/步进器 | 多卡切分模型用。默认等于已选 GPU 数，单卡为 1。 | 必须小于等于已选 GPU 数；通常等于已选 GPU 数。 |
| streaming | 开关 | 开启后长回答可以边生成边返回，用户等待感更低。 | 默认开启。 |
| `dtype` | 下拉 | 模型权重加载精度。`auto` 最稳，优先尊重模型配置。 | 默认 `auto`；手动改动只建议排障或显存优化时使用。 |
| `max_model_len` | 数字输入，可空 | 限制最大上下文长度。留空表示使用模型默认值。 | 不得超过模型配置上限；过大可能导致显存不足。 |
| `gpu_memory_utilization` | 滑杆 + 数字输入 | vLLM 可使用的 GPU 显存比例。默认 0.85，为系统和并发保留余量。 | 允许 0.50-0.92；超过 0.92 拒绝保存。 |
| `max_num_seqs` | 数字输入 | 同时处理的序列数量。越高吞吐可能越好，但显存和抖动风险更高。 | 默认 8；允许 1-32。 |
| `max_num_batched_tokens` | 数字输入，可空 | vLLM 批处理 token 上限。留空使用 vLLM 默认调度。 | 第一版建议留空，只有排障时设置。 |
| `trust_remote_code` | 危险开关 | 允许模型加载自定义 Python 代码。某些 Qwen/VLM 模型可能需要，但有安全风险。 | 默认关闭；开启时必须二次确认。 |
| `limit_mm_per_prompt` | 只读/固定项 | VLM 单请求最多允许 1 张图片，避免单个请求占满显存。 | VLM 固定 `image=1`；text 模型不显示。 |
| `host` | 只读 | vLLM 只监听 `127.0.0.1`，用户必须通过 Gateway 访问。 | 第一版不允许修改。 |
| `port` | 只读 | 系统自动分配本机端口，避免冲突。 | 不允许手填。 |
| 内部 vLLM Key | 不展示 | 系统自动生成，只给 Gateway 调 vLLM 使用。 | 永不展示给用户或管理员页面。 |
| `disable_log_requests` | 只读 | 固定开启，避免用户 prompt 进入 vLLM 明文日志。 | 第一版不允许关闭。 |

启动前摘要必须展示：

- 目标模型路径。
- API 模型名。
- 模型类型和输入限制。
- GPU index / UUID。
- `tensor_parallel_size`。
- `gpu_memory_utilization`。
- 是否开启 `trust_remote_code`。
- 预计 vLLM 监听地址：`127.0.0.1:<auto-port>`。

启动前预检失败时，不进入 `starting`：

- 模型路径不存在。
- API 模型名重复。
- GPU 被本系统其他运行中模型服务占用。
- `tensor_parallel_size` 与已选 GPU 不匹配。
- VLM 未设置图片数量限制。
- 端口池无可用端口。
- 危险参数未确认。

### 3.6 启动前预检与资源冲突判定

模型启动必须先完成预检和资源预约。第一版只保护本系统能准确控制的资源，不尝试接管宿主机上的未知进程。

开发和验收阶段默认只执行无 GPU 的 HTTP smoke、导入检查和单元测试。任何真实 vLLM/GPU 模型启动测试都必须先明确目标 GPU、预计显存、预计时长、停止/清理方式，并在人工确认后执行，避免影响同机其他业务。

当前已获得真实用户模拟验收授权：需要端到端验证时，可以先通过模型工厂融合进化功能融合一个 7B 模型，再把该模型纳入 serving 发布和调用流程。该动作不属于默认 smoke，执行前仍必须确认 GPU 空闲、预计耗时、停止方式、日志路径和回滚方式。

资源判定分三层：

1. **硬件快照层**：通过 `nvidia-smi` 获取 GPU index、UUID、PCI bus id、总显存、空闲显存和进程占用摘要。
2. **系统预约层**：读取 `serving_model_services` 中 `starting`、`running`、`stopping` 状态的 GPU 和端口占用。
3. **外部占用保护层**：发现非本系统 GPU 进程时，只展示占用摘要并阻断默认启动，不提供 kill 或覆盖启动。

复用现有边界：

- `core.gpu_topology.query_gpus()` 可作为第一版 GPU 显存快照基础。
- `core.gpu_lock.lock_file()` 可作为启动流程的进程内互斥锁，避免两个管理员同时启动抢同一资源。
- `core.process_manager.ProcessManager.create_process_group_kwargs()` 可用于启动独立进程组。
- 停止模型时不能直接复用粗粒度 kill；必须先校验 `MERGEKIT_MODEL_GATEWAY_SERVICE_ID=<service_id>` 或等价服务标记。

预检步骤：

1. 模型路径：
   - 路径存在。
   - 位于允许的模型目录范围。
   - 存在 `config.json`。
   - 存在 tokenizer 文件。
   - 存在 `.safetensors` 或 `.bin` 等权重文件。
   - text/vlm 类型和模型 config 的初步判断一致；不一致时要求管理员确认。
2. API 模型名：
   - `served_model_name` 唯一。
   - 只允许字母、数字、`.`、`_`、`-`。
   - 不和已发布的非 `deleted` 服务冲突。
3. GPU：
   - 选中 GPU 的 UUID 当前仍存在。
   - GPU index 和 UUID 与启动前快照一致。
   - 未被本系统其他 `starting`、`running`、`stopping` 模型服务预约。
   - 没有非本系统重占用；第一版遇到外部重占用默认阻断。
   - 空闲显存满足安全阈值。
4. TP：
   - `tensor_parallel_size <= 已选 GPU 数`。
   - 第一版推荐 `tensor_parallel_size == 已选 GPU 数` 或 `tensor_parallel_size == 1`。
   - 不自动使用未选择 GPU。
5. 显存：
   - `gpu_memory_utilization` 默认 `0.85`。
   - `gpu_memory_utilization` 最大 `0.92`。
   - 每张卡至少保留基础余量，默认建议 4-6 GiB。
   - 如果能估算模型权重大小，则展示“权重大小 / TP + KV cache 余量”的风险提示。
   - 如果无法估算，不伪装成精确预测，只做保守阈值检查。
6. 端口：
   - 只从固定端口池选择，例如 `18000-18999`。
   - 端口未被本机监听。
   - 端口未被 DB 中其他 `starting` 或 `running` 服务占用。
   - 用户不能手填端口。
7. 危险参数：
   - `trust_remote_code=true` 必须二次确认。
   - VLM 必须固定 `limit_mm_per_prompt=image=1`。
   - `host` 必须是 `127.0.0.1`。
   - `disable_log_requests` 必须开启。

资源预约流程：

```text
获取 serving 启动锁
  -> 重新读取 GPU/端口快照
  -> 检查 DB 中 GPU/端口占用
  -> 写入 status=starting、gpu_ids、gpu_uuids、vllm_port
  -> 释放启动锁
  -> 启动 vLLM 子进程
```

预约规则：

- 预约只代表本系统计划使用，不代表宿主机全局锁。
- 预约写入 DB 后，如果启动失败必须释放 GPU 和端口字段。
- 同一时间只能有一个启动流程持有 serving 启动锁。
- GPU 冲突以 UUID 为准，index 只作为展示和 `CUDA_VISIBLE_DEVICES` 输入。
- 系统不按端口或模型名杀进程。

失败处理：

- 预检失败：不进入 `starting`，直接返回结构化错误。
- 预约成功但 vLLM 启动失败：状态改为 `failed`，释放 GPU/端口预约，保存 `last_error`。
- 健康检查超时：状态改为 `failed`，终止该服务进程组，释放预约。
- 发现 PID、PGID 或服务标记不匹配：不杀进程，标记人工处理。
- 发现外部 GPU 占用：不终止外部进程，不自动覆盖启动。

管理员界面必须展示：

- GPU index、UUID、PCI bus id。
- 总显存、空闲显存、已使用显存。
- 本系统占用：模型服务名、状态、PID。
- 外部占用摘要：PID、进程名、显存占用；不提供 kill 按钮。
- 预检结果：通过、阻断、警告。
- 启动摘要：模型、GPU、TP、显存比例、端口、危险参数。

第一版不做：

- 不做外部进程强制清理。
- 不做管理员“强制覆盖外部 GPU 占用”。
- 不做跨容器全局 GPU 调度。
- 不接管融合/评测任务已有 Ray/vLLM TP 调度逻辑。

验收标准：

- 两个启动请求同时选择同一 GPU 时，只能有一个成功预约。
- GPU UUID 变化时阻断启动，并提示刷新硬件快照。
- 本系统已有 running 服务占用 GPU 时，不能再次发布到同一 GPU。
- 外部进程占用显存超过阈值时，默认阻断启动。
- 预约后启动失败时，GPU/端口预约会释放。
- force-stop 只影响服务标记匹配的进程组，不影响其他进程。

### 3.7 模型服务状态机

模型服务状态必须由 Gateway/Runtime Manager 统一维护。前端、队列和用户 API 只读取同一份状态，避免出现“前端显示在线，但请求实际不可用”的情况。

第一版只保留 6 个运行状态：

| 状态 | 含义 | 用户请求行为 | 管理员可执行操作 |
|------|------|--------------|------------------|
| `stopped` | 已发布但未运行，或系统重启后进程未恢复 | 返回 `503 model_not_running`；已排队任务进入 `paused_model_offline` | start / edit / delete |
| `starting` | 正在启动 vLLM 进程并等待健康检查 | 新请求可入队，但不直接转发；超过等待时间返回 request/job ID | force_stop |
| `running` | vLLM 健康检查通过，可正常服务 | 正常转发或排队执行 | stop / restart |
| `stopping` | 管理员请求停止，正在释放进程和端口 | 新请求返回 `503 model_stopping`；队列不再派发到该模型 | force_stop |
| `failed` | 启动失败、健康检查连续失败或进程异常退出 | 返回 `503 model_failed`，响应中带失败摘要 | restart / edit / stop |
| `deleted` | 发布记录已删除，不再可用 | 返回 `404 model_not_found` | 无 |

状态流转规则：

```text
publish -> stopped
stopped --admin start--> starting
starting --health ok--> running
starting --start timeout/error--> failed
running --health failed/process exit--> failed
running --admin stop--> stopping
stopping --process exited--> stopped
failed --admin restart--> starting
failed --admin stop--> stopped
stopped/failed --admin delete--> deleted
```

系统重启后的恢复规则：

- 不自动恢复 vLLM 进程。
- 启动时扫描数据库中非 `deleted` 的模型服务记录。
- 如果原状态是 `running`、`starting` 或 `stopping`，统一改为 `stopped`，并记录 `last_exit_reason=system_restarted_manual_recovery_required`。
- 如果原状态是 `failed`，保持 `failed`，方便管理员看到重启前的错误。
- 队列任务不会自动拉起模型；只进入等待模型状态。

健康检查规则：

- `starting` 阶段必须调用 vLLM `/health` 或等价探针。
- 启动超时默认 300 秒，超过则进入 `failed`。
- `running` 阶段连续 3 次健康检查失败后进入 `failed`。
- 每次状态变化记录 `last_status_change_at`、`last_error`、`pid`、`port`、`gpu_ids`。

停止规则：

- 普通 stop 先停止派发新任务，再终止 vLLM 进程。
- 已经转发给 vLLM 的请求尽力完成；管理员选择 force stop 时可以直接终止进程。
- stop 完成后释放端口和 GPU 绑定记录。

状态机验收标准：

- 模型未启动时，用户 API 返回明确 `503 model_not_running`。
- 启动失败时进入 `failed`，并能在管理员界面看到失败摘要。
- 进程被手动 kill 后，健康检查能把状态改为 `failed`。
- 系统重启后，原 `running` 模型不会自动占用 GPU，状态变为 `stopped`。
- `deleted` 模型不能被队列恢复，也不能被 API Key allowlist 继续访问。

### 3.8 vLLM Runtime 操作细节

第一版由 `app/model-gateway/runtime.py` 管理 vLLM 子进程，不引入 Kubernetes、Ray Serve、动态 systemd service 或每模型一个 Docker 容器。这样能复用当前 Docker Compose 部署，同时避免把模型发布功能和现有融合、评测调度绑在一起。

运行时记录字段：

- `vllm_pid`：当前 vLLM 进程 PID。
- `vllm_pgid`：当前 vLLM 进程组 ID，用于安全停止。
- `vllm_host`：固定为 `127.0.0.1`。
- `vllm_port`：系统从固定范围自动分配，例如 `18000-18999`。
- `last_health_at`：最近一次健康检查通过时间。
- `started_at`：最近一次启动时间。
- `last_error` / `last_exit_reason`：失败摘要和退出原因。

启动流程：

1. 管理员调用 start。
2. 校验模型路径存在，并位于允许的模型目录范围内。
3. 校验所选 GPU 未被本系统已运行模型服务占用。
4. 校验 `tensor_parallel_size` 与 GPU 数量匹配。
5. 从端口范围内选择空闲端口，并在数据库中占用该端口。
6. 生成内部 vLLM API key，只允许 Gateway 使用。
7. 状态改为 `starting`。
8. 使用独立进程组启动 vLLM，环境中设置 `CUDA_VISIBLE_DEVICES` 和 `MERGEKIT_MODEL_GATEWAY_SERVICE_ID=<service_id>`。
9. 日志写入 `logs/model-gateway/<service_id>.log`，并固定开启 `--disable-log-requests`。
10. 轮询 `http://127.0.0.1:<port>/health`，最多等待 300 秒。
11. 健康检查通过后状态改为 `running`；启动失败或超时则改为 `failed`，保存失败摘要。

停止流程：

1. 管理员调用 stop。
2. 状态改为 `stopping`。
3. 队列 worker 不再派发新请求到该模型。
4. 根据数据库记录的 `vllm_pid` / `vllm_pgid` 查找进程。
5. 停止前校验进程命令或环境中包含 `MERGEKIT_MODEL_GATEWAY_SERVICE_ID=<service_id>`。
6. 普通 stop 对进程组发送 `SIGTERM`，等待最多 30 秒。
7. 如果仍未退出，管理员可调用 force-stop，对进程组发送 `SIGKILL`。
8. 进程退出后状态改为 `stopped`，释放端口和 GPU 占用记录。

安全约束：

- 不按端口杀进程。
- 不按模型名杀进程。
- 不处理数据库之外的未知 vLLM 进程。
- 不把用户 API Key 透传给 vLLM。
- vLLM 只监听 `127.0.0.1`，用户只能访问 Gateway。
- 系统重启后不自动启动模型，也不根据旧 PID 终止进程。

健康检查：

- `starting` 阶段必须等 `/health` 通过后才能进入 `running`。
- `running` 阶段连续 3 次健康检查失败后进入 `failed`。
- 如果进程已退出，立即进入 `failed`。
- 健康检查只记录状态、时间和错误摘要，不记录用户 prompt。

操作验收：

- 启动成功后，`/v1/models` 能看到该服务暴露的模型名。
- 启动失败后，服务状态为 `failed`，管理员能看到 `last_error`。
- 普通 stop 后，进程退出，端口释放，状态为 `stopped`。
- force-stop 只终止对应服务的进程组，不影响其他容器或其他模型服务。
- 系统重启后，旧 `running` 服务变为 `stopped`，不会自动占用 GPU。

---

## 4. API 与门户范围

### 4.1 OpenAI-compatible API

第一版支持：

- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /v1/requests/<request_id>`
- `POST /v1/requests/<request_id>/cancel`
- streaming response
- token usage 记录

暂不完整实现 OpenAI Files / Assistants / Batch API。

用户推理 API 走 OpenAI-compatible 路径，方便用户复用现有 SDK：

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/v1/models` | 用户 API Key | 返回该 Key allowlist 内可见模型 |
| POST | `/v1/chat/completions` | 用户 API Key | 文本/VLM 对话推理，支持 streaming |
| GET | `/v1/requests/<request_id>` | 用户 API Key | 查询自己请求的状态与 usage 摘要 |
| POST | `/v1/requests/<request_id>/cancel` | 用户 API Key | 取消自己未完成的请求 |

鉴权方式：

```http
Authorization: Bearer mk_live_xxx
```

请求规则：

- 支持 `stream: true`。
- 支持 `Idempotency-Key` header。
- 非流式请求 60 秒内完成则同步返回。
- 非流式请求 60 秒内未完成则返回 `202 Accepted` 和 `request_id`。
- VLM 图片第一版只支持 base64/data URL。
- 用户不能直接访问 vLLM 内部端口。

### 4.2 One API / New API 借鉴边界

One API / New API 的成熟点是 API 中转站模型：统一 API Base、用户 Token、渠道、模型映射、用量统计和后续计费。第一版借鉴这些概念，但不把 One API / New API 作为运行依赖。

概念映射：

| One API / New API 概念 | 本系统概念 | 第一版实现 |
|------------------------|------------|------------|
| Channel | Model Service | 只实现本地 vLLM 服务 |
| Token | API Key | 管理员生成，用户调用 `/v1/*` |
| API Base | Gateway `/v1/*` | OpenAI-compatible |
| Usage / Cost | `serving_usage_records` | 先记录 token，支付后接 |
| Channel health | Model Service health | vLLM `/health` + 进程检查 |

保留扩展字段：

- `backend_type=local_vllm`：第一版唯一启用类型。
- `backend_type=external_openai_compatible`：预留给未来外部模型供应商或 One API 类上游。

第一版不做：

- 不接入 One API / New API 数据库。
- 不复用它们的前端和用户系统。
- 不做多渠道负载均衡。
- 不把外部供应商配置暴露给普通用户。
- 不允许 API 创建或启动 `external_openai_compatible` 后端。
- 不提前保存外部上游 URL、外部 API Key 或供应商密钥。
- 不做复杂模型映射和请求体重写，只做必要校验、裁剪和透传。

这样可以复用成熟中转站的产品模型，但避免双系统、双权限、双数据库和双前端带来的维护成本。

### 4.3 Gateway 转发、usage 与失败策略

Gateway 参考 New API 的成熟边界：统一 API Base、统一鉴权、模型访问限制、usage 统计和错误归一。第一版不做复杂渠道路由，只转发到本地 `local_vllm` 服务。

请求流：

```text
User / SDK
  -> Gateway
  -> API Key 鉴权
  -> model allowlist 检查
  -> 模型服务状态检查
  -> 参数校验和裁剪
  -> vLLM OpenAI-compatible endpoint
  -> usage / request 状态记录
  -> 返回用户
```

转发规则：

- `stream: true`：Gateway 直接流式转发到 vLLM，并监听客户端断连。
- `stream: false`：Gateway 最多等待 60 秒。
- 60 秒内完成：同步返回 vLLM 结果。
- 60 秒内未完成：返回 `202 Accepted` 和 `request_id`，后台继续执行。
- 请求体只透传 OpenAI-compatible 标准字段。
- 禁止用户指定后端 URL、端口、host、内部 vLLM API Key 或任意上游配置。
- Gateway 调 vLLM 时只使用模型服务内部 Key。

usage 记录：

- 优先使用 vLLM 返回的 `usage.prompt_tokens`、`usage.completion_tokens`、`usage.total_tokens`。
- 成功读取 usage 时写入 `serving_usage_records`，`usage_source=vllm`。
- vLLM 未返回 usage 时不阻塞用户请求，写入 `usage_source=missing`。
- 第一版不强行用 tokenizer 重算计费级 token；付费上线前再补 `usage_source=estimated` 的兜底规则。

可重试错误：

- vLLM 连接短暂失败。
- vLLM 请求超时。
- 进程健康检查短暂抖动。

不可重试错误：

- API Key 无效、禁用、撤销或过期。
- 模型不在 allowlist。
- 请求体非法。
- 文件不存在或不属于当前 API Key。
- context 超长。
- vLLM 明确返回 4xx。

失败状态规则：

- 可恢复错误进入 `retrying`，超过最大重试次数后进入 `dead_letter`。
- 不可恢复错误直接进入 `failed`。
- 用户取消后进入 `cancel_requested` 或 `canceled`，Reconciler 不再恢复。
- 流式请求连接断开后，不恢复同一条 HTTP 流；后台状态按是否已提交结果决定。

### 4.3.1 请求执行与流式响应可靠性

第一版把聊天请求分成三类执行路径，避免所有请求都走同一套复杂任务系统。

执行路径：

| 请求类型 | 条件 | 执行方式 | 用户看到的结果 |
|----------|------|----------|----------------|
| 直接流式 | `stream=true` 且模型 `running` | Gateway 直接代理 vLLM 流式响应 | SSE 流式 token |
| 同步等待 | `stream=false`，预计可在 60 秒内完成 | Gateway 创建 request 记录并等待执行结果 | 60 秒内返回最终 JSON |
| 异步后台 | `stream=false` 但超过 60 秒，或科研/文档任务 | DB + Redis Streams + worker | 返回 `202` 和 `request_id` / `job_id` |

直接流式规则：

- 进入 vLLM 前先创建 `serving_requests`，状态为 `running`。
- vLLM 首个 chunk 返回后，状态改为 `streaming`。
- Gateway 逐块转发 SSE，不缓存完整正文到 DB。
- 客户端主动断开时，标记 `cancel_requested`，尽力关闭到 vLLM 的上游连接。
- 如果 vLLM 已经完成并返回 usage，则写入 usage；否则写入 `usage_source=missing`。
- 流式请求断开后不恢复同一个 HTTP 流，也不尝试把后续 token 补发给客户端。
- 系统重启时仍处于 `streaming` 的请求转为 `failed`，错误码 `stream_interrupted_by_restart`。

同步等待规则：

- `stream=false` 时，Gateway 创建 request 记录，状态为 `running` 或 `queued`。
- Gateway 最多等待 60 秒。
- 60 秒内完成：同步返回 vLLM 的 OpenAI-compatible JSON，并写 usage。
- 超过 60 秒：返回 `202 Accepted`，响应体包含 `request_id`、`status_url` 和 `cancel_url`，后台继续执行。
- 等待期间如果客户端断开，不自动取消后台任务；用户可以用 cancel API 显式取消。
- 同步等待不保存长期用户正文，payload 按第 10 章 TTL 清理。
- 当前第一版已实现该语义：普通聊天非流式请求使用轻量后台线程继续执行，默认等待 `MERGEKIT_MODEL_GATEWAY_SYNC_WAIT_SECONDS=60` 秒；完整 Redis Streams worker 仍留给科研任务和更可靠队列阶段。

异步后台规则：

- 科研任务、文档任务和超过 60 秒的非流式请求进入队列。
- DB 是状态事实源，Redis Streams 只负责投递。
- worker 开始执行前必须 claim DB lease。
- worker 完成后先写 DB 终态和 usage，再 ACK Redis 消息。
- 重复投递时，DB 终态优先，不能覆盖已完成、已取消或已失败的结果。

超时策略：

| 超时项 | 默认值 | 行为 |
|--------|--------|------|
| Gateway 同步等待 | 60 秒 | 超时返回 `202`，后台继续执行 |
| Gateway 连接 vLLM 超时 | 10 秒 | 可重试，仍失败则 `retrying` 或 `model_failed` |
| vLLM 首 token 超时 | 120 秒 | 标记 `request_timeout`，可按策略重试 |
| 非流式总执行超时 | 可配置，默认 600 秒 | 超时进入 `failed` 或 `dead_letter` |
| 科研任务总执行超时 | 可配置，默认 1800 秒 | 超时进入 `failed` 或 `dead_letter` |

重试规则：

- 只重试可恢复错误：连接短暂失败、上游 5xx、健康检查短暂抖动、worker lease 过期。
- 不重试用户错误：鉴权失败、allowlist 不匹配、参数非法、context 超长、文件越权、vLLM 4xx。
- 默认最多重试 2 次。
- 每次重试写 `serving_events`，记录错误码和摘要，不记录 prompt。
- 超过重试次数进入 `dead_letter`，管理员可在队列页面查看摘要。

取消规则：

- `queued`、`paused_model_offline`、`retrying`：立即变为 `canceled`。
- `running` 非流式：标记 `cancel_requested`，worker 停止后写 `canceled`。
- `streaming`：关闭客户端连接并尽力关闭上游连接；未提交的结果丢弃。
- 已 `completed`、`failed`、`dead_letter`、`expired` 的请求不能取消。
- 取消不删除 usage；如果取消前已经产生 usage，则保留 usage 记录。
- 当前已实现第一版取消 API：`POST /v1/requests/<request_id>/cancel`，只允许 API Key 取消自己的请求；它负责写入状态，不负责强杀 vLLM 进程。

状态查询规则：

- 当前已实现第一版状态查询 API：`GET /v1/requests/<request_id>`。
- 只允许 API Key 查询自己创建的请求；跨用户查询返回 `404 request_not_found`。
- 响应返回请求状态摘要和最近一条 usage 摘要。
- 未记录 usage 时返回 0 token 和 `usage_source=not_recorded`。
- 该接口不返回用户 prompt 正文，避免把短期会话内容长期留在服务器响应链路里。

usage 提交规则：

- 只有请求进入终态后才提交 usage。
- 成功读取 vLLM usage 时使用 `usage_source=vllm`。
- vLLM 未返回 usage 时写 `usage_source=missing`，不阻断响应。
- 取消或断连后，如果 vLLM 未返回 usage，不做猜测计费。
- 付费上线前再补 tokenizer 估算，使用 `usage_source=estimated`。

幂等规则：

- `Idempotency-Key` 只对同一 API Key、同一模型、同一请求类型生效。
- 短窗口内重复请求返回同一个 `request_id` 或已有结果。
- 如果原请求仍在运行，重复请求返回当前状态，不创建新任务。
- 如果原请求已完成，重复请求返回结果摘要或结果地址。
- 如果原请求已取消或失败，重复请求不自动复活旧任务，需用户重新提交新 key。

验收标准：

- `stream=true` 能收到 SSE chunk；客户端断开后服务端不继续长期占用连接。
- `stream=false` 在 60 秒内完成时返回最终 JSON。
- `stream=false` 超过 60 秒时返回 `202` 和可查询 request ID。
- vLLM 连接失败只对可恢复错误重试，用户参数错误不重试。
- usage 缺失时响应不失败，但记录 `usage_source=missing`。
- 重复 `Idempotency-Key` 不创建多个有效任务。
- 取消 queued 任务立即生效；取消 running/streaming 任务不会提交未完成结果。

### 4.4 请求字段白名单与安全上限

Gateway 对 `/v1/chat/completions` 使用白名单校验。第一版只允许明确支持的 OpenAI-compatible 字段；未知字段默认拒绝，不静默透传。原因是 vLLM 支持一些非 OpenAI extra parameters，直接放行会绕过平台的安全上限、usage 统计和后续计费边界。

允许字段：

| 字段 | 第一版规则 |
|------|------------|
| `model` | 必填；必须存在于当前 API Key 的 allowlist。 |
| `messages` | 必填；只允许 `system` / `user` / `assistant` 三类角色。 |
| `stream` | 可选 boolean；默认 `false`。 |
| `max_tokens` | 可选 integer；默认 1024；最大 4096，可按模型服务配置调低或调高。 |
| `temperature` | 可选 number；范围 `0.0` - `2.0`。 |
| `top_p` | 可选 number；范围 `0.0` - `1.0`。 |
| `stop` | 可选 string 或 string list；最多 4 个 stop 序列。 |
| `presence_penalty` | 可选 number；范围 `-2.0` - `2.0`。 |
| `frequency_penalty` | 可选 number；范围 `-2.0` - `2.0`。 |
| `seed` | 可选 integer；用于可复现实验。 |
| `response_format` | 可选；第一版只允许 `{"type":"text"}` 或 `{"type":"json_object"}`。 |

第一版拒绝字段：

- `tools`
- `tool_choice`
- `parallel_tool_calls`
- `functions`
- `function_call`
- `logprobs`
- `top_logprobs`
- `n` 大于 1
- `modalities`
- `audio`
- `metadata`
- `store`
- vLLM extra parameters，例如 `guided_choice`、`guided_json`、`structured_outputs`、`extra_body`

拒绝原因：

- 工具调用属于 Agent 编排范围，第一版不实现。
- 多候选输出会放大显存和 token 成本，第一版只允许 `n=1`。
- logprobs、audio 和多模态音频不属于第一版交付目标。
- `metadata` / `store` 属于外部平台能力，本系统不长期保存用户对话正文。
- vLLM extra parameters 必须后续逐项白名单化，不能由用户自由透传。

VLM 消息规则：

- 文本模型只允许 `content` 为 string。
- VLM 模型允许 `content` 为 OpenAI-style content parts。
- content part 第一版只允许 `text` 和 `image_url`。
- `image_url.url` 只允许 `data:image/...;base64,...`。
- 不允许远程图片 URL。
- 不允许 `file_id` 图片引用。
- 每次请求最多 1 张图片。
- 单张 base64 图片解码后默认最大 10MB。
- 拒绝 `image_url.detail`，因为 vLLM OpenAI-compatible 文档说明该参数不支持。

请求体大小和结构限制：

- JSON 请求体默认最大 20MB。
- `messages` 默认最多 64 条。
- 单条文本 content 默认最大 64k 字符。
- 总输入是否超过上下文，由 Gateway 基于模型服务 `max_model_len` 做预检；无法可靠预估时交给 vLLM 返回错误，并记录为 `context_length_exceeded`。

参数处理原则：

- 超出范围的字段返回 `422 invalid_request`，不自动修正。
- 未支持字段返回 `422 unsupported_field`。
- 不符合当前模型类型的字段返回 `422 incompatible_input`。
- Gateway 可以注入 `max_tokens` 默认值，但不能改写用户正文。
- Gateway 不做复杂 prompt 模板拼接；聊天模板由 vLLM/模型 tokenizer 处理。

### 4.5 自定义科研任务 API

科研任务不强行塞进纯聊天接口，新增平台 API：

- `POST /api/model-gateway/files`
- `GET /api/model-gateway/files`
- `GET /api/model-gateway/files/<file_id>`
- `DELETE /api/model-gateway/files/<file_id>`
- `POST /api/model-gateway/research/jobs`
- `GET /api/model-gateway/research/jobs/<job_id>`
- `POST /api/model-gateway/research/jobs/<job_id>/cancel`
- `GET /api/model-gateway/requests/<request_id>`
- `POST /api/model-gateway/requests/<request_id>/cancel`

平台增强 API 使用用户 API Key 鉴权：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/model-gateway/files` | multipart 上传 PDF/DOC/DOCX/PPT/PPTX |
| GET | `/api/model-gateway/files` | 列出当前 API Key 可见文件 |
| GET | `/api/model-gateway/files/<file_id>` | 查看文件解析、索引和 TTL 状态 |
| DELETE | `/api/model-gateway/files/<file_id>` | 删除文件、chunk 和可重建索引 |
| POST | `/api/model-gateway/research/jobs` | 创建文档问答、摘要、对比、结构化抽取任务 |
| GET | `/api/model-gateway/research/jobs/<job_id>` | 查询科研任务状态和结果 |
| POST | `/api/model-gateway/research/jobs/<job_id>/cancel` | 取消科研任务 |
| GET | `/api/model-gateway/requests/<request_id>` | 查询普通推理请求状态 |
| POST | `/api/model-gateway/requests/<request_id>/cancel` | 取消普通异步请求 |

科研任务请求体核心字段：

```json
{
  "model": "served-model-name",
  "task_type": "document_qa",
  "file_ids": ["file_xxx"],
  "input": "请总结这份论文的方法和实验结论",
  "output_format": "markdown",
  "require_citations": true
}
```

约束：

- `file_ids` 必须属于当前 API Key。
- 文档事实默认要求引用。
- 查询结果不能返回已过期 payload。
- 删除文件后，相关 chunk 和向量索引不再可召回。

### 4.6 管理员 API

管理员 API 使用 `MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN`，不使用用户 API Key。

鉴权方式：

```http
Authorization: Bearer <MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN>
```

模型服务管理：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/model-gateway/admin/model-services` | 列出发布服务 |
| POST | `/api/model-gateway/admin/model-services` | 创建模型服务发布记录 |
| GET | `/api/model-gateway/admin/model-services/<service_id>` | 查看服务详情 |
| PATCH | `/api/model-gateway/admin/model-services/<service_id>` | 修改未运行服务配置 |
| DELETE | `/api/model-gateway/admin/model-services/<service_id>` | 删除发布记录 |
| POST | `/api/model-gateway/admin/model-services/<service_id>/start` | 启动 vLLM |
| POST | `/api/model-gateway/admin/model-services/<service_id>/stop` | 停止 vLLM |
| POST | `/api/model-gateway/admin/model-services/<service_id>/restart` | 重启 vLLM |
| POST | `/api/model-gateway/admin/model-services/<service_id>/force-stop` | 强制停止 vLLM |

API Key 管理：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/model-gateway/admin/api-keys` | 列出 Key 摘要 |
| POST | `/api/model-gateway/admin/api-keys` | 创建 Key，明文只返回一次 |
| POST | `/api/model-gateway/admin/api-keys/<key_id>/disable` | 禁用 Key |
| POST | `/api/model-gateway/admin/api-keys/<key_id>/revoke` | 撤销 Key |

观测接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/model-gateway/admin/requests` | 查询请求和任务 |
| GET | `/api/model-gateway/admin/usage` | 查询 token usage |
| GET | `/api/model-gateway/admin/events` | 查询服务事件 |

管理员约束：

- 正常用户不能访问 `/api/model-gateway/admin/*`。
- 启动模型前必须通过参数白名单校验。
- 删除模型服务前必须先停止运行进程。
- 删除 API Key 不删除历史 usage，只阻断后续访问。

### 4.7 统一错误格式

新接口统一返回结构化错误：

```json
{
  "error": {
    "code": "model_not_running",
    "message": "Model service is not running.",
    "request_id": "req_xxx"
  }
}
```

常见错误码：

| HTTP | code | 含义 |
|------|------|------|
| 401 | `invalid_api_key` | API Key 无效 |
| 403 | `model_not_allowed` | API Key 无权访问该模型 |
| 404 | `model_not_found` | 模型、文件或任务不存在 |
| 409 | `request_already_exists` | 幂等 key 对应任务已存在 |
| 413 | `file_too_large` | 上传文件过大 |
| 422 | `invalid_request` | 请求体不合法 |
| 422 | `unsupported_field` | 请求字段第一版不支持 |
| 422 | `incompatible_input` | 输入与模型类型不兼容 |
| 429 | `queue_full` | 队列已满 |
| 503 | `model_not_running` | 模型未运行 |
| 503 | `model_failed` | 模型服务失败 |
| 504 | `request_timeout` | 请求超时 |

### 4.8 路由实现边界

当前 `app/routes.py` 已承载大量融合、评测、测试集和历史接口。第一版 serving 功能应使用独立蓝图：

```text
app/model-gateway/routes.py
```

注册到当前 Flask app 后暴露 `/v1/*` 和 `/api/model-gateway/*`。这样避免继续扩大 `routes.py`，也便于后续将 serving 模块独立测试。

### 4.9 后端模块拆分

Serving 功能采用独立包，不继续把新逻辑堆进 `app/routes.py` 或 `app/services.py`。

推荐目录：

```text
app/model-gateway/
  __init__.py
  auth.py
  routes.py
  services.py
  repositories.py
  runtime.py
  queue.py
  documents.py
  retrieval.py
  schemas.py
  errors.py
```

模块职责：

| 模块 | 职责 |
|------|------|
| `auth.py` | 用户 API Key、管理员 Token 校验；只返回身份上下文，不做业务 |
| `routes.py` | Flask Blueprint，处理 `/v1/*` 和 `/api/model-gateway/*` 请求/响应 |
| `services.py` | 编排业务流程：创建请求、上传文件、创建科研任务、记录 usage |
| `repositories.py` | 所有 `serving_*` 表读写；路由层不直接 `db.session` |
| `runtime.py` | vLLM 进程启动、停止、健康检查、端口和状态机 |
| `queue.py` | Redis Streams、lease、reconciler、取消和恢复 |
| `documents.py` | 文件保存、原生解析、chunk 生成；旧 Office 由私网 parser 处理 |
| `retrieval.py` | SQLite FTS、FAISS、hybrid retrieve、引用 scope 校验 |
| `schemas.py` | 请求体/响应体校验和序列化；第一版可用轻量函数，不强制引入新框架 |
| `errors.py` | 统一错误码、HTTP 状态和错误响应结构 |

注册方式：

- 在 `app/__init__.py` 中保持现有 `register_routes(app, state, services, dataset_service)`。
- 额外注册 serving blueprint，例如 `register_model_gateway_routes(app)`。
- serving 模块初始化不得自动启动模型服务。
- serving worker/reconciler 是否启动由配置控制，避免 CLI 脚本 import 时误启动后台进程。

依赖边界：

- `routes.py` 只调用 `serving.services`，不直接调用 vLLM、FAISS、Redis 或 DB。
- `services.py` 可以调用 `repositories/runtime/queue/documents/retrieval`。
- `runtime.py` 不依赖 Flask request，不读取用户 payload。
- `documents.py` 不调用模型推理。
- `retrieval.py` 不调用 vLLM，只返回带 `chunk_id` 的上下文片段。
- `repositories.py` 是唯一写 `serving_*` 表的模块。

第一版避免新增过重抽象：

- 不做接口类/工厂模式。
- 不做插件系统。
- 不做完整依赖注入容器。
- 用清晰模块和函数边界即可。

### 4.10 第一版科研任务模板

第一版支持四类任务：

1. 文档问答：围绕上传文件提问。
2. 文档摘要：对单个或多个文件生成结构化摘要。
3. 多文档对比：比较方法、实验、结论、限制和差异。
4. 结构化抽取：抽取指标、数据集、方法、结论、限制等，输出 Markdown 或 JSON。

### 4.11 第一版不做完整 Agent 工具编排

第一版学习成熟 Agent 平台的稳定结构，但不实现完整 Agent 工具循环。

采用的成熟结构：

- Gateway 统一鉴权、路由、usage 和错误码。
- Request/Job 抽象统一管理排队、运行、取消、失败和恢复。
- File -> Chunk -> Retrieval -> Citation 的文档处理链路。
- 持久化状态表 + 队列 + lease/reconciler 的恢复模型。
- 模型运行时与用户请求解耦，用户不直接访问 vLLM。

暂不实现：

- 模型自主选择和调用任意工具。
- 多步 planner/executor 循环。
- 代码执行沙箱。
- 浏览器/联网搜索 Agent。
- 长期服务器端 memory。
- OpenAI Assistants 或 File Search API 的全量兼容。

原因：

- 当前首要目标是让融合模型稳定可用，而不是构建全功能 Agent 框架。
- 工具编排会引入权限、沙箱、审计、重试和安全边界，第一版难以可靠验收。
- 文档问答、摘要、对比和结构化抽取已经覆盖第一版科研任务主路径。

---

## 5. Token 与 API Key

### 5.1 管理员 Token

第一版不做完整账号系统。管理员接口使用环境变量：

- `MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN`

生成方式：

- 使用 Python `secrets.token_urlsafe()` 生成。
- 仅保存在环境变量或部署配置中。
- 不写入数据库。

权限边界：

- 只能访问 `/api/model-gateway/admin/*`。
- 不用于普通用户推理接口。
- 不透传给 vLLM。
- 轮换管理员 Token 后，旧 Token 立即失效。

门户保存策略：

- 管理员 Token 可以长期保存在当前浏览器，方便后续管理。
- 勾选“记住管理员 Token”时，前端保存到 `localStorage`。
- 未勾选时，前端只保存到 `sessionStorage`，浏览器会话结束后失效。
- 门户提供“保存 Token”、“测试权限”、“清除 Token”三个操作。
- 保存前必须提示：仅在可信设备上长期保存管理员 Token。
- 管理员 Token 不写入服务器数据库，不和用户 API Key 混用。
- 管理员 Token 只用于门户调用 `/api/model-gateway/admin/*`，不用于 `/v1/*` 用户推理。

### 5.2 用户 API Key

用户 Key 由管理员生成：

- 前缀：`mk_live_`
- 明文只展示一次。
- 数据库存储 SHA-256 hash、prefix、last4、状态、模型 allowlist。
- 第一版只做模型 allowlist，不做套餐、角色、余额和扣费。

Key 生命周期状态：

| 状态 | 含义 | 新请求行为 | 是否可恢复 |
|------|------|------------|------------|
| `active` | 正常可用 | 允许继续鉴权 | 是 |
| `disabled` | 暂停使用 | 拒绝新请求 | 是 |
| `revoked` | 永久撤销 | 拒绝新请求 | 否 |

创建规则：

1. 管理员填写 `owner_label`、可选 `expires_at`、模型 allowlist 和备注。
2. 系统生成随机明文 Key，例如 `mk_live_<random>`。
3. 明文只在创建响应中返回一次。
4. 数据库只保存 `key_hash`、`prefix`、`last4`、状态和 allowlist。
5. 前端必须提示管理员：关闭弹窗后无法再次查看明文 Key。

鉴权流程：

1. 读取 `Authorization: Bearer mk_live_xxx`。
2. 对明文 Key 做 SHA-256 后查询 `serving_api_keys.key_hash`。
3. 检查 Key 状态为 `active`。
4. 检查 `expires_at` 为空或尚未过期。
5. 根据请求中的 `model` 找到 `serving_model_services`。
6. 检查目标模型在 `model_allowlist` 中。
7. 更新 `last_used_at`。
8. 通过后继续进入 Gateway、队列或 vLLM。

禁用和撤销规则：

- `disabled` / `revoked` 后，新请求立即拒绝。
- 已排队但未开始的任务进入 `canceled`。
- 已运行任务标记 `cancel_requested`，尽力停止；如果已转发到 vLLM，则丢弃结果。
- 已完成 usage 不删除，后续账单仍可追溯。

默认安全限制：

- 单个 API Key 默认绑定 1-3 个模型。
- 单请求 `max_tokens` 默认上限 4096，可由管理员按模型调整。
- VLM 每次请求最多 1 张图片。
- 文件上传默认单文件 50MB，上限通过配置调整。
- API Key 不能访问 `/api/model-gateway/admin/*`。

### 5.3 vLLM 内部 Key

每个 vLLM 服务使用内部 key，只允许 Gateway 调用。用户 API Key 不直接透传给 vLLM。

内部 Key 规则：

- 每个模型服务启动时生成内部 Key。
- Gateway 调用 vLLM 时使用该内部 Key。
- 用户 API Key 永远不透传到 vLLM。
- 模型服务停止或重启后可以重新生成内部 Key。

---

## 6. 数据库表设计

第一版不引入用户账号表。用户身份边界以 API Key 为准，后续接支付和账号系统时，再把 API Key 绑定到正式用户。

新增表统一使用 `serving_*` 前缀，不改现有 `tasks`、`models`、`testsets`、`evaluation_results` 的职责。

### 6.1 `serving_model_services`

记录一个模型发布服务。`models` 表仍是模型注册真相，本表只表示“该模型被发布为可调用服务”。

| 字段 | 类型建议 | 说明 |
|------|----------|------|
| `id` | string uuid | 主键 |
| `model_id` | string fk nullable | 关联 `models.id`，模型被移除时可置空保留历史 |
| `model_path` | string | 启动 vLLM 使用的模型路径快照 |
| `display_name` | string | 门户展示名 |
| `served_model_name` | string unique | `/v1/models` 暴露名称 |
| `backend_type` | string | 第一版固定 `local_vllm`；预留 `external_openai_compatible` |
| `model_type` | string | `text` / `vlm` |
| `status` | string index | `stopped` / `starting` / `running` / `stopping` / `failed` / `deleted` |
| `vllm_host` | string | 固定 `127.0.0.1` |
| `vllm_port` | integer nullable | 系统分配端口 |
| `vllm_pid` | integer nullable | 当前进程 PID |
| `gpu_ids` | json | 管理员选择的 GPU |
| `tensor_parallel_size` | integer | vLLM TP |
| `vllm_args` | json | 白名单参数快照 |
| `internal_api_key_hash` | string | Gateway 调 vLLM 的内部 key hash |
| `last_error` | text nullable | 最近失败摘要 |
| `last_exit_reason` | string nullable | 例如 `system_restarted_manual_recovery_required` |
| `last_status_change_at` | datetime | 最近状态变化时间 |
| `created_at` / `updated_at` | datetime | 审计字段 |

必要索引：

- `served_model_name`
- `status`
- `model_id`

### 6.2 `serving_api_keys`

记录用户 API Key。第一版不保存明文 key。

| 字段 | 类型建议 | 说明 |
|------|----------|------|
| `id` | string uuid | 主键 |
| `key_hash` | string unique | SHA-256 hash |
| `prefix` | string | 例如 `mk_live_` |
| `last4` | string | 展示和排障 |
| `owner_label` | string nullable | 管理员填写的使用方名称 |
| `status` | string index | `active` / `disabled` / `revoked` |
| `model_allowlist` | json nullable | 允许访问的 `serving_model_services.id` 列表；空表示不允许 |
| `notes` | text nullable | 管理员备注 |
| `last_used_at` | datetime nullable | 最近使用时间 |
| `expires_at` | datetime nullable | 可选过期时间 |
| `created_at` / `updated_at` | datetime | 审计字段 |

必要索引：

- `key_hash`
- `status`
- `expires_at`

### 6.3 `serving_requests`

统一记录聊天请求、科研任务和文档问答任务。正文不长期保存在 DB；大 payload 走临时文件路径和 TTL。

| 字段 | 类型建议 | 说明 |
|------|----------|------|
| `id` | string uuid | request/job ID |
| `request_type` | string index | `chat` / `research` / `document_qa` |
| `status` | string index | 请求生命周期状态 |
| `api_key_id` | string fk nullable | 关联 `serving_api_keys.id` |
| `model_service_id` | string fk nullable | 关联 `serving_model_services.id` |
| `idempotency_key` | string nullable | 客户端幂等 key |
| `payload_path` | string nullable | 临时 payload 文件 |
| `result_path` | string nullable | 临时结果文件 |
| `payload_expires_at` | datetime nullable | 正文清理时间 |
| `lease_owner` | string nullable | worker 标识 |
| `lease_expires_at` | datetime nullable | lease 到期时间 |
| `retry_count` | integer | 已重试次数 |
| `max_retries` | integer | 最大重试次数 |
| `cancel_requested_at` | datetime nullable | 取消请求时间 |
| `error_code` | string nullable | 机器可读错误 |
| `error_summary` | text nullable | 不含用户正文的错误摘要 |
| `created_at` / `updated_at` | datetime | 审计字段 |
| `started_at` / `finished_at` | datetime nullable | 执行时序 |

必要索引：

- `status`
- `api_key_id`
- `model_service_id`
- `(api_key_id, model_service_id, idempotency_key)`
- `lease_expires_at`
- `payload_expires_at`

### 6.4 `serving_usage_records`

记录 token 使用量，为后续计费做准备，不保存用户正文。

| 字段 | 类型建议 | 说明 |
|------|----------|------|
| `id` | string uuid | 主键 |
| `request_id` | string fk nullable | 关联 `serving_requests.id` |
| `api_key_id` | string fk nullable | 关联 API Key |
| `model_service_id` | string fk nullable | 关联模型服务 |
| `served_model_name` | string | 模型名快照 |
| `prompt_tokens` | integer nullable | 输入 token |
| `completion_tokens` | integer nullable | 输出 token |
| `total_tokens` | integer nullable | 总 token |
| `usage_source` | string | `vllm` / `estimated` / `missing` |
| `created_at` | datetime | 记录时间 |

必要索引：

- `request_id`
- `api_key_id`
- `model_service_id`
- `created_at`

### 6.5 `serving_files`

记录上传文件及解析状态。

| 字段 | 类型建议 | 说明 |
|------|----------|------|
| `id` | string uuid | file ID |
| `api_key_id` | string fk nullable | 所属 API Key |
| `original_filename` | string | 原文件名 |
| `mime_type` | string nullable | MIME |
| `file_ext` | string | 扩展名 |
| `size_bytes` | integer | 文件大小 |
| `status` | string index | `uploaded` / `parsing` / `parsed` / `indexing` / `indexed` / `index_failed` / `expired` / `deleted` |
| `raw_path` | string nullable | 原文件临时路径 |
| `parsed_text_path` | string nullable | 解析文本临时路径 |
| `parse_error` | text nullable | 解析失败摘要 |
| `chunk_count` | integer | chunk 数 |
| `expires_at` | datetime nullable | 文件数据 TTL |
| `created_at` / `updated_at` | datetime | 审计字段 |

必要索引：

- `api_key_id`
- `status`
- `expires_at`

### 6.6 `serving_document_chunks`

记录可引用的文档片段。SQLite FTS 可基于本表或镜像 FTS 虚表构建。

| 字段 | 类型建议 | 说明 |
|------|----------|------|
| `id` | string uuid | 主键 |
| `file_id` | string fk | 关联 `serving_files.id` |
| `api_key_id` | string fk nullable | 冗余所属边界，方便检索隔离 |
| `chunk_id` | string unique | 例如 `file_abc:p12:c03` |
| `chunk_index` | integer | 文件内顺序 |
| `page_number` | integer nullable | PDF 页码 |
| `slide_number` | integer nullable | PPT 页码 |
| `paragraph_index` | integer nullable | DOC 段落 |
| `text` | text | chunk 文本 |
| `text_hash` | string | 去重/索引校验 |
| `chunk_metadata` | json nullable | 标题、来源等 |
| `created_at` | datetime | 创建时间 |

必要索引：

- `file_id`
- `api_key_id`
- `chunk_id`
- `text_hash`

### 6.7 `serving_vector_indexes`

记录 FAISS 索引元数据。FAISS 文件本身不是事实源，可以从 `serving_document_chunks` 重建。

| 字段 | 类型建议 | 说明 |
|------|----------|------|
| `id` | string uuid | 主键 |
| `file_id` | string fk | 关联文件 |
| `api_key_id` | string fk nullable | 所属边界 |
| `embedding_model` | string | embedding 模型名 |
| `embedding_dim` | integer nullable | 向量维度 |
| `index_path` | string | FAISS 文件路径 |
| `status` | string index | `building` / `ready` / `failed` / `stale` / `deleted` |
| `chunk_count` | integer | 索引 chunk 数 |
| `is_rebuildable` | boolean | 第一版必须为 true |
| `last_error` | text nullable | 失败摘要 |
| `created_at` / `updated_at` | datetime | 审计字段 |

必要索引：

- `file_id`
- `api_key_id`
- `status`
- `(file_id, embedding_model)`

### 6.8 `serving_events`

记录服务侧审计事件，只保存摘要，不保存用户正文。

| 字段 | 类型建议 | 说明 |
|------|----------|------|
| `id` | string uuid | 主键 |
| `event_type` | string index | `model_status_changed` / `request_retried` / `request_canceled` / `index_failed` 等 |
| `model_service_id` | string nullable | 相关模型服务 |
| `request_id` | string nullable | 相关请求 |
| `api_key_id` | string nullable | 相关 API Key |
| `severity` | string | `info` / `warning` / `error` |
| `summary` | text | 不含用户正文的摘要 |
| `event_metadata` | json nullable | 安全元数据 |
| `created_at` | datetime | 事件时间 |

必要索引：

- `event_type`
- `model_service_id`
- `request_id`
- `api_key_id`
- `created_at`

### 6.9 数据库设计约束

- 状态字段使用字符串，不使用数据库 enum，保持 SQLite 友好。
- 新表都加 `created_at`，有更新行为的表加 `updated_at`。
- 大文本正文、原始文件和解析结果默认不长期放 DB。
- DB 保存路径、状态、usage、错误摘要和可恢复所需信息。
- 所有检索、请求、文件访问必须以 `api_key_id` 作为第一版隔离边界。
- 用户侧请求、文件和 chunk 的 `api_key_id` 必须非空；只有系统级审计事件允许为空。
- ORM 字段避免使用 SQLAlchemy 易冲突名称，例如不用 `metadata`，改用 `chunk_metadata` / `event_metadata`。
- 后续账号系统接入时，新增用户表并把 `serving_api_keys.owner_label` 升级为外键，不改请求主流程。

---

## 7. 文件解析与引用定位

### 7.1 支持格式

第一版支持：

- PDF
- DOC
- DOCX
- PPT
- PPTX

解析策略：

- 新格式优先直接解析。
- 旧格式 `.doc` / `.ppt` 由私网 Apache POI parser 直接提取文本；DOC 引用段落，PPT 引用幻灯片。
- 解析依赖放在独立文档解析环境中，避免污染主 `mergenetic` 环境。

### 7.2 Chunk 来源结构

文件解析后生成带来源信息的 chunk：

```text
chunk_id: file_abc:p12:c03
file_id: file_abc
filename: paper.pdf
page: 12
chunk_index: 3
text: ...
```

不同文件的定位方式：

- PDF：页码 + chunk 编号 + 原文摘录。
- PPT/PPTX：幻灯片编号 + chunk 编号。
- DOC/DOCX：优先转换后按页定位；失败时退回标题/段落编号。
- 图片/VLM：第一版按整体图片输入，不做局部区域坐标引用。

### 7.3 引用规则

科研任务采用以下规则：

- 涉及文档事实的内容必须带引用。
- 模型自己的解释、归纳、建议、组织性表达不强制引用。
- 数字、实验结果、数据集、方法描述、作者结论必须引用。
- 引用格式使用 chunk_id，例如 `[file_abc:p12:c03]`。

### 7.4 引用校验

输出后做轻量校验：

- 引用 ID 必须真实存在。
- 引用 ID 必须属于本用户、本任务允许的文件集合。
- 引用不存在或越权时，自动重试一次。
- 重试后仍失败，返回结果但标记 `citation_invalid`。
- 如果科研任务包含文档事实但完全没有引用，标记 `citation_missing`。

第一版不承诺 PDF 精确坐标高亮，只承诺：

```text
答案 -> 文件 -> 页码/幻灯片/段落 -> 原文片段
```

---

## 8. 混合检索设计

### 8.1 标准方案

第一版采用：

```text
SQLite 元数据 + SQLite FTS 保底 + FAISS 向量增强 + 引用校验兜底
```

### 8.2 可靠性原则

- SQLite 是唯一事实源，保存文件、chunk、页码、用户归属和解析状态。
- FAISS 只保存向量索引，是可重建的增强层。
- FAISS 索引损坏时，可以从 SQLite chunk 重新生成。
- embedding 或 FAISS 失败时，自动降级到 FTS。
- FTS 和 FAISS 都没有召回时，不让模型凭空回答。

### 8.3 召回策略

默认策略：

- FTS top 20
- FAISS top 20
- 合并去重
- 重排后最多送入模型 8-15 个 chunk
- 超过 token 限制时按分数截断

检索必须带 scope：

- api_key_id
- request_id 或 job_id
- file_id allowlist
- user/session scope

严禁跨用户、跨任务召回。

### 8.4 Embedding 策略

第一版 embedding 默认跑 CPU，优先小模型和稳定性。

原则：

- 不抢占 vLLM GPU。
- embedding 失败不影响主流程。
- 后续可由管理员配置 embedding 模型。

---

## 9. 队列与恢复

### 9.1 队列形态

采用 Redis Streams + DB 状态表：

- Redis Streams 负责执行队列和 worker 分发。
- DB 是任务状态、幂等、结果和恢复的唯一事实源。
- Worker 使用 lease/heartbeat。
- Reconciler 恢复 lease 过期任务。

设计原则：

- 先写 DB，再写 Redis Stream。
- Redis Stream 消息只保存 `request_id`、`model_service_id`、`request_type` 等轻量字段，不保存 prompt、文件正文或解析结果。
- Redis 可以重启或丢失短窗口消息，但不能导致任务永久丢失；DB Reconciler 必须能补投递。
- 第一版采用 at-least-once execution，不承诺 exactly-once；结果提交必须幂等。

### 9.2 Redis 部署与持久化

Docker Compose 增加独立 Redis 服务，开启 AOF：

```yaml
redis:
  image: redis:7-alpine
  command: ["redis-server", "--appendonly", "yes", "--appendfsync", "everysec"]
  volumes:
    - ./mergeKit_beta/redis_data:/data
  restart: unless-stopped
```

`mergekit-beta` 增加环境变量：

```text
MERGEKIT_MODEL_GATEWAY_REDIS_URL=redis://redis:6379/0
MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND=redis
```

持久化策略：

- `appendfsync everysec` 是性能和耐久性的折中。
- Redis 最多可能丢失秒级消息，因此不能作为唯一事实源。
- `redis_data/` 是运行数据，后续应加入忽略规则，不进入 Git。

### 9.3 投递、消费与确认

标准流程：

```text
Gateway
  -> INSERT serving_requests(status=queued)
  -> XADD serving:requests request_id=...
  -> Worker XREADGROUP
  -> DB claim lease + status=running
  -> 执行任务
  -> DB 写 completed/failed/canceled + usage
  -> XACK
```

规则：

- Gateway 写 DB 成功但写 Redis 失败时，返回 `503 queue_unavailable`，并由 Reconciler 后续补投递。
- Worker 消费 Redis 后，必须先在 DB claim lease，claim 失败则 `XACK` 或跳过该消息。
- Worker 完成任务后，先写 DB 终态，再 `XACK`。
- 如果 `XACK` 失败，任务可能再次投递；DB 终态必须阻止重复结果生效。
- Consumer group 名称固定，例如 `serving-workers`；consumer 名称使用 `hostname:pid`。

### 9.4 Reconciler 恢复规则

Reconciler 定时扫描 DB，而不是只相信 Redis pending 列表。

需要补投递的状态：

- `queued`
- `retrying`
- `paused_model_offline` 且目标模型已 `running`
- lease 过期的 `running`

不得补投递的状态：

- `completed`
- `canceled`
- `failed`
- `dead_letter`
- `expired`

Redis pending 恢复：

- Worker 崩溃后，消息会留在 consumer group pending 中。
- Reconciler 或 worker 可 claim 超时 pending 消息。
- claim 前必须检查 DB 状态和 lease，避免恢复已取消或已完成任务。

### 9.5 Redis 不可用时的行为

生产模式不降级到 Python 内存队列。

原因：

- 内存队列无法可靠重启恢复。
- 取消和幂等语义会变假。
- 用户会看到“已排队”但任务实际丢失。

行为：

- Redis 不可用时，新异步任务返回 `503 queue_unavailable`。
- `/readyz` 应报告 serving queue 不 ready。
- 已运行的 vLLM 服务不自动停止。
- 管理员页面展示 Redis/队列不可用。

开发模式可以允许 DB-only backend：

```text
MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND=redis|db
```

默认和生产推荐为 `redis`。`db` 仅用于本地开发和单元测试。

### 9.6 HTTP 等待规则

- 普通请求最多保持 HTTP 连接等待 60 秒。
- 60 秒内完成则直接返回结果。
- 超过 60 秒返回 `request_id` / `job_id`，用户可查询或取消。

### 9.7 取消规则

用户可以立即取消尚未完成的排队任务：

- queued：立即取消。
- running：尽力取消；如果 vLLM 请求已开始，标记取消并丢弃结果。
- canceled/completed/failed/dead_letter：Reconciler 不再恢复。

### 9.8 重启恢复

系统重启后：

- 未完成任务从 DB 恢复。
- 已取消、已完成、失败终止任务不恢复。
- 模型服务仍等待管理员手动启动。
- 任务可进入等待模型状态，不自动占用 GPU。

### 9.9 请求与科研任务生命周期

用户请求和科研任务必须使用同一套生命周期状态。聊天请求、文档问答和科研任务可以有不同 payload，但状态机一致。

第一版请求状态：

| 状态 | 含义 | 是否可恢复 | 是否可取消 |
|------|------|------------|------------|
| `received` | Gateway 已接收请求，正在鉴权和校验参数 | 是 | 是 |
| `queued` | 请求已写入 DB 和 Redis Stream，等待 worker | 是 | 是 |
| `paused_model_offline` | 目标模型未运行，等待管理员启动 | 是 | 是 |
| `running` | worker 已领取任务并获得 lease | 是，lease 超时后可重入队 | 尽力取消 |
| `streaming` | 正在向用户返回流式结果 | 否，连接断开后转入异步结果或失败 | 尽力取消 |
| `retrying` | 任务因可恢复错误准备重试 | 是 | 是 |
| `completed` | 任务完成，usage 已记录 | 否 | 否 |
| `cancel_requested` | 用户已请求取消，worker 正在停止或丢弃结果 | 是 | 否 |
| `canceled` | 任务已取消，payload 可立即清理 | 否 | 否 |
| `failed` | 任务失败但保留短期排障信息 | 否 | 否 |
| `dead_letter` | 超过重试次数或出现不可恢复错误 | 否 | 否 |
| `expired` | TTL 到期，服务端正文/payload 已清理 | 否 | 否 |

状态流转规则：

```text
received -> queued
queued -> running
queued -> paused_model_offline
paused_model_offline -> queued
running -> streaming
running -> completed
running -> retrying
running -> failed
retrying -> queued
queued/paused_model_offline/running/streaming -> cancel_requested
cancel_requested -> canceled
completed/failed/dead_letter/canceled -> expired
```

模型状态与请求状态的关系：

- 目标模型是 `running`：`queued` 请求可派发给 worker。
- 目标模型是 `stopped`：新请求返回 `503 model_not_running`；已存在异步任务进入 `paused_model_offline`。
- 目标模型是 `starting`：新请求可进入 `queued`，但 worker 不转发到 vLLM。
- 目标模型是 `stopping`：新请求返回 `503 model_stopping`。
- 目标模型是 `failed`：新请求返回 `503 model_failed`；已存在任务进入 `paused_model_offline` 或 `failed`，由任务类型决定。
- 目标模型是 `deleted`：请求返回 `404 model_not_found`，相关未完成任务转入 `dead_letter`。

重启恢复规则：

- `queued`、`paused_model_offline`、`retrying`：恢复为可调度状态。
- `running`：如果 lease 已过期，恢复为 `queued`；如果 lease 未过期，等待 Reconciler 判定。
- `streaming`：HTTP 连接无法恢复，重启后转为 `failed`，错误为 `stream_interrupted_by_restart`。
- `cancel_requested`：恢复后继续完成取消，最终进入 `canceled`。
- `completed`、`canceled`、`failed`、`dead_letter`、`expired`：不恢复。

幂等与重复执行规则：

- 客户端可以传 `Idempotency-Key`。
- 同一 API Key、同一模型、同一 `Idempotency-Key` 在短期窗口内只创建一个任务。
- worker crash 后允许 at-least-once 重试；结果写入必须按 request/job ID 幂等。
- 如果重复执行产生第二个结果，只保留第一个成功提交的结果，后续结果丢弃并记录日志。

取消规则：

- `queued` / `paused_model_offline` / `retrying`：立即进入 `canceled`。
- `running`：标记 `cancel_requested`，worker 停止后进入 `canceled`。
- `streaming`：关闭连接并标记 `cancel_requested`；如果 vLLM 已返回结果，丢弃未提交部分。
- `completed` / `failed` / `dead_letter`：不能取消，只能清理客户端本地历史。

HTTP 60 秒等待规则：

- 60 秒内完成：同步返回最终结果。
- 60 秒内未完成：返回 `202 Accepted`，包含 `request_id` 或 `job_id`。
- 流式请求进入 `streaming` 后，如果连接断开，服务端不尝试恢复同一个 HTTP 流。
- 用户可通过状态查询接口继续获取异步任务状态。

payload 清理规则：

- `completed`：正文/payload 默认 30 分钟后清理。
- `canceled`：正文/payload 立即清理。
- `failed` / `dead_letter`：最多保留 24 小时用于排障。
- `expired`：只保留 usage、状态摘要和不含用户正文的错误摘要。

生命周期验收标准：

- 排队任务可取消，且取消后不会被 Reconciler 恢复。
- worker 崩溃后，lease 过期的 `running` 任务可回到 `queued`。
- 系统重启后，`completed` 和 `canceled` 任务不会重复执行。
- 模型离线时，异步科研任务进入 `paused_model_offline` 而不是卡死。
- 同一 `Idempotency-Key` 重复提交不会产生多个有效任务。
- payload TTL 到期后，服务端不再保留用户正文，但 usage 仍可统计。

---

## 10. 数据保留与隐私

第一版不长期保存用户对话正文。

策略：

- 门户长期会话历史保存在用户浏览器 IndexedDB。
- 服务器只保存队列和恢复所需的临时 payload。
- completed payload 默认 30 分钟后清理。
- canceled payload 立即清理。
- failed/dead_letter payload 最多保留 24 小时用于排障。
- 上传原文件解析成功后可删除原始文件。
- chunk 和 parsed text 在任务完成或取消后按 TTL 清理。
- usage 记录长期保留，但不保存用户原文。

第一版暂不做 app 层加密，付费和敏感内容能力接入前再补。

---

## 11. 前端门户

### 11.1 嵌入式门户

门户嵌入当前 Flask 系统，第一版不拆独立前端工程。

入口：

- 新增 `/model-gateway` 作为统一入口。
- 在现有侧边栏增加“模型服务”导航项。
- 第一版使用一个模板和一个前端脚本承载门户，不拆独立 React/Next/Vue 工程。

页面结构：

- 用户工作台：聊天、文件上传、科研任务、引用查看。
- 文件与科研任务：文件列表、解析状态、索引状态、文档问答、摘要、对比和结构化抽取。
- API Key 页面：管理员创建 Key，明文只展示一次；普通用户可检查 Key 权限和可用模型。
- 管理员发布页：选择融合模型、GPU、vLLM 参数并启动/停止。
- 队列与用量页：查看任务状态、模型状态、token usage。

关键交互：

- 用户聊天历史默认保存在浏览器 IndexedDB，不长期保存在服务器。
- 用户 API Key 明文只在创建后展示一次；关闭弹窗后不可恢复，只能重新生成。
- 管理员 Token 可以选择长期保存在浏览器 `localStorage`，但必须有可信设备提示和一键清除。
- force-stop 模型服务前，要求管理员输入模型服务名确认。
- 删除模型服务前，要求模型已停止，并输入 `served_model_name` 确认。
- 撤销 API Key 是不可逆操作，前端必须明确提示撤销后不能恢复。
- 启动模型前展示 GPU 占用、TP、显存安全参数和 vLLM 参数摘要。

### 11.2 视觉与动画

用户明确要求高级感和流畅动画。实现原则：

- 使用当前系统静态资源结构。
- 使用 taste-skill 做视觉方向约束。
- 使用 GSAP 做关键状态动画。
- 动画只使用 transform/opacity 等低成本属性。
- 支持 `prefers-reduced-motion`。
- 不做营销页，打开即是可用工作台。
- 视觉方向是技术工作台，不做宣传落地页。
- 避免大面积紫蓝渐变；状态颜色优先服务于运行、失败、排队、重试等可观测信息。
- 关键动画只用于状态变化、面板切换、队列流转和错误提示，不影响操作效率。

---

## 12. 第一版验收标准

### 12.1 模型服务

- 管理员能发布并启动一个 Qwen 文本模型。
- Gateway 能通过 `/v1/chat/completions` 调用模型。
- streaming 能正常返回。
- 模型未启动时返回明确 `503 model_not_running`。
- token usage 能记录。

### 12.2 文件与科研任务

- PDF/DOCX/PPTX 至少各有一个样例能解析出 chunk。
- 旧格式 DOC/PPT 能通过私网 parser 路径处理，失败时有明确错误。
- 科研任务回答中的文档事实包含引用。
- 引用 ID 不存在时会被校验发现。
- 删除文件后不能再召回对应 chunk。

### 12.3 检索可靠性

- FAISS 索引删除后可以重建。
- embedding 不可用时降级 FTS。
- FTS 和 FAISS 均无结果时不让模型编造文档事实。
- 同名文件跨用户不会互相召回。

### 12.4 队列恢复

- 任务超过 60 秒时返回 job/request ID。
- 排队任务可取消。
- 服务重启后未完成任务状态可恢复。
- 模型服务重启后仍由管理员手动启动。
- Redis 重启后，DB 中 `queued` / `retrying` 任务可补投递。
- worker 被 kill 后，lease 过期任务可被 Reconciler 恢复。
- 同一任务重复投递不会产生两个有效结果。
- Redis 不可用时，新异步任务返回 `503 queue_unavailable`，不进入假排队。

---

## 13. 暂缓事项

以下事项不进入第一版：

- 真实支付和扣费。
- 长期服务器端对话历史。
- 完整账号密码体系。
- OpenAI Assistants/File Search API 全量兼容。
- 完整 Agent 工具编排、planner/executor 循环和代码执行沙箱。
- PDF 精确坐标高亮。
- 联网搜索。
- 多租户套餐和复杂权限模型。
- 自动启动模型服务。
