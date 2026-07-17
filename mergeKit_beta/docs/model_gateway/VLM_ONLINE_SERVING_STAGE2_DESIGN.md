# Stage 2：VLM 在线服务与文档视觉研究设计

> 状态：设计已确认，尚未实施  
> 核验日期：2026-07-17（Asia/Shanghai）  
> 基线分支：`checkpoint/20260715-platform-gateway`  
> 基线提交：`52de609`  
> 范围：模型工厂到正式发布、VLM 在线推理、图片输入、PDF/PPT 图表理解、门户与管理员联动  
> 非范围：网页图片抓取、OCR/扫描件识别、视频、图片生成、任意 Agent 工具循环、支付计费

## 1. 目标与成功定义

Stage 2 不是单独搭建一个 Qwen2.5-VL 演示服务，而是把模型工厂产生的、经过正式发布门禁的视觉模型接入现有 Model Gateway。

系统需要同时满足：

1. Qwen2/Qwen2.5 文本模型继续稳定运行，不因 VLM 依赖升级发生回归。
2. Qwen2-VL 是第一版必须兼容的视觉架构。
3. Qwen2.5-VL 是可选增强结果；只有融合、验证、发布和运行时兼容性全部通过后才可对外服务。
4. 用户可通过现有 OpenAI-compatible API 和 `/research` 门户完成单图问答。
5. 用户可对 PDF、PPT、PPTX 中经过检索选出的图表页面执行视觉分析，并获得可核验引用。
6. 所有 GPU 操作使用不可变 UUID 作为最终身份，不占用受保护 GPU。
7. 最终验收必须从模型工厂正式 HTTP/任务路径开始，经过配方、发布、Gateway 和用户门户，不能从内部函数或 vendor 脚本中途起跑。

小样本融合和单题视觉推理只能证明链路可用，不能作为模型质量结论。模型质量需使用独立、足量的评测集另行报告。

## 2. 已确认事实

以下事实来自 2026-07-17 的源码、正式资产和运行环境只读核验：

- publication feature 已通过 `2fc0406` 合并到当前 checkpoint，但常驻 5000 主容器启动于合并前，尚未应用新的 published mount 和环境变量。
- 模型工厂主环境为 `/opt/conda/envs/mergenetic`，其中 vLLM 为 0.7.0。
- 当前正式 VLM 资产是完整的 `Qwen2_5_VLForConditionalGeneration` 资产，约 8.29B BF16 参数，权重约 15.82 GiB。
- 该资产已通过结构检查、Transformers 图像推理和 CMMMU 功能样本，但因 vLLM 0.7.0 不支持该架构而保持 `blocked/unsupported_architecture`。
- `app/model_inspection.py` 已识别 `qwen2_vl`、`qwen2_5_vl` 及对应架构。
- `model_composition.py` 不写死 Qwen2.5-VL；它以完整 VLM 为外壳，仅在张量名称和形状严格一致时替换语言塔，并保留视觉塔、processor 和 VLM config。
- `ServingModelService` 已有 `gpu_ids` 和 `gpu_uuids` 字段，但当前服务创建没有填充 UUID，启动时仍把可变 index 写入 `CUDA_VISIBLE_DEVICES`。
- publication GPU 预检已经实现 index、UUID、PCI bus ID、保护集合、允许集合、显存和外部计算进程校验，可抽取身份门禁供 Gateway 复用。
- 当前研究链路使用 PostgreSQL 保存业务状态、Redis Streams 交付任务；源文件和结果受 TTL 管理，模型服务重启后由管理员手动恢复。

## 3. 方案比较与结论

### 3.1 采用：同一镜像、同一主容器、两个 Conda 环境

- 模型工厂、文本评测和文本 vLLM 继续使用 `mergenetic`。
- 视觉在线服务使用 `/opt/conda/envs/vlm-serving`。
- Gateway 根据正式资产的 `artifact_type` 和 manifest 架构选择解释器。
- 复用现有管理员生命周期 API、loopback 端口、PID/PGID 管理、API Key、usage 和研究队列。

优点是不会改变现有服务边界，也不需要把 Docker socket 暴露给应用。代价是镜像体积增加，并且必须严格测试解释器分派。

### 3.2 不采用：原地升级 `mergenetic`

Qwen2.5-VL 所需的 vLLM/Transformers 组合与模型工厂当前依赖存在冲突。原地升级会把 Stage 2 风险扩散到融合、评测和文本服务，因此禁止。

### 3.3 暂不采用：独立 VLM Compose 服务

独立服务隔离更强，但会新增跨容器生命周期控制、GPU 锁、内部认证、端口发现和恢复协议。当前单机内部试用阶段没有足够收益，待多机或独立扩缩容成为真实需求时再评估。

## 4. 运行时与兼容性

### 4.1 候选依赖

首个受控候选环境固定为：

- Python 3.11
- vLLM 0.7.3
- PyTorch 2.5.1
- Transformers 4.49.0 对应固定提交 `a22a4378d97d06b7a1d9abad6e0086d30fdea199`

vLLM 0.7.3 官方模型表同时包含 `Qwen2VLForConditionalGeneration` 和 `Qwen2_5_VLForConditionalGeneration`，且该版本包含 Qwen2.5-VL 修复。不得安装浮动的 Transformers `main`。

上述版本只是第一候选，不是未经验证的生产承诺。候选构建或真实模型验收失败时停止，不自动跳到其他 vLLM 版本。

### 4.2 运行时 profile

运行时 profile 只有两个：

| Profile | 资产 | 解释器 |
| --- | --- | --- |
| `text` | `artifact_type=text` | 现有 `MERGEKIT_MODEL_GATEWAY_PYTHON` |
| `vision` | `artifact_type=vlm` | 新增 `MERGEKIT_MODEL_GATEWAY_VLM_PYTHON` |

不新增 Gateway 数据库列。`ServingModelService.model_type` 已能确定 profile；manifest 的 `compatibility.serving` 可增加可选 `runtime_profile`，旧 schema 1/2 缺失该字段时根据 `artifact_type` 推断。

### 4.3 显式兼容性复检

新增管理员显式操作，建议接口：

```text
POST /api/model-gateway/admin/publishable-models/<model_id>/recheck
```

执行规则：

1. 校验管理员 Token 和正式 published Model 绑定。
2. 根据 artifact type 选择配置的解释器。
3. 以 `CUDA_VISIBLE_DEVICES=`、无网络依赖、有限超时的子进程导入对应 vLLM，并用 ModelRegistry 检查 manifest 中的真实 architectures。
4. 在 publication lock 下原子更新 manifest 的 `compatibility.serving`。
5. 写入 backend、runtime profile、tested version、status、reason code 和 checked time。
6. GET 列表只读，不启动检查子进程、不修改 manifest。

版本或 profile 与配置不一致时返回 `stale`；环境缺失或架构不支持时返回 `blocked`。服务创建和启动仍执行完整资产哈希，因此兼容性复检不能绕过资产完整性门禁。

## 5. GPU 身份与资源门禁

### 5.1 身份规则

Stage 2 的第一项代码任务是修复 Gateway GPU 身份门禁：

1. 管理员仍可通过稳定 UI 选择当前 index。
2. 服务创建时立即解析并保存对应 UUID；API 不接受客户端提交 UUID 作为可信事实。
3. 每次启动重新查询 index、UUID 和 PCI bus ID，并与已保存 UUID 比较。
4. `MERGEKIT_PROTECTED_GPU_UUIDS` 和 `MERGEKIT_PUBLICATION_ALLOWED_GPU_UUIDS` 必须非空、合法且互不重叠。
5. 受保护、不在允许集合、存在外部计算进程或显存不足的 GPU 均拒绝启动。
6. 子进程 `CUDA_VISIBLE_DEVICES` 只写 UUID，不写 index。
7. `gpu_ids` 数量必须严格等于 `tensor_parallel_size`，避免把未使用 GPU 暴露给进程。

实现时从 publication 预检抽取最小的共享 GPU 身份查询/校验能力到 `core/gpu_topology.py`；publication 的显存估算和错误语义保持原样，禁止借机重构发布流程。

### 5.2 VLM 默认资源

当前正式 VLM 权重约 15.82 GiB。24 GiB GPU 在 `gpu_memory_utilization=0.85` 时，vLLM 管理预算约 20.89 GiB。

默认配置：

- TP=1
- `gpu_memory_utilization=0.85`
- `max_num_seqs=1`
- `max_model_len=16384`
- 每次请求最多一张图片
- 解码后图片最多 10 MiB、约 1,003,520 像素，超出时等比例缩放

上下文规则：

| 上下文 | GPU 条件 | 状态 |
| --- | --- | --- |
| 16K | TP=1 | 默认，必须通过真实验收 |
| 32K | TP=1 | 可选，必须通过单卡压力测试 |
| 64K | TP=2 | 可选，必须通过双卡真实长上下文验收后才在 UI 开放 |

该模型 16K BF16 KV cache 估算约 896 MiB。64K KV cache 约 3.5 GiB，单卡安全余量不足，因此不向单卡开放。

## 6. 直接图片问答

### 6.1 API 契约

继续使用 `/v1/chat/completions`，采用 OpenAI-compatible structured message content：文本块和一个 `image_url` data URL。第一版只允许：

- JPEG
- PNG
- WebP
- base64/data URL
- 单张图片

拒绝远程图片 URL、`file://`、SVG、GIF、视频和多图。API 客户端必须明确选择视觉模型；只有用户门户可按已确认规则提示或自动切换模型。

### 6.2 校验与生命周期

Gateway 在转发前执行：

1. 严格 base64 解码和解码后字节上限。
2. Pillow 格式验证、像素上限和解压炸弹保护。
3. 统一色彩空间和等比例缩放。
4. 重新编码以移除 EXIF 等非必要元数据。
5. 确认目标服务 capability 包含 vision。

图片只存在于当前请求内存中，不写入服务器磁盘、数据库、usage 或日志。门户当前标签页内可以复用图片；刷新、关闭标签页或明确移除后图片消失。长期本机会话归档只保留“图片已失效”的元数据，不保存 base64。

## 7. PDF/PPT 文档视觉研究

### 7.1 支持边界

第一版视觉页面来源：

- PDF
- PPT
- PPTX

DOC/DOCX 保持文本解析。网页 URL 使用正文、表格、标题、图片 alt/caption 等文本信息，不下载网页中的实际图片。扫描件/OCR 不在本阶段范围。

### 7.2 原件和渲染

当前 file worker 在成功解析后删除原件。Stage 2 改为：

- 文件通过隔离、ClamAV 和格式校验后，PDF/PPT/PPTX 原件移动到 API Key 私有的 `visual_sources/<file_id>/`。
- 原件权限为 0600，目录权限为 0700，最多保留现有 24 小时 TTL。
- 用户关闭浏览器不删除资料；明确删除或 TTL 到期才删除。
- 页面渲染图只用于当前任务，成功、失败、取消或恢复清理时立即删除。

不引入 LibreOffice。扩展现有无 GPU、私网隔离的 Java Apache POI 服务：

- PPT 使用 HSLF。
- PPTX 使用 XSLF。
- PDF 使用 Apache PDFBox。
- 内部 `POST /render` 只接受已鉴权的本地文件引用和有限 locator，返回有界 PNG 结果。

渲染服务保持无外网、只读根文件系统、资源限制和 Worker Token 边界。

### 7.3 检索和视觉分析

1. 使用现有 BGE-M3、PostgreSQL 词法检索和 FAISS 混合召回文本 chunk。
2. 从命中的 chunk locator 中按检索排名选出不同页码/幻灯片。
3. 首轮最多 4 页，每次继续增加 4 页，单任务最多 12 页，总视觉步骤超时 300 秒。
4. 每页单独渲染并以单图请求调用 VLM。
5. VLM 只提取与用户问题相关的可核验视觉观察，不直接生成最终研究结论。
6. 视觉观察继承原 file ID 和 page/slide locator，作为临时证据加入最终综合提示。
7. 最终回答继续使用 `[S<n>]` 引用；不新增第二套引用格式。

### 7.4 继续分析与结果状态

`ResearchJob` 不新增数据库字段。TTL 结果 JSON 增加 `visual_state`：

- 已分析 locator
- 临时视觉观察
- 已分析数量
- 下一批位置
- 最大页数
- 警告列表

继续分析创建一个新的不可变 ResearchJob，并引用上一任务 ID；它必须属于同一 API Key、仍在 TTL 内、使用相同资料范围和兼容视觉模型。PNG 和 base64 永远不写入 `visual_state`。

## 8. 队列、取消、usage 与故障语义

- PostgreSQL 继续作为 ResearchJob 最终状态真相，Redis 只负责交付。
- 排队任务可立即取消；运行任务进入 `cancel_requested`，在检索、渲染、每次 VLM 调用和最终提交前检查取消状态。
- 被取消任务不得提交最终结果；已生成的页面图立即清理。
- 所有视觉子调用和最终综合调用的 prompt/completion tokens 汇总后幂等写入 usage，Redis 重投和任务重试不能重复记账。
- 模型服务离线时任务进入现有 `paused_model_offline`，管理员手动启动后才能恢复；系统重启不自动加载模型。
- 直接图片推理失败时整个请求失败。
- 文档中单页视觉分析失败时继续处理其他页。
- 若全部视觉分析失败但仍有文本证据，返回明确标记的仅文本结果，并附 `visual_analysis_unavailable` 警告；不得暗示图表已经被阅读。
- 引用缺失、引用越界或结构化 JSON 非法时，结果不得提交。

稳定错误码至少覆盖：

- `serving_runtime_unavailable`
- `version_changed`
- `unsupported_architecture`
- `protected_gpu`
- `unapproved_gpu`
- `gpu_identity_changed`
- `gpu_busy`
- `insufficient_gpu_memory`
- `model_not_vision_capable`
- `invalid_image_payload`
- `image_too_large`
- `visual_render_failed`
- `visual_analysis_unavailable`
- `visual_limit_reached`
- `visual_timeout`

普通用户错误不得包含模型私有路径、内部端口、UUID 全值或堆栈。管理员事件可以记录脱敏诊断信息。

## 9. 管理员和用户门户

### 9.1 管理员门户

- 新增管理员 GPU inventory API，返回 index、UUID 脱敏尾段、PCI bus ID、显存、占用和 allowed/protected 状态。
- 服务创建由手写 GPU index 改为可选择 GPU 项；保护或未批准设备显示但禁用。
- 正式资产展示 architecture、capabilities、runtime profile 和兼容性状态。
- `stale/blocked` 资产提供显式“重新检查”。
- VLM 表单提供安全的 TP/上下文组合：单卡 16K/32K，双卡 64K。
- 不新增用户门户到管理员门户的跳转。

### 9.2 用户门户

- `/v1/models` 保留现有 OpenAI-compatible 字段，并增加客户端可忽略的 `capabilities`。
- 输入框右侧现有“添加资料”工具栏增加图片入口和内存预览。
- 当前模型支持 vision 时直接使用；当前为文本模型且只有一个获授权、正在运行的 VLM 时自动切换并提示；多个 VLM 时要求用户选择；没有运行中的 VLM 时阻止提交。
- 移除图片后不自动切回原文本模型。
- 文档是否进入视觉步骤取决于用户所选模型；选择文本模型时继续执行 Stage 1 文本研究。
- 保留侧边栏、本机会话归档、资料复用和引用展开。

### 9.3 视觉与动画

延续当前研究门户和管理员门户的配色、字体和设计语言，不重新换肤。增加状态驱动动画：

- 附件和图片 chip 进入/移除
- 模型自动切换提示
- 服务兼容性复检
- 排队、检索、视觉分析和综合阶段过渡
- 引用展开、侧栏切换和按钮 hover/focus/press
- 成功、警告和失败状态转换

动效使用短促、略带弹性的节奏增强鲜活感，但不运行长期背景循环。实现只动画 transform 和 opacity，使用 GSAP matchMedia 支持 `prefers-reduced-motion`，并确保移动端、桌面端无组件遮挡或布局抖动。

## 10. 配置项

新增配置均需在代码、`.env` 示例或运维文档旁提供中文说明、安全值和调整后果：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `MERGEKIT_MODEL_GATEWAY_VLM_PYTHON` | `/opt/conda/envs/vlm-serving/bin/python` | VLM 隔离解释器 |
| `MERGEKIT_MODEL_GATEWAY_VLM_VLLM_VERSION` | `0.7.3` | GET 列表判断兼容性是否 stale，不替代真实探测 |
| `MERGEKIT_MODEL_GATEWAY_VLM_START_TIMEOUT_SECONDS` | `300` | 7B VLM 首次加载上限 |
| `MERGEKIT_MODEL_GATEWAY_IMAGE_MAX_BYTES` | `10485760` | base64 解码后图片最大 10 MiB |
| `MERGEKIT_MODEL_GATEWAY_IMAGE_MAX_PIXELS` | `1003520` | 单图约 100 万像素 |
| `MERGEKIT_MODEL_GATEWAY_VISUAL_INITIAL_PAGES` | `4` | 首轮视觉页数 |
| `MERGEKIT_MODEL_GATEWAY_VISUAL_PAGE_BATCH` | `4` | 每次继续分析页数 |
| `MERGEKIT_MODEL_GATEWAY_VISUAL_MAX_PAGES` | `12` | 单研究任务硬上限 |
| `MERGEKIT_MODEL_GATEWAY_VISUAL_TIMEOUT_SECONDS` | `300` | 单研究任务视觉步骤总超时 |

安全边界不得仅依赖前端；所有值在后端再次校验，并设置不可被环境变量突破的硬上限。

## 11. 分阶段实施顺序

1. **P0 部署基线**：按既有受控方案重建合并后主容器，确认 published mount、环境变量、数据库注册和 358 项历史测试基线。
2. **P1 GPU 身份门禁**：先修复服务创建和启动的 UUID/PCI/保护集合校验，再接入任何新 VLM 进程。
3. **P2 双运行时与兼容性**：构建候选镜像、增加 vision profile、显式兼容性复检和管理员状态展示。
4. **P3 直接图片问答**：实现 structured content 校验、单图转发、capabilities 和门户图片交互。
5. **P4 文档视觉研究**：保留视觉原件、扩展 Java renderer、视觉页选择、临时证据、继续分析和清理。
6. **P5 完整验收**：从模型工厂融合开始走到用户门户，并执行资源清理和回滚演练。

每一阶段独立提交和验收；任一阶段失败时停止，不将后续功能混入修复。

## 12. 验收设计

### Gate 0：当前基线

- `docker compose config --quiet`
- `docker compose --profile research config --quiet`
- 记录 Git 状态、镜像 ID、容器、数据库计数、Redis pending/lag、GPU UUID/bus ID/显存/进程。
- 使用唯一正确解释器运行完整 `unittest`。
- 明确当前常驻容器是否已重建到 publication merge 后版本。

### Gate 1：GPU 和运行时

- 测试 index 重排、UUID 变化、PCI 变化、保护/允许集合重叠、重复 GPU、繁忙 GPU 和显存不足。
- 验证 CUDA 子进程只收到 UUID。
- 候选镜像中分别验证 `mergenetic` 与 `vlm-serving` 的解释器和包版本。
- Qwen2 文本运行时命令保持不变。
- Qwen2-VL 和 Qwen2.5-VL 使用各自真实 architecture 进行无 GPU ModelRegistry 探测。

### Gate 2：API、存储与安全

- 测试合法 JPEG/PNG/WebP、伪 MIME、非法 base64、超限字节、超限像素、解压炸弹、SVG/GIF 和远程 URL。
- 验证图片不写入磁盘、数据库或日志。
- 验证 PDF/PPT/PPTX 原件只在扫描成功后进入私有 TTL 目录。
- 模拟成功、失败、取消、Worker 崩溃和 TTL，确认渲染图片与原件按规则清理。
- 验证跨 API Key 文件、任务、结果和视觉状态访问返回 404。
- 验证重试、Redis pending 恢复和继续分析不产生重复 usage。

### Gate 3：前端

- `node --check` 覆盖所有修改 JS。
- 桌面和移动浏览器检查模型选择、图片附件、资料复用、任务状态、警告、引用和取消。
- 检查键盘焦点、屏幕阅读标签、hover/press、reduced-motion 和无布局重叠。
- 检查空闲页面无长期动画循环。

### Gate 4：真实 VLM 在线服务

- 验收前确认 GPU 2/受保护 UUID 未变化且无任务影响。
- 管理员通过正式 API 对 published VLM 执行兼容性复检、创建服务和手动启动。
- TP=1 真实验证 16K、base64 图片问答、PDF 图表和 PPT/PPTX 幻灯片研究。
- TP=1 单独压力测试 32K。
- TP=2 真实验证 64K 长上下文加图片后，才开放 64K UI 选项。
- 通过正式 Qwen2 文本服务重复文本聊天和文档研究，确认 Stage 1 无回归。

### Gate 5：从模型工厂源头到门户

这是最终交付门禁，必须使用系统正式入口：

1. 通过模型工厂清单选择实际存在、语言签名严格兼容的 Qwen2 系父模型和一个完整 VLM base；找不到兼容组合时停止，不静默下载或转换。
2. 通过 `/api/merge_evolutionary` 注册一个小规模但真实的 VLM 进化融合任务，使用真实 CMMMU 图像样本。
3. 等待任务正常结束，核对任务状态、日志、GPU 快照、配方、父模型顺序、指纹、VLM base、visual weight 和语言签名。
4. 按产品规则确认临时融合产物被清理而配方保留。
5. 通过 `/api/model-publications` 从配方物化完整 VLM；通过显式 GPU validation、真实图像推理、CMMMU 样本、manifest schema 2 和原子发布。
6. 通过管理员兼容性复检将该资产置为 ready；不允许手工编辑 manifest。
7. 通过管理员 API 创建并启动服务。
8. 使用新建的临时用户 API Key，从 `/research` 和 `/v1/chat/completions` 完成图片问答和带引用文档研究。
9. 取消一个排队任务和一个运行任务，验证没有错误终态、重复 usage 或临时图片残留。
10. 停止服务、撤销临时 Key，并确认 vLLM/Ray/浏览器进程清理、GPU 显存回落、Redis/PostgreSQL 状态一致。

该 Gate 的融合样本规模用于链路验收。必须在报告中把“功能链路通过”和“模型质量达到目标”分开表述。

## 13. 停止条件与回滚

出现以下任一情况立即停止当前阶段：

- 受保护 GPU 状态变化。
- index 解析到不同 UUID/PCI bus ID。
- Qwen2 文本融合、评测或服务测试新增失败。
- 正式资产哈希或 manifest 身份异常。
- 跨 API Key 数据泄露。
- 原件、base64 或渲染图超出生命周期仍存在。
- 引用指向不存在的页码/幻灯片。
- vLLM OOM、驱动错误或停止后显存不回落。
- 候选镜像构建需要未批准的依赖版本跳跃。

回滚原则：

- 候选镜像使用独立标签，保留当前镜像 ID。
- 不原地修改运行容器环境。
- 不删除或移动正式 published asset。
- 通过管理员 API 停止 VLM 服务，再恢复旧镜像和 Compose 配置。
- 新 manifest compatibility 在旧运行时下会变为 stale，不能被误启动。
- 本设计不要求 Gateway schema migration；若实施中发现必须新增 schema，停止并另行设计迁移与回滚。
- 文档 renderer、VLM profile 和前端能力都应可独立关闭，不影响模型工厂和 Stage 1 文本研究。

## 14. 实施时使用的 Skills

实施记录继续写入 `docs/model_gateway/IMPLEMENTATION_SKILLS.md`。至少使用：

- `brainstorming`：需求与边界确认
- `ponytail`：最小复杂度和依赖控制
- `test-driven-development`：GPU、API、安全和生命周期测试先行
- `systematic-debugging`：真实模型、vLLM、渲染和队列故障定位
- `frontend-design`、`design-taste-frontend`：保持既有设计语言并提升视觉层级
- `gsap-core`、`gsap-performance`：状态动效、reduced-motion 和性能
- `requesting-code-review`：阶段完成后的独立审查
- `verification-before-completion`：只依据当前会话新鲜证据声明完成

## 15. 参考

- vLLM 0.7.3 release：<https://github.com/vllm-project/vllm/releases/tag/v0.7.3>
- vLLM 0.7.3 supported models：<https://docs.vllm.ai/en/v0.7.3/models/supported_models.html>
- Apache POI slideshow rendering：<https://poi.apache.org/components/slideshow/how-to-shapes.html>
- 现有 Gateway 架构：[`ARCHITECTURE.md`](ARCHITECTURE.md)
- 正式模型发布设计：[`MODEL_PUBLICATION_PIPELINE_DESIGN.md`](MODEL_PUBLICATION_PIPELINE_DESIGN.md)
- 模型发布验收：[`ACCEPTANCE_20260717_MODEL_PUBLICATION.md`](ACCEPTANCE_20260717_MODEL_PUBLICATION.md)
