# 模型发布管线真实验收记录（2026-07-17）

## 结论

模型发布管线已通过文本模型真实调用、标准 VLM 进化、schema 2 正式发布、
GPU UUID 安全门禁、重启恢复、删除保护和用户不可见性验收。

Qwen2.5-VL 资产已通过 Transformers 图像推理和一条真实 CMMMU 功能样本，
但当前 `vLLM==0.7.0` 不支持 `Qwen2_5_VLForConditionalGeneration`。
管理员端将其标记为 `blocked/unsupported_architecture`；服务创建返回 HTTP 409，
`/v1/models` 和 `/research` 均不展示该模型。VLM 资产发布完成不等于 VLM 在线服务完成。

## 隔离边界

- 工作树：`/home/a/Workspace/Model_factory/.worktrees/model-publication-pipeline`
- 分支：`feature/model-publication-pipeline`
- 隔离 Compose project：`mergekit_publication_task7`
- 隔离端口：`127.0.0.1:5057`
- 第二轮真实验收 DB：`logs/model_gateway/acceptance/20260717_model_publication/runtime-v2/model_factory.db`
- CPU 单测发布目录：`logs/model_gateway/acceptance/20260717_model_publication/runtime/published_models/`
- 正式发布目录：`/home/a/Model_factory_data/published_models/`
- 原始脱敏证据：`logs/model_gateway/acceptance/20260717_model_publication/`
- 允许验证 GPU：UUID `GPU-23348268-...`、`GPU-3f409b14-...`、`GPU-ddf96bec-...`
- 保护 GPU：UUID `GPU-7eff453d-...`（物理 GPU 2）

CPU 单测 override 使用独立 `/data/PublishedModels`，避免 Flask import 时的 startup reconcile
接触正式资产。真实发布 override 只暴露物理 GPU 0/1；GPU 2 不进入容器设备清单。

## 关键命令

所有真实操作均通过 Compose、HTTP 和系统解释器执行。管理员 token 只从忽略文件读取：

```bash
EVIDENCE=mergeKit_beta/logs/model_gateway/acceptance/20260717_model_publication
TOKEN="$(cat "$EVIDENCE/.admin-token")"

TASK7_ADMIN_TOKEN="$TOKEN" docker compose \
  --env-file /home/a/Workspace/Model_factory/.env \
  -p mergekit_publication_task7 \
  -f docker-compose.yml -f "$EVIDENCE/compose-task7-gpu.yml" \
  up -d --force-recreate mergekit-beta

curl -H 'Content-Type: application/json' \
  --data-binary @"$EVIDENCE/vlm-evolution-v2-request.json" \
  http://127.0.0.1:5057/api/merge_evolutionary

curl -H "Authorization: Bearer $TOKEN" \
  -H 'Idempotency-Key: task7-vlm-provenance-v2-20260717' \
  -H 'Content-Type: application/json' \
  --data-binary @"$EVIDENCE/vlm-publication-v3-request.json" \
  http://127.0.0.1:5057/api/model-publications

curl -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  --data '{"gpu_ids":[0]}' \
  http://127.0.0.1:5057/api/model-publications/600c1ed4aa7747fdac4a4f2f472bfbf0/validate
```

轮询读取任务实体状态，不读取 HTTP wrapper 的 `status`：

```bash
curl -fsS http://127.0.0.1:5057/api/history/4424f954 | jq -r '.data.status'
curl -fsS -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:5057/api/model-publications/600c1ed4aa7747fdac4a4f2f472bfbf0 \
  | jq -r '.task.status'
```

最终非 GPU 验收：

```bash
docker compose -p mergekit_publication_task7 exec -T mergekit-beta \
  /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
node --check mergeKit_beta/static/model_gateway/console.js
git diff --check
```

## 文本发布与调用

| 项目 | 实际值 |
| --- | --- |
| 发布任务 | `6fed54ce0a0f49e088cfc342452215c3` |
| publication ID | `aeebc61b35c64d5e8bb0765e156bccf7` |
| 首轮 core model ID | `cf118b6a-ec1c-496a-bb75-2e69b8f08ad4` |
| 服务 ID | `e553e5a8-0bc6-4b4c-a3b9-b3167051f123` |
| served model | `task7-qwen-text` |
| 正式目录 | `/home/a/Model_factory_data/published_models/aeebc61b35c64d5e8bb0765e156bccf7` |

真实 vLLM 调用得到有效回答。两条 usage 分别为 `48 + 20 = 68` 和
`32 + 27 = 59` tokens；请求 `94181343-cec4-427c-b98d-49ffac150518`
状态为 `success`，usage 来源为 `vllm_response`。服务已停止并软删除，临时 Key 已撤销。
该 schema 1 文本资产在 schema 2 上线后仍可读取，不会被当作损坏资产隔离。

## VLM 进化与配方

| 项目 | 实际值 |
| --- | --- |
| 进化任务 | `4424f954` |
| 数据集 | `m-a-p/CMMMU`, `health_and_medicine`, `val` |
| 父模型 | `Qwen2.5-VL-7B-TextOnly`、`HuatuoGPT-Vision-TextOnly` |
| 视觉基座 | `Qwen2.5-VL-7B-Instruct` |
| best genotype | `[0.390743353558069, 0.4229049526310821]` |
| 两次 CMMMU | `1.0`、`1.0` |
| API 任务耗时 | `189.31s` |
| 正式配方 | `mergeKit_beta/recipes/4424f954.json` |
| 配方 SHA-256 | `192e9414956dd03ca46d9efde46ac2f82b9f8ebe51e65b3da8e2a29ac68e3319` |

配方状态为 `success`。搜索开始和结束均校验父权重；配方记录 shard 和 index：

| 来源 | `weights_sha256` |
| --- | --- |
| Qwen2.5-VL-7B-TextOnly | `bbd1eba5e92ab7f12ffbea66410c6b1fe29d755124738cde415bfbea0a2c4b2b` |
| HuatuoGPT-Vision-TextOnly | `58c94a89620f96f3634c6a6b518f4d0e862efdf4b1c0bfc20016584b12ef31ae` |
| Qwen2.5-VL-7B-Instruct | `e6181ea3119f7bbe299d8b0ab439eb1b0885008948ded2544e18d56b920be760` |

正式发布后，搜索模型通过
`DELETE /api/model_repo/150c80b8-313c-4642-a158-a9871e4ad415` 清理；
`merges/4424f954/` 仅保留约 108 KiB 日志和指标。

## VLM 正式发布

| 项目 | 实际值 |
| --- | --- |
| 发布任务 | `600c1ed4aa7747fdac4a4f2f472bfbf0` |
| publication ID | `24c69e3ef8d549e3ae3e72e6e1a0a4bd` |
| core model ID | `5c13de8d-d482-47d8-bcf4-7a94561e7c18` |
| 正式目录 | `/home/a/Model_factory_data/published_models/24c69e3ef8d549e3ae3e72e6e1a0a4bd` |
| 资产字节数 | `16,595,843,236` |
| manifest | schema `2`, mode `0644`, state `published` |

验收结果：

- recipe SHA、完整 snapshot、两个有序父指纹和 VLM 基座指纹一致。
- shard 与 safetensors index 均进入来源聚合哈希。
- 图像推理和文本生成通过。
- CMMMU `val` 第 1 条完成推理，`samples=1`；该条回答错误，`acc=0.0`，
  只证明功能路径，不作为质量分数。
- 管理员候选 `selectable=false`，原因 `unsupported_architecture`。
- 服务创建返回 HTTP 409，未创建服务行。
- 临时用户 Key 的 `/v1/models` 返回 0 个模型，`/research` 不含该 VLM；Key 已撤销。

## 恢复与删除保护

受控崩溃探针使用任务 `ac64dbd47fae4abf81ffb06c5f72d705` 和 publication
`d415ec3a07ca4e02987b9587de17dd6f`。仅隔离 override 同时开启两个测试开关；
进程在 rename 后、注册前以 86 退出。普通配置重启后开关为 unset，Task 收敛为
`completed` 且 core model 行恰好 1 条，随后通过正式删除 API 清理探针。

删除保护使用 stopped 服务 `e51684aa-5b33-4b58-b822-741f70f5dfa0`；
引用文本资产时删除返回 HTTP 409 `asset_in_use`，软删除服务后引用归零。

## GPU 与资源

| GPU | UUID 前缀 | 验收前 MiB | 最终 CPU 门禁 MiB |
| --- | --- | ---: | ---: |
| 0 | `GPU-23348268` | 130 | 130 |
| 1 | `GPU-3f409b14` | 19 | 19 |
| 2（保护） | `GPU-7eff453d` | 19 | 19 |
| 3 | `GPU-ddf96bec` | 19 | 19 |

- VLM 搜索只使用 GPU 0；观测到的搜索阶段快照为 10,895 MiB。
- GPU 2 UUID、bus ID `00000000:81:00.0` 和 19 MiB 基线未变化。
- 最终 `nvidia-smi --query-compute-apps` 为空。
- 最终无 vLLM、Ray、geckodriver 或 headless browser 进程。
- 主服务 `model_factory-mergekit-beta-1` 为 healthy，`RestartCount=0`。
- 隔离 CPU 容器为 `runtime=runc`、`NVIDIA_VISIBLE_DEVICES=void`，故障开关均 unset。

## 自动化验收

- 完整容器测试：`358/358` 通过。
- 8 个 HTTP smoke 路由均返回 200。
- Python 关键导入通过。
- `node --check` 和 `git diff --check` 通过。
- 正式 `.staging`、`.trash`、`.quarantine` 均为空。

## 修复提交

| Commit | 作用 |
| --- | --- |
| `e40ea5b` | 隔离功能验证子进程输出。 |
| `215fa4a` | 成功重试后清除旧错误。 |
| `7ac2f6f` | VLM 临时语言模型使用 CPU，避免双模型 OOM。 |
| `d5e601e` | 有界搜索后物化 best genotype。 |
| `17a1022` | 原子 recipe mode 固定为 0644。 |
| `b64b5ad` | 发布验证复用本地 CMMMU，不依赖缺失的 `lmms_eval`。 |
| `f7c0e98` | 保存完整 recipe provenance。 |
| `8acfec0` | 受控 rename 后崩溃与恢复探针。 |
| `33d5951` | 最终模型缺失/保存失败不再误报成功。 |
| `62953c7` | 来源 shard 哈希和 UUID GPU 安全门禁。 |
| `ef58faa` | 进化前后父模型指纹契约。 |
| `7719273` | manifest schema 2、schema 1 兼容、index/现有模型指纹。 |
| `52e3153` | 中间 recipe metadata 禁止同步 Task DB。 |
| `adb5f43` | 中间 recipe metadata 使用独立文件，消除 backfill 覆盖。 |

## 保留与归档

- 正式文本资产：`published_models/aeebc61b35c64d5e8bb0765e156bccf7`。
- 正式 VLM schema 2 资产：`published_models/24c69e3ef8d549e3ae3e72e6e1a0a4bd`。
- 正式 VLM 配方：`mergeKit_beta/recipes/4424f954.json`。
- 旧 schema 1 VLM 验收资产未删除，归档为
  `/home/a/Model_factory_data/acceptance_archives/model_publication_20260717/legacy-374b73b7f4b049f0bb715315a38eaf15`。
- 旧 `d86dad3d` 配方原样归档于忽略的验收证据目录，不再作为活动 recipe。

## 回滚

1. 停止隔离项目：
   `docker compose -p mergekit_publication_task7 down`。
2. 逆序 `git revert` 本分支模型发布提交；不要直接删除正式资产目录。
3. 生产 `.env` 必须同时配置 `MERGEKIT_PROTECTED_GPU_UUIDS` 和
   `MERGEKIT_PUBLICATION_ALLOWED_GPU_UUIDS`；两者不得重叠。
4. 正式资产删除只走 `DELETE /api/model-publications/<publication_id>`；
   有服务引用时先停止并软删除服务。
5. legacy 归档只有在管理员确认新 schema 2 资产可替代后才能人工删除。
