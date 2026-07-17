# Model Gateway Operations

## 模型发布与服务创建

“发布资产”和“启动服务”是两个独立动作：

1. 融合或配方只产生候选模型。
2. `POST /api/model-publications` 将候选物化到 `/data/PublishedModels/.staging/`。
3. 管理员显式提交 GPU 验证后，资产经原子 rename 注册为正式 core model。
4. 只有 manifest 的 serving 状态为 `ready` 时，管理员才能创建 Gateway service。
5. 容器重启后模型服务保持 stopped，必须由管理员手动启动。

`ready` 表示当前 vLLM 版本已通过架构检查；`blocked` 表示明确不支持；`stale`
表示验证时和当前运行时版本不同，需要重新验证。不要用目录存在替代 manifest 状态。

## 文件与数据库权威

| 对象 | 权威来源 |
| --- | --- |
| 正式资产内容 | `/data/PublishedModels/<publication_id>/` |
| 资产契约与哈希 | `publication_manifest.json` |
| 发布状态机 | core `Task` 行 |
| 模型仓库记录 | core `Model` 行，`source=published` |
| 在线服务状态 | Gateway `serving_model_services` |

控制目录 `.staging`、`.trash`、`.quarantine` 必须与正式资产同一文件系统。
不要手动移动正式资产；删除使用
`DELETE /api/model-publications/<publication_id>`，有 service 引用时先停止并软删除 service。

## Manifest 版本

- schema 1：只用于兼容升级前已合法发布的资产，不会自动升级或隔离。
- schema 2：所有新发布资产。recipe 发布必须记录 recipe SHA、完整 snapshot、
  有序父模型、每个父模型 shard/index 哈希和 VLM 基座哈希。
- existing-model 发布在 schema 2 中必须记录 core model ID 和复制前后相同的来源指纹。

来源 fingerprint 覆盖 safetensors shard 和 index 文件。配方在进化开始/结束以及
发布开始/结束都会比较来源；同一路径原地替换权重会以
`source_fingerprint_mismatch` 失败。

Publication 内部 recipe 物化日志写入 `_publication_recipe_metadata.json`。
禁止改回标准 `metadata.json`，否则历史 backfill 可能覆盖正式 Task.config。

## GPU 安全配置

宿主机忽略的 `.env` 必须配置：

```text
MERGEKIT_PROTECTED_GPU_UUIDS=<不可用于发布验证的 GPU UUID，逗号分隔>
MERGEKIT_PUBLICATION_ALLOWED_GPU_UUIDS=<获准用于发布验证的 GPU UUID，逗号分隔>
```

安全规则：

- 两项必须非空、格式合法且不得重叠。
- 前端/API 仍提交当前容器 index，但 preflight 必须解析出 UUID 和 PCI bus ID。
- 所选 UUID 必须在 allowed 集合且不在 protected 集合。
- 验证子进程的 `CUDA_VISIBLE_DEVICES` 使用 UUID，不使用可变 index。
- 保护卡可以不暴露给容器；此时它不可能被选中。允许列表过期时，所选卡会 fail-closed。

修改 GPU 配置前后记录宿主机：

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.used,memory.total \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory \
  --format=csv,noheader,nounits
```

## 恢复

- `validating`：staging 和 Task 持久化；重启后管理员可再次提交验证。
- `registration_pending`：正式目录已 rename；startup reconcile 验证 manifest 后，
  幂等注册唯一 core Model 行并把 Task 收敛为 `completed`。
- `completed`：不要重新物化；以 manifest 和 core Model 行核对。
- `failed`/`canceled`：不得自动提交 staging；确认没有引用后清理失败目录。

受控崩溃开关仅限隔离验收：

```text
MERGEKIT_ENABLE_TEST_FAULTS=1
MERGEKIT_PUBLICATION_TEST_CRASH_AFTER_RENAME=1
```

普通 Compose 和生产 `.env` 禁止设置。恢复后先确认两个变量均为 unset，再继续服务。

## 验收

```bash
docker compose config --quiet
docker compose ps
curl -fsS http://127.0.0.1:5000/healthz
curl -fsS http://127.0.0.1:5000/readyz
docker compose exec -T mergekit-beta \
  /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
node --check mergeKit_beta/static/model_gateway/console.js
git diff --check
```

真实模型验收完成后必须停止测试服务、撤销临时 API Key、确认无 vLLM/Ray/browser
进程、显存回落，并保留脱敏请求、Task/publication/model ID 和 manifest SHA。
