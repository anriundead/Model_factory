# mergeKit_beta K8s 部署参考（与现有 compose 对齐）

本文档用于沉淀当前团队关于 Kubernetes 部署的共识，作为后续开发与运维实施基线。

## 1. 目标与边界

- 双轨运行：
  - 开发/调试继续使用 `docker-compose`
  - 生产/准生产使用 Kubernetes
- K8s 方案以现有 `Model_factory/docker-compose.yml` 行为为基线，不另起一套逻辑。
- 首期目标是“稳定可运行”，不是一次性完成多副本高可用架构升级。

## 2. 当前已确认决策

- 工作负载形态：`mergekit-beta` 首期单副本（`StatefulSet replicas=1` 或等价单副本 Deployment）。
- GPU 形态：单 Pod 多 GPU（对齐当前单机多卡任务执行方式）。
- 模型只读数据：与现网一致，使用节点本地路径（hostPath/本地 PV 等价方案）。
- 调度策略：通过 `nodeSelector/nodeAffinity` 固定到已准备模型目录的 GPU 节点。
- 数据库首期建议：SQLite + RWO（ReadWriteOnce）持久卷 + 单副本。
- 不纳入首期：多副本 Flask、Ray 多节点、任务系统大改（如拆独立 Job 队列）。

## 3. 给新同学的 K8s 架构解释

一套最小可落地的结构如下：

- Pod：运行 `mergekit-beta` 容器。
- Workload Controller：建议 `StatefulSet(1)`，确保有状态目录和单副本一致。
- Service：稳定暴露容器 `5000` 端口。
- Ingress（可选）：对外 HTTP/HTTPS 入口与域名/TLS。
- PVC（读写）：保存 `merges/`、`app.db`、缓存等需要持久化的数据。
- hostPath/本地 PV（只读）：挂载模型目录（例如 `/data/Models`）。
- nodeSelector/nodeAffinity：保证 Pod 只调度到有模型目录的 GPU 节点。
- GPU 资源限制：`resources.limits["nvidia.com/gpu"]` 声明需要的 GPU 数量。

## 4. compose 到 K8s 的关键映射

- 环境变量：
  - compose 中的 `LOCAL_MODELS_PATH`、`MERGEKIT_MODEL_POOL`、`HF_*`、`MERGEKIT_*`、`PYTORCH_CUDA_ALLOC_CONF` 等，拆分到 ConfigMap/Secret。
- 存储卷：
  - compose bind mount 的读写目录 -> PVC（RWO）。
  - compose bind mount 的模型只读目录 -> hostPath/本地 PV（只读）+ 节点绑定调度。
- 共享内存：
  - compose `shm_size: "10g"` -> Pod `emptyDir`（按需 `medium: Memory`）实现等价容量。
- 健康检查：
  - 保持 `/healthz`（可配 readiness/liveness）。

## 5. 为什么首期不直接多副本

- 当前默认数据库是 SQLite（单文件），多副本并发写有风险。
- 模型目录采用节点本地路径，天然带调度约束。
- 因此首期优先单副本稳定；若要横向扩容，需要同时升级到 PostgreSQL + 共享存储/对象存储方案。

## 6. 集群未定时的选型维度

- 是否托管控制面（降低运维复杂度）。
- GPU 支持方案（NVIDIA 驱动 + device plugin / GPU Operator）。
- CSI/StorageClass 能力（PVC 动态供给、性能、备份）。
- Ingress Controller 与 TLS 方案（cert-manager 或手工证书）。
- 镜像仓库连通性与拉取策略。

建议路径：先在测试命名空间验证单副本全链路（启动、健康检查、短 GPU 冒烟），再进入生产切换。

## 7. 与本次“进化融合秒失败”排障经验

2026-04-21 两次失败任务（`eeca55d9`、`b6a92ff0`）均在启动后约 15 秒内报错 `子进程退出码: 1`。

从容器内任务日志可确认根因是 CUDA OOM：

- 文件：`/app/ServiceEndFiles/Workspaces/mergeKit_beta/merges/b6a92ff0/subprocess_output.log`
- 关键错误：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.03 GiB`
- 同时日志显示当时 GPU0 仅约 `1.10 GiB` 空闲，且已有多个进程占用显存。

结论：这是资源竞争导致的显存不足，不是 K8s/接口参数格式错误。

## 8. 本项目后续实施建议（按优先级）

- 优先级 1：为进化任务增加“提交前显存门禁”和更直观报错（直接返回 OOM 明文）。
- 优先级 2：为 `merge_evolutionary` 增加自动降载策略（降低 batch/并发，必要时切换卡或排队）。
- 优先级 3：补充 `deploy/k8s/README.md`，提供 compose 与 K8s 配置对照表。
- 优先级 4：当业务要求多副本 Web 时，单独推进 PostgreSQL 迁移方案。

## 9. 参考文件

- `Model_factory/docker-compose.yml`
- `mergeKit_beta/Dockerfile`
- `mergeKit_beta/DEVELOPMENT.md`
- `mergeKit_beta/app/services.py`
