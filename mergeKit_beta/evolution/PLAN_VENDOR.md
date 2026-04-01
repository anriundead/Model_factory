# 进化融合内置化开发计划（VLM 算法进仓）

## 目标

- **默认**不再依赖宿主机/容器外「单独一棵树」里的 `run_vlm_search.py`；算法与配套脚本位于 **`mergeKit_beta/evolution/vendor/vlm_merge/`**。
- Worker 仍只拉起 **一个** 子进程：`MERGENETIC_PYTHON` + **`evolution.runner`**；Runner 再 `Popen` 仓内 `vendor/vlm_merge/run_vlm_search.py`（行为与现在一致，仅路径改为内置）。
- **Docker 镜像/compose**：本轮不改动；宿主机验证通过后再做打包阶段。
- **回滚**：设置 **`VLM_SEARCH_DIR`** 指向旧外部树时，Runner **优先使用该路径**（与当前逻辑一致：非空环境变量覆盖默认内置目录）。

## 非目标（本轮）

- 不复制外部仓库中 **`testset_repo/` 下大批量缓存数据**（体积大；数据加载继续走 HF + `HF_DATASETS_CACHE` / 项目既有缓存）。
- 不改为「完全 in-process 不调子进程」（可选后续优化）。

## 阶段与任务

| 阶段 | 内容 | 验收 |
|------|------|------|
| A | 从 `Workspaces/modelmerge_visual/VLM_merge_total/VLM_merge` **快照复制** Python 与 `eval/` 内 yaml/jsonl（不含上级 `testset_repo` 数据目录）到 `evolution/vendor/vlm_merge/` | 目录内存在 `run_vlm_search.py`、`eval_final.py`、`eval/prompt_mmlu.yaml` |
| B | **小补丁**：`eval_final.py` 的 `cache_dir` 使用 `HF_DATASETS_CACHE`（由 Runner 已传入子进程）；去除 `run_vlm_search.py` 中仅调试用 `print(PROJECT_ROOT)` | 无双副本硬编码依赖 `./testset_repo/data` |
| C | **`evolution/runner.py`**：`VLM_SEARCH_DIR` 为空时默认 **`Path(__file__).parent / "vendor" / "vlm_merge"`**；非空则仍用外部树 | 不设 `VLM_SEARCH_DIR` 时可找到 `run_vlm_search.py` |
| D | 更新 **`evolution/contracts.md`**、`vendor/README.md`（记录上游路径与同步说明） | 契约与来源一致 |
| E | 宿主机冒烟：`python -m evolution.runner --task-id …` 或提交最小进化任务 | 与改造前同等级成功/失败（GPU/依赖仍属环境要求） |

## 依赖（仍由环境保证）

- `mergenetic`、`ray`、`pymoo`、`torch`、`transformers`、`datasets` 等与当前外部树一致；**不因进仓而减少依赖**。

## 待你确认（若与下述默认不一致请说明）

1. **同步策略**：默认采用 **快照复制 + `vendor/README.md` 记来源**；是否要改为 **git submodule** 跟踪 `modelmerge_visual`？
2. **默认覆盖**：默认 **内置目录优先**；仅当 **`VLM_SEARCH_DIR` 非空** 时改用外部树。是否坚持 **仅允许内置、禁止外置**（将移除覆盖能力）？

---

**执行状态（2026-03-30）**：阶段 A–D 已按默认在仓库内落地（快照目录 `vendor/vlm_merge`、Runner 默认路径、`eval_final` 缓存取自 `HF_DATASETS_CACHE`）。阶段 E 需在具备 GPU 与 mergenetic 依赖的环境执行冒烟。
