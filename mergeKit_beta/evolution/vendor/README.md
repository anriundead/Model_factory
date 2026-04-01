# 内置 VLM 进化算法快照

本目录为 `Workspaces/modelmerge_visual/VLM_merge_total/VLM_merge` 的**源码快照**（不含其下大规模 `testset_repo/data` 缓存）。

## 升级 / 对齐上游

1. 在上游仓库查看变更，按需将对应 `.py` 与 `eval/` 下小文件复制到 `vlm_merge/`。
2. 保留本 README；可在提交说明中写上上游 commit 或日期。
3. 重新跑宿主机最小进化任务冒烟。

## 覆盖默认内置目录

设置环境变量 **`VLM_SEARCH_DIR`** 指向另一套树时，`evolution.runner` 将改用该目录中的 `run_vlm_search.py`（用于对比调试或临时回滚）。
