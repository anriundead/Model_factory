# 配置目录

## 评估任务映射（可选）

若需在**部署后**为测试集仓库增加 lm_eval 任务映射，可在此目录下创建 `eval_task_mapping.json`：

- 格式：`{"hf_dataset": "lm_eval 任务前缀", "hf_dataset|subset": "完整任务名"}`
- 示例见 `eval_task_mapping.example.json`，复制为 `eval_task_mapping.json` 后按需修改。
- 与内置映射（如 MMLU）合并使用；测试集创建或前端选择时也会自动推断并写入 `lm_eval_task`，无需手写映射即可使用 MMLU 等已支持数据集。
