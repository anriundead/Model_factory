# 双 Split 与最终评测说明

## 设计

- **进化阶段**：使用 `--hf-split`（建议 `validation`）进行训练与验证，得到最优 genotype。
- **最终评测**：使用 `--hf-split-final`（建议 `test`）对最优模型做一次准确率评估，结果写入 `--final-acc-file`。

## run_vlm_search.py 需支持的参数

桥接脚本会视情况传入以下参数（当 `hf_split` ≠ `hf_split_final` 时）：

- **`--hf-split-final`**：最终评测使用的 split（如 `test`）。
- **`--final-acc-file`**：保存最终 test 准确率的 JSON 文件路径。脚本应在对最优模型完成 `hf_split_final` 上的评估后，将结果写入该文件，例如：  
  `{"final_test_acc": 0.xxxx}` 或 `{"accuracy": 0.xxxx}`。

桥接会读取该文件并将 `final_test_acc` 写入 fusion_info.json 与配方 JSON。

## 前端与 API

- 完全融合提交时可传 `hf_split`（训练/验证用）和 `hf_split_final`（最终准确率用，可选）。  
  若未传 `hf_split_final` 且 `hf_split` 为 `validation`，桥接默认使用 `test` 作为最终 split。
- 配方中会保存 `hf_split`、`hf_split_final`、`current_best_acc`、`final_test_acc`（若有）。
