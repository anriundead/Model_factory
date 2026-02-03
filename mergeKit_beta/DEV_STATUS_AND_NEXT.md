# mergeKit_beta 开发状态与后续顺序

本文档对照当前实现整理已完成项与建议的后续开发顺序。若你有《模型工厂开发文档》的副本，可将其放在项目内以便逐条对照。

---

## 已完成

### 1. 基座模型路径统一
- **配置**：`config.py` 中 `LOCAL_MODELS_PATH` 为基座模型正式存放路径（默认 `/home/a/ServiceEndFiles/Models`），可通过环境变量 `LOCAL_MODELS_PATH` 覆盖。
- **API**：`GET /api/models` 与 `GET /api/models_pool` 从 `LOCAL_MODELS_PATH` 扫描目录，返回带 `path` 的模型列表；响应中增加 `base_path` 便于前端展示。
- **标准融合**：支持 `model_paths`（绝对路径）或 `models`（名称）；`merge_manager.run_merge_task` 优先使用 `params.model_paths`，否则用 `MODEL_POOL_PATH + models`。
- **前端**：模型选择保存 `{ name, path }`，提交时优先传 `model_paths`，保证基座模型来自真实路径。

### 2. 完全融合界面：排版与下拉
- 完全融合面板与标准融合统一排版风格，参数使用与标准融合一致的 **下拉选择器**（种群大小、迭代次数、最大样本量、精度、Ray GPU 数）。
- **数据集类型**：增加「MMLU（纯文本）」/「CMMMU（多模态）」下拉；**子集**随数据集类型动态加载（MMLU 子集 / CMMMU 子集）。
- 后端：`GET /api/mmlu_subsets`、`GET /api/cmmmu_subsets`；提交 `merge_evolutionary` 时传 `hf_dataset`（`cais/mmlu` 或 `m-a-p/CMMMU`）、`hf_subset`、`hf_split`（MMLU 用 `test`，CMMMU 用 `val`）。

### 3. VLM 检测与自动切换数据集
- **API**：`GET /api/model_is_vlm?path=<模型路径>`，根据该路径下 `config.json` 的 `model_type` / `architectures` 判断是否为 VLM（含 vision/vl/qwen2_vl 等关键词）。
- **前端**：完全融合面板中，用户选择/取消模型后自动检测当前已选模型是否包含 VLM；若有，则将「数据集类型」自动设为 CMMMU 并加载 CMMMU 子集，用户再选子集即可。

---

## 建议的后续开发顺序

1. **开发文档对照**  
   将《模型工厂开发文档》内容放入本仓库或粘贴给助手，逐条对照文档中的功能列表，标出已实现/未实现，并据此调整本清单与优先级。

2. **模型仓库路径与展示**  
   若「模型仓库」页需要展示真实存储路径或从多路径聚合（例如基座路径 + 融合输出目录），可在 `model_repo` 的列表/解析逻辑中接入 `LOCAL_MODELS_PATH` 与 `MERGE_DIR`，并确保 `model_repo/data/models.json` 与真实路径一致或由后端动态补充 path。

3. **桥接脚本与 CMMMU**  
   当前桥接脚本固定使用 `eval/prompt_mmlu.yaml`。若 CMMMU 需不同 prompt 配置，可在 `run_vlm_search_bridge.py` 中根据 `metadata.json` 的 `hf_dataset` 选择对应 prompt 文件并传给 `run_vlm_search.py`。

4. **测试集仓库 (testset_repo)**  
   `testset_repo` 的创建/管理目前为占位。若文档中有测试集创建、下载或与评估流水线联动的需求，按文档优先级实现。

5. **评估任务 (run_eval_only_task)**  
   `merge_manager.run_eval_only_task` 仍为占位，若需单独跑评估流水线，在此实现并接好前端「模型测试」与历史记录。

---

## 文件清理约定

任何**删除或批量清理**文件的操作，需得到你的**二次确认**后再执行。详见 `.cursor/rules/DO_NOT_DELETE_FILES.md`。

---

## 环境与路径速查

| 用途           | 配置/环境变量           | 默认值示例 |
|----------------|-------------------------|------------|
| 基座模型目录   | `LOCAL_MODELS_PATH`     | `/home/a/ServiceEndFiles/Models` |
| 融合池目录     | `MERGEKIT_MODEL_POOL`   | `…/mergeKit/models_pool` |
| 融合输出目录   | `Config.MERGE_DIR`      | `mergeKit_beta/merges` |
| VLM 搜索脚本   | `VLM_SEARCH_DIR`        | `…/modelmerge_visual/VLM_merge_total/VLM_merge` |
