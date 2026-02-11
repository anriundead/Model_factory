# Mergenetic Pro 开发文档

## 项目概述

Mergenetic Pro (Beta) 是一个基于 Web 的模型融合系统，支持标准融合和进化融合两种模式，使用 TIES-DARE 算法和 CMA-ES 优化器进行模型权重融合。目前正处于从 Beta 向正式版演进的阶段，重点在于增强自动化评估、数据管理和可视化分析能力。

### 核心功能 (已实现)

1. **标准融合**：基于用户指定的权重直接融合模型
2. **进化融合**：使用进化算法自动搜索最优融合权重
3. **配方管理**：保存和复用融合配方
4. **模型评估**：支持多种基准测试（MMLU、CMMMU 等）
5. **历史记录**：追踪所有融合任务和结果
6. **兼容性检查**：基于 `hidden_size` 和 `num_hidden_layers` 的自动校验

## 项目结构

```
mergeKit_beta/
├── app.py                      # Flask 后端主应用入口
├── config.py                   # 配置管理
├── merge_manager.py            # 融合与评估任务核心调度逻辑
├── app/                        # 应用逻辑模块
│   ├── routes.py              # API 路由定义
│   ├── services.py            # 业务服务层（模型路径、基准线查找等）
│   ├── models.py              # 数据库模型定义 (Task等)
│   ├── extensions.py          # Flask 扩展初始化 (db, migrate, admin)
│   └── admin.py               # Flask-Admin 后台视图配置
├── core/                       # 核心底层模块
│   ├── task_manager.py        # 任务队列管理
│   └── process_manager.py     # 进程管理
├── scripts/                    # 脚本目录
│   ├── run_vlm_search_bridge.py  # 进化融合桥接脚本
│   └── cleanup_orphaned_models.py # 清理遗留模型脚本
├── templates/                  # HTML 模板
│   ├── index.html             # 主页面
│   ├── fusion_history.html    # 融合历史页面
│   └── model_repo.html        # 模型仓库页面
├── static/                     # 静态资源
│   ├── app.js                 # 前端主逻辑
│   └── styles.css             # 样式文件
├── merges/                     # 融合结果目录
├── recipes/                    # 配方存储目录
└── logs/                       # 日志目录
```

## 核心概念与实现机制

### 1. 模型兼容性检查

两个或多个模型能否进行融合，取决于它们是否属于**同一架构**。判定依据：

- **`hidden_size`**：隐藏层维度，必须一致
- **`num_hidden_layers`**：隐藏层数，必须一致

**实现位置**：`app/services.py` (原 `app.py`) 中的 `ModelCompatibilityMixin` 类。

**注意**：不同来源的模型可能使用不同的 `model_type`（如 `llama`、`llama3`），但只要 `hidden_size` 和 `num_hidden_layers` 一致，即可融合。

### 2. 双 Split 评估机制

进化融合采用双 split 评估策略：

- **进化阶段**：使用 `--hf-split`（建议 `validation`）进行训练与验证，得到最优 genotype
- **最终评测**：使用 `--hf-split-final`（建议 `test`）对最优模型做一次准确率评估

**参数传递**：
- `--hf-split-final`：最终评测使用的 split（如 `test`）
- `--final-acc-file`：保存最终 test 准确率的 JSON 文件路径

**实现位置**：
- 桥接脚本：`scripts/run_vlm_search_bridge.py`
- 外部脚本：`run_vlm_search.py`（需支持上述参数）

### 3. 配方系统

配方（Recipe）保存了完整的融合配置，包括：
- `best_genotype`：最优权重向量
- `model_paths`：父模型路径
- `hf_dataset`、`hf_subsets`：数据集配置
- `current_best_acc`、`final_test_acc`：准确率指标

**配方应用**：可以通过配方直接融合出最终模型，跳过进化搜索过程。

**存储位置**：`recipes/<task_id>.json`

### 4. 基准线与排行榜 (Leaderboard)

系统动态查找基准线数据以生成雷达图：
1. **优先**：查找当前测试集 ID 对应的 `leaderboard.json` 数据。
2. **全局回退**：如果在当前测试集未找到，则扫描所有排行榜数据，匹配模型名称 + 数据集 + 子集。
3. **硬编码回退**：最后回退到 `Meta-Llama-3-8B-Instruct` 等常用基准。

**实现位置**：`app/services.py` 中的 `status_from_disk` 方法。

### 5. 数据库与管理后台

系统引入了 ORM 层以支持更强大的数据查询与管理能力：

- **技术栈**：Flask-SQLAlchemy (ORM) + Flask-Migrate (迁移) + Flask-Admin (后台)
- **数据库**：默认使用 SQLite (`app.db`)，可配置为 PostgreSQL。
- **管理后台**：访问 `/admin` 路径，提供可视化的数据增删改查功能。
- **数据模型**：`Task` (任务状态与配置)，后续将扩展 `EvaluationResult` 等。

## 未来架构与需求规划 (Future Architecture & Roadmap)

详细的开发需求与演进规划请参考独立文档：[development_requirements.md](../development_requirements.md)。

---

## API 接口参考

### 模型相关

- `GET /api/models` - 获取本地模型列表
- `GET /api/models_pool` - 获取模型池列表
- `GET /api/merged_models` - 获取已融合模型列表
- `GET /api/resolve_model_path` - 解析模型路径
- `POST /api/models/delete` - 删除模型（支持文件删除）

### 融合任务

- `POST /api/merge` - 提交标准融合任务
- `POST /api/merge_evolutionary` - 提交进化融合任务
- `POST /api/recipes/apply` - 根据配方直接融合
- `GET /api/status/<task_id>` - 获取任务状态和进度
- `POST /api/stop/<task_id>` - 停止任务
- `POST /api/resume/<task_id>` - 恢复任务

### 历史记录

- `GET /api/history` - 获取历史记录列表
- `GET /api/history/<task_id>` - 获取任务详情
- `DELETE /api/history/<task_id>` - 删除历史记录
- `GET /api/fusion_history` - 获取融合历史（含性能指标）

### 评估任务

- `POST /api/evaluate` - 提交评估任务（支持内置基准、测试集仓库、自定义数据集）
- `GET /api/status/<task_id>` - 获取任务状态和进度（含评估结果）

### 测试集管理

- `GET /api/testset/list` - 获取测试集仓库列表
- `POST /api/testset/create` - 创建新测试集（从 HuggingFace 下载）
- `POST /api/testset/search` - 搜索测试集
- `POST /api/dataset/hf_info` - 获取 HuggingFace 数据集信息（configs、splits）

### 配方管理

- `GET /api/recipes` - 获取所有配方列表
- `GET /api/recipes/<recipe_id>` - 获取配方详情

### 搜索

- `GET /api/search?q=<query>` - 搜索任务、配方、模型

## 配置说明

### 环境变量

- `LOCAL_MODELS_PATH`：本地模型存储路径（默认：`/home/a/ServiceEndFiles/Models`）
- `MERGEKIT_MODEL_POOL`：模型池路径
- `HF_DATASETS_CACHE`：HuggingFace 数据集缓存目录
- `MERGENETIC_PYTHON`：Python 解释器路径（用于运行外部脚本）

### 配置文件

主要配置在 `config.py` 中：

```python
class Config:
    PROJECT_ROOT = ...           # 项目根目录
    LOCAL_MODELS_PATH = ...      # 本地模型路径
    MERGE_DIR = ...              # 融合结果目录
    RECIPES_DIR = ...            # 配方目录
    HF_ENDPOINT = ...            # HuggingFace 镜像地址
    MERGENETIC_PYTHON = ...      # Python 解释器
```

## 任务执行流程

### 标准融合流程

1. 用户提交融合任务（模型路径、权重、方法等）
2. 任务加入队列，等待执行
3. `merge_manager.py` 中的 `run_merge_task()` 执行融合
4. 使用 `mergenetic` 库进行模型融合
5. 结果保存到 `merges/<task_id>/output/`
6. 更新 `metadata.json` 状态为 `success`

### 进化融合流程

1. 用户提交进化融合任务（模型路径、数据集、参数等）
2. 任务加入队列，等待执行
3. `merge_manager.py` 调用桥接脚本 `run_vlm_search_bridge.py`
4. 桥接脚本读取 `metadata.json`，调用外部 `run_vlm_search.py`
5. `run_vlm_search.py` 执行 CMA-ES 进化搜索：
   - 每次迭代生成多个 genotype（权重向量）
   - 对每个 genotype 进行融合和评估
   - 更新最优 genotype
   - 将进度写入 `progress.json`
6. 搜索完成后，保存最优模型到 `final_vlm`
7. 桥接脚本将模型复制到命名目录（`<model1>_<model2>_<timestamp>`）
8. 清理中间目录 `final_vlm`
9. 保存配方到 `recipes/<task_id>.json`
10. 更新 `metadata.json` 状态为 `success`

## 数据集验证

在启动 `run_vlm_search.py` 前，桥接脚本会验证数据集是否可以成功加载：

1. 尝试加载少量样本（最多 4 个）
2. 检查样本结构（必须包含 `question`、`choices`、`answer`）
3. 验证失败时记录错误并终止任务

**实现位置**：`scripts/run_vlm_search_bridge.py` 第 100-130 行

## 进度追踪

### 进度文件格式

`merges/<task_id>/progress.json`：

```json
{
  "step": 1,
  "current_best": 0.125,
  "global_best": 0.125,
  "best_genotype": [0.44, 0.45],
  "eta_seconds": 3600,
  "estimated_completion": 1234567890.0,
  "current_step": 5,
  "total_expected_steps": 20
}
```

### ETA 计算

- 从标准输出解析评估信息（step、acc、耗时）
- 基于平均评估耗时和剩余步数计算 ETA
- 每 10 步更新一次 `progress.json` 中的 ETA

## 模型清理策略

### 自动清理

1. **中间模型**：每次评估后自动清理（`run_vlm_search.py` 中）
2. **final_vlm 目录**：任务完成后自动清理（桥接脚本中）
3. **命名目录**：保留最终融合结果，不清理

### 手动清理

使用清理脚本清理遗留的中间模型：

```bash
python3 scripts/cleanup_orphaned_models.py
```

**清理策略**：
- 任务状态为 `success` 且有命名目录 → 清理
- 任务状态为 `error` → 清理
- 目录大于 1GB 且状态未知 → 清理（可能是遗留）

## 前端功能

### 主页面（index.html）

- 标准融合工作台
- 进化融合工作台
- 任务进度显示
- 配方直接融合

### 融合历史页面（fusion_history.html）

- 显示所有完全融合任务
- 性能指标（current_best、global_best）
- 完成步骤（评估次数/迭代次数）
- 搜索功能

### 模型仓库页面（model_repo.html）

- 显示基础模型和融合模型
- 显示融合配方
- 模型详情查看

## 开发指南

### 添加新的融合方法

1. 在 `merge_manager.py` 中实现融合逻辑
2. 在 `app.py` 中添加对应的 API 端点
3. 在前端添加相应的 UI

### 添加新的评估数据集

1. 确保数据集支持 HuggingFace `datasets` 库
2. 在桥接脚本中添加数据集加载逻辑
3. 更新前端数据集选择器

### 调试技巧

1. **查看日志**：
   - 应用日志：`logs/app.log`
   - 桥接日志：`merges/<task_id>/bridge.log`
   - 任务日志：`merges/<task_id>/task.log`

2. **检查任务状态**：
   - `merges/<task_id>/metadata.json`
   - `merges/<task_id>/progress.json`

3. **测试 API**：
   ```bash
   curl http://localhost:5000/api/status/<task_id>
   ```

## 评估功能详解

### 评估任务执行流程

1. **前端提交**：用户选择模型、数据集、测试深度（10%/50%/100%）、采样方式（顺序/随机）
2. **后端处理**：`app.py` 的 `start_evaluation_task()` 接收参数，生成任务 ID
3. **任务执行**：`merge_manager.py` 的 `run_eval_only_task()` 调用 `run_lm_eval_stream()`
4. **lm_eval 执行**：根据数据集自动发现或映射到对应的 lm_eval 任务名，执行评估
5. **结果解析**：从 lm_eval 输出中提取准确率、F1、样本数、上下文长度等指标
6. **结果展示**：前端显示数值和雷达图（Accuracy、Efficiency、Context 三维）

### 数据集支持

- **内置基准**：`all`、`hellaswag`、`arc_easy`、`boolq`、`winogrande`、`piqa`
- **测试集仓库**：从 HuggingFace 下载的数据集（如 `TIGER-Lab/MMLU-Pro`）
- **自定义数据集**：支持任意 HuggingFace 数据集 ID

### 自动任务映射

系统支持自动发现 HuggingFace 数据集到 lm_eval 任务的映射：

1. **优先级**：手动配置 > 内置映射 > 自动发现
2. **自动发现策略**：
   - 精确匹配数据集和子集名称
   - 关键词匹配（如 `health_and_medicine` → `professional_medicine`）
   - MMLU 特殊处理（支持 MMLU-Pro 等变体）
   - 智能评分选择最佳匹配任务
3. **映射保存**：自动发现的映射会保存到 `config/eval_task_mapping.json`

### 任务组过滤

系统会过滤掉 lm_eval 任务组（如 `mmlu`），只使用具体任务（如 `mmlu_professional_medicine`），避免 `TypeError: TaskConfig.__init__() got an unexpected keyword argument 'group'` 错误。

### 采样方式

- **顺序采样**：使用 `--limit` 参数，从数据集开头顺序取样本
- **随机采样**：由于 lm_eval CLI 不支持 `--samples`，当前使用 `--limit` 近似实现（未来可改进）

### 评估指标计算

- **Accuracy**：从 lm_eval 结果中提取
- **F1 Score**：从 lm_eval 结果中提取（如有）
- **Efficiency**：基于 `samples/time` 计算，归一化到 0-100（基准：100 samples/s = 100分）
- **Context**：从模型配置中提取上下文长度，归一化到 0-100

## 常见问题

### Q: 为什么融合步骤显示为 "1次评估 (1/5 迭代)"？

A: 这个显示有两个部分：
1. **"1次评估"**：表示实际完成了 1 次评估（`step=1`）
2. **"(1/5 迭代)"**：表示根据评估次数和种群大小计算的迭代进度

**问题原因**：
- `step` 表示**评估次数**，每次调用 `_evaluate()` 方法时 `step` 会加 1
- CMA-ES 算法每次迭代应该评估 `pop_size` 个个体（例如 `pop_size=4` 时，每次迭代应评估 4 次）
- 如果 `step=1`，说明只完成了 1 次评估，这可能是因为：
  1. **任务提前终止**：CMA-ES 可能在第一次评估后就找到了足够好的解，或者遇到了错误
  2. **并行评估问题**：如果使用 Ray 并行，可能某些评估任务失败或未完成
  3. **进度更新延迟**：`progress.json` 可能没有及时更新所有评估结果
  4. **CSV 保存问题**：`results_df` 可能没有正确保存到 CSV 文件（检查 `vlm_search_results/vlm_search/vlm_search.csv`）

**如何判断是否正常**：
- 如果任务状态为 `success` 且有最终模型输出，说明任务正常完成
- 如果 `step` 远小于 `pop_size × n_iter`，可能是任务提前结束或遇到问题
- 检查 `bridge.log` 和 `subprocess_output.log` 查看是否有错误信息
- 检查 `vlm_search_results/vlm_search/configs/` 目录中的配置文件数量，如果有很多配置文件但 CSV 为空，可能是 CSV 保存逻辑有问题

**显示逻辑说明**：
前端使用 `Math.ceil(currentStep / popSize)` 计算迭代次数，这是估算值。例如：
- `step=1, pop_size=4` → `Math.ceil(1/4) = 1` → 显示 "(1/5 迭代)"
- `step=4, pop_size=4` → `Math.ceil(4/4) = 1` → 显示 "(1/5 迭代)"
- `step=5, pop_size=4` → `Math.ceil(5/4) = 2` → 显示 "(2/5 迭代)"

**排查方法**：
1. 检查 `merges/<task_id>/subprocess_output.log` 文件，查看子进程的完整输出
2. 检查 `vlm_search_results/vlm_search/configs/` 目录中的配置文件数量
3. 检查 `vlm_search_results/vlm_search/vlm_search.csv` 文件是否有数据行
4. 如果配置文件很多但 CSV 为空，可能是 `results_df.to_csv()` 调用时机有问题，或者 `results_df` 在保存前被清空

### Q: 准确率为什么是 0？

A: 可能的原因：
1. 数据集加载失败
2. 评估过程未执行
3. `run_vlm_search.py` 未正确写入准确率

检查方法：
- 查看 `bridge.log` 中的数据集验证日志
- 检查 `final_test_acc.json` 是否存在
- 查看 `run_vlm_search.py` 的标准输出

### Q: 磁盘空间占用过大？

A: 运行清理脚本：
```bash
python3 scripts/cleanup_orphaned_models.py
```

定期清理可以释放大量空间（通常可释放 50-100GB）。

### Q: 如何重启应用？

A: 使用重启脚本：
```bash
./restart_app.sh
```

或手动：
```bash
# 查找并终止现有进程
ps aux | grep app.py | grep -v grep | awk '{print $2}' | xargs kill

# 启动应用
./start_app.sh
```

### Q: 评估任务失败，提示 "无法推断有效的 lm_eval 任务名"？

A: 可能原因：
1. 数据集名称或子集名称无法映射到 lm_eval 任务
2. 自动发现功能未找到匹配的任务

解决方法：
1. 检查 `config/eval_task_mapping.json`，手动添加映射
2. 查看日志中的自动发现过程，确认是否找到候选任务
3. 对于 MMLU-Pro 等新数据集，系统会自动尝试智能匹配

### Q: 评估结果显示 0 或 N/A？

A: 可能原因：
1. lm_eval 执行失败（检查 `eval_stderr.txt`）
2. 数据集加载失败
3. 任务组被误用（应使用具体任务名）

解决方法：
1. 查看任务目录下的 `eval_stderr.txt` 文件
2. 检查数据集是否正确下载到缓存目录
3. 确认使用的任务名不是任务组（如 `mmlu` 应改为 `mmlu_professional_medicine`）

### Q: 测试集下载后，下拉栏中看不到子集？

A: 系统会优先从本地已加载的数据集读取真实的 splits 和 configs。如果看不到：
1. 确认数据集已成功下载到缓存目录
2. 检查 `config/eval_task_mapping.json` 中是否有该数据集的映射
3. 前端会自动添加已存储的 `hf_subset` 到下拉选项

## 版本历史

### v1.1 (2026-02-04)

- ✅ 修复评估结果显示 0s/N/A 的问题
- ✅ 修复雷达图显示问题，支持单任务三维展示
- ✅ 修复 lm_eval CLI 参数错误（移除不支持的 --task_args）
- ✅ 实现 Efficiency 参数的有意义计算（基于 samples/time）
- ✅ 修复测试集下载后 hf_subset 和 sample_count 显示问题
- ✅ 实现随机采样和顺序采样功能
- ✅ 修复 TypeError: TaskConfig.__init__() got an unexpected keyword argument 'group' 错误
- ✅ 实现自动发现 lm_eval 任务映射功能（支持 MMLU-Pro 等新数据集）
- ✅ 优化从本地数据集读取真实的 splits 和 configs
- ✅ 增强任务组过滤和验证机制
- ✅ 实现资源分割与任务队列设计方案
- ✅ 定义铁人五项领域特化测试标准

### v1.0 (2026-02-03)

- ✅ 实现标准融合和进化融合
- ✅ 添加配方管理系统
- ✅ 实现双 split 评估机制
- ✅ 添加数据集验证功能
- ✅ 优化进度条显示（支持 ETA）
- ✅ 实现模型清理机制
- ✅ 添加融合历史页面
- ✅ 实现搜索功能

## 贡献指南

1. 遵循现有代码风格
2. 添加必要的注释和文档
3. 确保向后兼容
4. 更新本文档

## 许可证

[根据项目实际情况填写]
