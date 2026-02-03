# Mergenetic Pro 开发文档

## 项目概述

Mergenetic Pro 是一个基于 Web 的模型融合系统，支持标准融合和进化融合两种模式，使用 TIES-DARE 算法和 CMA-ES 优化器进行模型权重融合。

### 核心功能

1. **标准融合**：基于用户指定的权重直接融合模型
2. **进化融合**：使用进化算法自动搜索最优融合权重
3. **配方管理**：保存和复用融合配方
4. **模型评估**：支持多种基准测试（MMLU、CMMMU 等）
5. **历史记录**：追踪所有融合任务和结果

## 项目结构

```
mergeKit_beta/
├── app.py                      # Flask 后端主应用
├── config.py                   # 配置管理
├── merge_manager.py            # 融合任务执行逻辑
├── core/                       # 核心模块
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
├── docs/                       # 文档目录
├── merges/                     # 融合结果目录
├── recipes/                    # 配方存储目录
└── logs/                       # 日志目录
```

## 核心概念

### 1. 模型兼容性检查

两个或多个模型能否进行融合，取决于它们是否属于**同一架构**。判定依据：

- **`hidden_size`**：隐藏层维度，必须一致
- **`num_hidden_layers`**：隐藏层数，必须一致

**实现位置**：`app.py` 中的 `_check_merge_compatible()` 函数

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

## API 接口

### 模型相关

- `GET /api/models` - 获取本地模型列表
- `GET /api/models_pool` - 获取模型池列表
- `GET /api/merged_models` - 获取已融合模型列表
- `GET /api/resolve_model_path` - 解析模型路径

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

## 常见问题

### Q: 为什么融合步骤显示为 1？

A: `step` 表示评估次数，不是迭代次数。如果任务只完成了 1 次评估就结束，`step` 就是 1。前端会显示为 "1 次评估 (1/4 迭代)" 的格式。

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

## 版本历史

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
