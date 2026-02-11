# Mergenetic Pro (Beta)

基于 Web 的模型融合系统，支持标准融合和进化融合两种模式。

## 快速开始

### 安装依赖

```bash
pip install flask mergenetic transformers datasets torch
```

### 配置环境

编辑 `config.py` 或设置环境变量：
- `LOCAL_MODELS_PATH`：本地模型存储路径
- `HF_DATASETS_CACHE`：HuggingFace 数据集缓存目录

### 启动应用

```bash
./start_app.sh
```

或

```bash
python3 app.py
```

访问 http://localhost:5000

## 主要功能 (已实现)

- ✅ **标准模型融合**（TIES-DARE）：支持基于权重的直接融合
- ✅ **进化融合**（CMA-ES）：自动化搜索最优融合权重
- ✅ **配方管理**：保存、复用和分享融合配置
- ✅ **模型评估**：
  - 集成 LM Evaluation Harness
  - 支持 YAML 配置和流式日志
  - 支持 MMLU、GSM8K 等主流 Benchmark
- ✅ **融合历史追踪**：记录所有任务状态与结果
- ✅ **模型兼容性检查**：自动校验架构一致性
- ✅ **前端可视化**：基础雷达图、进度监控与日志查看

## 开发路线图 (Roadmap)

我们正在进行下一阶段的重大升级，计划包含以下核心特性：

1.  **基础设施升级**
    - [ ] **数据库接入**：从 JSON 文件迁移至 SQLite/PostgreSQL，支持大规模数据管理。
    - [ ] **任务队列**：引入 Redis + Celery/RQ 管理高负载评估任务。

2.  **核心评估体系**
    - [ ] **铁人五项测试**：引入 Reasoning, Knowledge, Coding, Instruction, Safety 五维深度评估。
    - [ ] **特化能力识别**：自动分析模型强项，打上 `Math-Specialist` 等标签。

3.  **自动化流水线**
    - [ ] **优胜劣汰机制**：自动保留 Top 10% 表现的模型。
    - [ ] **自动清理**：自动删除低分模型权重（保留元数据），优化磁盘空间。

4.  **独立可视化系统**
    - [ ] **动态榜单**：支持多维度排序与基准线对比。
    - [ ] **高级图表**：趋势图、能力雷达、散点图（性价比分析）。
    - [ ] **深度透视**：展示模型血缘关系与完整进化路径。

详细需求文档请参阅 [DEVELOPMENT.md](DEVELOPMENT.md)

## 项目结构

- `app.py` - Flask 后端主应用
- `merge_manager.py` - 融合任务执行逻辑
- `app/` - 应用逻辑模块
  - `routes.py` - API 路由
  - `services.py` - 业务逻辑服务
- `scripts/` - 桥接脚本和工具脚本
- `templates/` - HTML 模板
- `static/` - 前端资源

## 文档

详细开发文档请参阅 [DEVELOPMENT.md](DEVELOPMENT.md)

## 清理遗留模型

定期运行清理脚本释放磁盘空间：

```bash
python3 scripts/cleanup_orphaned_models.py
```

## 许可证

[根据项目实际情况填写]
