# Mergenetic Pro

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

## 主要功能

- ✅ 标准模型融合（TIES-DARE）
- ✅ 进化融合（CMA-ES 优化）
- ✅ 配方管理和复用
- ✅ 模型评估（MMLU、CMMMU）
- ✅ 融合历史追踪
- ✅ 模型兼容性检查

## 项目结构

- `app.py` - Flask 后端主应用
- `merge_manager.py` - 融合任务执行逻辑
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
