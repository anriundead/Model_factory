#!/usr/bin/env bash
# mergeKit_beta 启动脚本（含 conda 环境 mergenetic 激活，完全融合需此环境）
# 使用方式: ./start_app.sh
set -e
cd "$(dirname "$0")"

# 激活 conda 环境 mergenetic（运行 mergenetic 完全融合必需）
CONDA_BASE=""
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null) || true
fi
if [ -n "$CONDA_BASE" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate mergenetic
    export MERGENETIC_PYTHON="$(which python 2>/dev/null || true)"
    echo "--- 已激活 conda 环境: mergenetic ---"
fi

# 可选：指定 Python（未激活 conda 时使用）
if [ -n "$MERGENETIC_PYTHON" ]; then
    PYTHON="$MERGENETIC_PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

# 基座模型目录：未设置时使用默认路径，便于前端选取模型后正确解析路径
export LOCAL_MODELS_PATH="${LOCAL_MODELS_PATH:-/home/a/ServiceEndFiles/Models}"
# 默认端口 5000，可通过环境变量覆盖
export PORT="${PORT:-5000}"
echo "--- mergeKit_beta 后端启动 (port=$PORT) ---"
exec "$PYTHON" - <<'PY'
import os
import sys

# 确保项目根在 sys.path 最前（merge_manager 等从项目根加载）
try:
    root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)
os.chdir(root)

# 统一使用 app 包命名空间，避免同一 ORM 模型被重复加载（如 app.models 与 mergekit_app.models）
from app import app as flask_app
if flask_app is None:
    raise SystemExit("模块化应用未提供 app 实例")
port = int(os.environ.get("PORT", "5000"))
print("--- 使用模块化应用入口 ---")
print(f"--- mergeKit_beta 后端启动 (port={port}) ---")
flask_app.run(host="0.0.0.0", debug=True, port=port, use_reloader=False)
PY
