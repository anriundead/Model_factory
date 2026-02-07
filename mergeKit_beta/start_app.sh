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
# 默认端口 5001，可通过环境变量覆盖
export PORT="${PORT:-5001}"
echo "--- mergeKit_beta 后端启动 (port=$PORT) ---"
exec "$PYTHON" - <<'PY'
import importlib.util
import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(root, "app")
app_init = os.path.join(app_dir, "__init__.py")
spec = importlib.util.spec_from_file_location("mergekit_app", app_init, submodule_search_locations=[app_dir])
if spec is None or spec.loader is None:
    raise SystemExit("无法加载模块化应用入口")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
app = getattr(mod, "app", None)
if app is None:
    raise SystemExit("模块化应用未提供 app 实例")
port = int(os.environ.get("PORT", "5001"))
print("--- 使用模块化应用入口 ---")
print(f"--- mergeKit_beta 后端启动 (port={port}) ---")
app.run(host="0.0.0.0", debug=True, port=port, use_reloader=False)
PY
