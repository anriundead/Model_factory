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

# 默认端口 5001，可通过环境变量覆盖
export PORT="${PORT:-5001}"
echo "--- mergeKit_beta 后端启动 (port=$PORT) ---"
exec "$PYTHON" app.py
