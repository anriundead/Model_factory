#!/usr/bin/env bash
# mergeKit_beta 重启脚本：自动终止旧进程并启动新进程
set -e
cd "$(dirname "$0")"

echo "--- 查找并终止现有进程 ---"
# 查找运行 app.py 的进程（排除 grep 自身）
PIDS=$(ps aux | grep -E "python.*app\.py" | grep -v grep | awk '{print $2}' || true)
if [ -n "$PIDS" ]; then
    echo "找到进程: $PIDS"
    for PID in $PIDS; do
        echo "终止进程 $PID..."
        kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null || true
    done
    sleep 2
    echo "--- 进程已终止 ---"
else
    echo "未找到运行中的进程"
fi

echo ""
echo "--- 启动新进程 ---"
exec ./start_app.sh
