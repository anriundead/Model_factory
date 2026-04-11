#!/usr/bin/env bash
# mergeKit_beta 重启脚本：自动终止占用端口的进程并启动新进程
set -e
cd "$(dirname "$0")"

PORT="${PORT:-5000}"
echo "--- 查找并终止占用端口 ${PORT} 的进程 ---"
# 按端口终止（start_app.sh 用 python - 启动，命令行不含脚本名，故按端口杀）
PIDS=$(lsof -ti ":$PORT" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    echo "找到进程: $PIDS"
    for PID in $PIDS; do
        echo "终止进程 $PID..."
        kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null || true
    done
    sleep 2
    echo "--- 进程已终止 ---"
else
    echo "端口 $PORT 无占用进程"
    # 兼容：若曾用旧入口 python app.py 启动（已归档为 app.py.legacy），仍尝试按命令行匹配终止
    PIDS=$(ps aux | grep -E "python.*app\.py" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$PIDS" ]; then
        for PID in $PIDS; do echo "终止进程 $PID..."; kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null || true; done
        sleep 2
    fi
fi

echo ""
echo "--- 启动新进程 ---"
exec ./start_app.sh
