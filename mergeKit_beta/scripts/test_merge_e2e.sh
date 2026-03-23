#!/usr/bin/env bash
# 全流程融合测试：提交任务 -> 轮询状态 -> 检查 output 目录
# 使用前请先执行 ./restart_app.sh 确保使用最新代码
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-5000}"
BASE="http://127.0.0.1:${PORT}"

echo "=== 1. 检查服务 ==="
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -s -o /dev/null -w "%{http_code}" "$BASE/api/models" | grep -q 200; then
    echo "服务已就绪"
    break
  fi
  if [ "$i" -eq 10 ]; then
    echo "服务未响应，请先运行 ./restart_app.sh"
    exit 1
  fi
  sleep 1
done

echo "=== 2. 提交融合任务 (linear, 两个模型) ==="
CUSTOM_NAME="test_linear_merge_$(date +%s)"
RESP=$(curl -s -X POST "$BASE/api/merge" \
  -H "Content-Type: application/json" \
  -d "{
    \"models\": [\"KaLM v2.5-0.5B\", \"Qwen2.5-0.5B\"],
    \"weights\": [0.5, 0.5],
    \"method\": \"linear\",
    \"dtype\": \"float16\",
    \"custom_name\": \"$CUSTOM_NAME\",
    \"limit\": \"0.1\",
    \"type\": \"merge\"
  }")
TASK_ID=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('task_id','') or '')" 2>/dev/null || true)
if [ -z "$TASK_ID" ]; then
  echo "提交失败: $RESP"
  exit 1
fi
echo "task_id=$TASK_ID"

echo "=== 3. 轮询状态 (最多 5 分钟) ==="
for i in $(seq 1 150); do
  STAT=$(curl -s "$BASE/api/status/$TASK_ID")
  STATUS=$(echo "$STAT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','') or '')" 2>/dev/null || true)
  PROG=$(echo "$STAT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))" 2>/dev/null || true)
  echo "  [$i] status=$STATUS progress=$PROG"
  case "$STATUS" in
    completed) echo "任务完成"; break ;;
    error) echo "任务失败"; echo "$STAT" | python3 -m json.tool 2>/dev/null || echo "$STAT"; exit 1 ;;
    interrupted|stopped) echo "任务已中断"; exit 1 ;;
  esac
  if [ "$i" -eq 150 ]; then
    echo "超时"
    exit 1
  fi
  sleep 2
done

echo "=== 4. 检查 output 目录 ==="
OUT_DIR="merges/$TASK_ID/output"
if [ ! -d "$OUT_DIR" ]; then
  echo "失败: $OUT_DIR 不存在"
  exit 1
fi
if ! ls "$OUT_DIR"/*.safetensors 1>/dev/null 2>&1; then
  echo "失败: $OUT_DIR 下无 .safetensors 文件"
  ls -la "$OUT_DIR" 2>/dev/null || true
  exit 1
fi
echo "通过: $OUT_DIR 含 .safetensors"
ls -la "$OUT_DIR" | head -5
echo "=== 全流程测试通过 ==="
