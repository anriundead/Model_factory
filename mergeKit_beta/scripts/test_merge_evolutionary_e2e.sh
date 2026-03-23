#!/usr/bin/env bash
# 进化融合 e2e：提交进化融合任务 -> 轮询状态 -> 检查 output 或命名目录
# 使用前请先执行 ./restart_app.sh。需要 run_vlm_search.py 与数据集，否则会报错退出。
# 使用最小参数 pop_size=2 n_iter=1 以缩短时间。
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

echo "=== 2. 提交进化融合任务 (pop_size=2, n_iter=1) ==="
CUSTOM_NAME="test_evolutionary_$(date +%s)"
RESP=$(curl -s -X POST "$BASE/api/merge_evolutionary" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_paths\": [\"KaLM v2.5-0.5B\", \"Qwen2.5-0.5B\"],
    \"custom_name\": \"$CUSTOM_NAME\",
    \"hf_dataset\": \"cais/mmlu\",
    \"hf_subsets\": [\"college_chemistry\"],
    \"pop_size\": 2,
    \"n_iter\": 1,
    \"max_samples\": 8
  }")
TASK_ID=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('task_id','') or '')" 2>/dev/null || true)
if [ -z "$TASK_ID" ]; then
  echo "提交失败: $RESP"
  exit 1
fi
echo "task_id=$TASK_ID"

echo "=== 3. 轮询状态 (最多 15 分钟) ==="
MAX_POLL=450
for i in $(seq 1 $MAX_POLL); do
  STAT=$(curl -s "$BASE/api/status/$TASK_ID")
  STATUS=$(echo "$STAT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','') or '')" 2>/dev/null || true)
  PROG=$(echo "$STAT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))" 2>/dev/null || true)
  echo "  [$i] status=$STATUS progress=$PROG"
  case "$STATUS" in
    completed) echo "任务完成"; break ;;
    error)
      MSG=$(echo "$STAT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message','') or '')" 2>/dev/null || true)
      if echo "$MSG" | grep -q "eval_times"; then
        echo "失败: 仍出现 eval_times 未定义错误，修复未生效"
        exit 1
      fi
      echo "任务失败: $MSG"
      echo "$STAT" | python3 -m json.tool 2>/dev/null || echo "$STAT"
      exit 1
      ;;
    interrupted|stopped) echo "任务已中断"; exit 1 ;;
  esac
  if [ "$i" -eq "$MAX_POLL" ]; then
    echo "超时"
    exit 1
  fi
  sleep 2
done

echo "=== 4. 检查 output 或命名目录 ==="
MERGE_DIR="merges/$TASK_ID"
OUT_LINK="$MERGE_DIR/output"
if [ -d "$OUT_LINK" ]; then
  OUT_DIR="$OUT_LINK"
elif [ -L "$OUT_LINK" ]; then
  OUT_DIR="$OUT_LINK"
else
  OUT_DIR=$(find "$MERGE_DIR" -maxdepth 1 -type d -name '*_*_202*' 2>/dev/null | head -1)
fi
if [ -z "$OUT_DIR" ] || [ ! -e "$OUT_DIR" ]; then
  echo "失败: 未找到 output 或命名目录 under $MERGE_DIR"
  ls -la "$MERGE_DIR" 2>/dev/null || true
  exit 1
fi
REAL_DIR="$OUT_DIR"
if [ -L "$OUT_DIR" ]; then
  REAL_DIR=$(readlink -f "$OUT_DIR" 2>/dev/null || true)
  [ -z "$REAL_DIR" ] && REAL_DIR="$MERGE_DIR/$(readlink "$OUT_DIR")"
fi
if [ -f "$REAL_DIR/fusion_info.json" ] || ls "$REAL_DIR"/*.safetensors 1>/dev/null 2>&1; then
  echo "通过: $OUT_DIR 含 fusion_info 或 .safetensors"
  ls -la "$REAL_DIR" | head -8
else
  echo "失败: $REAL_DIR 下无 fusion_info.json 或 .safetensors"
  ls -la "$REAL_DIR" 2>/dev/null || true
  exit 1
fi
echo "=== 进化融合 e2e 测试通过 ==="
