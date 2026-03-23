#!/usr/bin/env python3
"""
升级 transformers 后全流程测试：标准融合（旧架构）+ 评测（旧架构）+ 评测（新架构 qwen3_5）。
依赖：服务已启动（PORT=5000），且 LOCAL_MODELS_PATH 下有可用模型。
"""
import json
import os
import sys
import time

try:
    import urllib.request as u
    import urllib.error as ue
except Exception:
    u = ue = None

PORT = int(os.environ.get("PORT", "5000"))
BASE = f"http://127.0.0.1:{PORT}"
TIMEOUT = 120  # 单次请求超时（秒）
POLL_INTERVAL = 2
MERGE_TIMEOUT = 600   # 10 分钟
EVAL_TIMEOUT = 600    # 10 分钟


def req(method, path, data=None):
    url = BASE + path
    if data is not None:
        data = json.dumps(data).encode("utf-8")
    r = u.urlopen(
        u.Request(url, data=data, method=method,
                  headers={"Content-Type": "application/json"} if data else {}),
        timeout=TIMEOUT
    )
    return json.loads(r.read().decode("utf-8"))


def get(path):
    return req("GET", path)


def post(path, data):
    return req("POST", path, data)


def wait_task(task_id, timeout_sec, kind="任务"):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        s = get(f"/api/status/{task_id}")
        status = s.get("status") or ""
        progress = s.get("progress", 0)
        print(f"  [{kind}] status={status} progress={progress}")
        if status == "completed":
            return True, s
        if status in ("error", "stopped", "interrupted"):
            return False, s
        time.sleep(POLL_INTERVAL)
    return False, {"message": "超时"}


def main():
    results = {"merge_old": None, "eval_old": None, "eval_new": None}

    # 1) 检查服务
    print("=== 1. 检查服务 ===")
    try:
        models_res = get("/api/models")
    except Exception as e:
        print("服务未响应:", e)
        print("请先执行: ./restart_app.sh")
        sys.exit(1)
    models = (models_res.get("models") or []) if models_res.get("status") == "success" else []
    if not models:
        print("未获取到基座模型，请检查 LOCAL_MODELS_PATH 与 /api/models")
        sys.exit(1)
    names = [m.get("name") or m.get("path", "").split("/")[-1] for m in models]
    print("可用基座模型:", names[:10])
    # 旧架构：名称中不含 qwen3（避免 Qwen3_5，mergekit 当前不支持）
    old_arch = [n for n in names if "qwen3" not in n.lower()]
    new_arch = [n for n in names if "qwen3" in n.lower()]

    # 2) 标准融合（旧架构：取两个非 qwen3 的模型）
    if len(old_arch) < 2:
        print("跳过融合测试：至少需要 2 个旧架构基座模型（当前旧架构: %s）" % old_arch)
    else:
        print("\n=== 2. 标准融合（旧架构，两模型） ===")
        a, b = old_arch[0], old_arch[1]
        payload = {
            "models": [a, b],
            "weights": [0.5, 0.5],
            "method": "linear",
            "dtype": "float16",
            "custom_name": f"e2e_merge_old_{int(time.time())}",
            "limit": "0.1",
            "type": "merge",
        }
        try:
            r = post("/api/merge", payload)
            tid = r.get("task_id") or ""
            if not tid:
                print("融合提交失败:", r)
                results["merge_old"] = "提交失败"
            else:
                ok, s = wait_task(tid, MERGE_TIMEOUT, "融合")
                results["merge_old"] = "通过" if ok else ("失败: " + (s.get("message") or str(s)))
                print("融合结果:", results["merge_old"])
        except Exception as e:
            results["merge_old"] = "异常: " + str(e)
            print("融合异常:", e)

    # 3) 测试集列表
    print("\n=== 3. 获取测试集 ===")
    try:
        ts = get("/api/testset/list?refresh=0")
        testsets = (ts.get("testsets") or []) if ts.get("status") == "success" else []
    except Exception as e:
        testsets = []
        print("获取测试集失败:", e)
    testset_id = None
    if testsets:
        t0 = testsets[0]
        testset_id = t0.get("testset_id") or t0.get("id") or t0.get("hf_dataset")
    if not testset_id and testsets:
        testset_id = testsets[0].get("hf_dataset")
    if not testset_id:
        print("无测试集，将使用内置 hellaswag 进行评测")

    # 4) 评测 - 旧架构模型（第一个旧架构，小规模快速验证）
    print("\n=== 4. 评测（旧架构模型） ===")
    if not old_arch:
        print("无旧架构模型，跳过")
        results["eval_old"] = "跳过(无旧架构模型)"
    else:
        name_old = old_arch[0]
        model_path_old = next((m.get("path") for m in models if (m.get("name") or "").strip() == name_old), None) or os.path.join("/home/a/ServiceEndFiles/Models", name_old)
        eval_payload = {
            "model_path": model_path_old,
            "dataset": "hellaswag",
            "depth": "full",
            "type": "eval_only",
            "limit": "0.05",
        }
        try:
            r = post("/api/evaluate", eval_payload)
            tid = r.get("task_id") or ""
            if not tid:
                print("评测提交失败:", r)
                results["eval_old"] = "提交失败"
            else:
                ok, s = wait_task(tid, EVAL_TIMEOUT, "评测旧架构")
                results["eval_old"] = "通过" if ok else ("失败: " + (s.get("message") or str(s))[:200])
                print("评测(旧架构)结果:", results["eval_old"])
        except Exception as e:
            results["eval_old"] = "异常: " + str(e)
            print("评测(旧架构)异常:", e)

    # 5) 评测 - 新架构（qwen3_5：选名称或路径含 qwen3 的模型）
    if "eval_old" not in results:
        results["eval_old"] = "未执行"
    qwen35_models = [m for m in models if "qwen3" in (m.get("name") or "").lower() or "qwen3" in (m.get("path") or "").lower()]
    if not qwen35_models:
        print("\n=== 5. 评测（新架构 qwen3_5）===")
        print("未发现名称/路径含 qwen3 的模型，跳过新架构评测")
        results["eval_new"] = "跳过(无 qwen3_5 模型)"
    else:
        print("\n=== 5. 评测（新架构 qwen3_5）===")
        m0 = qwen35_models[0]
        path_new = m0.get("path") or os.path.join("/home/a/ServiceEndFiles/Models", m0.get("name", ""))
        eval_payload_new = {
            "model_path": path_new,
            "dataset": "hellaswag",
            "depth": "full",
            "type": "eval_only",
            "limit": "0.05",
        }
        try:
            r = post("/api/evaluate", eval_payload_new)
            tid = r.get("task_id") or ""
            if not tid:
                print("评测(新架构)提交失败:", r)
                results["eval_new"] = "提交失败"
            else:
                ok, s = wait_task(tid, EVAL_TIMEOUT, "评测新架构")
                results["eval_new"] = "通过" if ok else ("失败: " + (s.get("message") or str(s))[:200])
                print("评测(新架构)结果:", results["eval_new"])
        except Exception as e:
            results["eval_new"] = "异常: " + str(e)
            print("评测(新架构)异常:", e)

    # 6) 汇总
    print("\n========== 全流程测试汇总 ==========")
    for k, v in results.items():
        print(f"  {k}: {v}")
    all_ok = all(v == "通过" or v and v.startswith("跳过") for v in results.values() if v)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
