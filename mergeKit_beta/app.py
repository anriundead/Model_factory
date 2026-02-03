"""
mergeKit_beta Flask 应用：融合任务队列、API 与前端路由。
/api/status/<task_id> 会从 merges/<task_id>/metadata.json 回退读取状态。
"""
import json
import logging
import os
import threading
import time
import uuid
import queue
import shutil
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_from_directory

# 配置与环境
from config import Config

Config.setup_environment()

from merge_manager import run_merge_task, run_eval_only_task, MODEL_POOL_PATH, MERGE_DIR
from core.process_manager import ProcessManager

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- 应用日志：记录任务提交、执行与结果，便于定位问题 ----------
def _app_log_dir():
    """应用日志目录，与 merge_manager 的 LOGS_DIR 区分，便于查看任务流。"""
    d = os.path.join(Config.PROJECT_ROOT, "logs")
    os.makedirs(d, exist_ok=True)
    return d


def _setup_app_logger():
    log_dir = _app_log_dir()
    log_file = os.path.join(log_dir, "app.log")
    logger = logging.getLogger("mergeKit_beta")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.info("应用日志文件: %s", os.path.abspath(log_file))
    return logger


_app_logger = _setup_app_logger()

PRIORITY_MAP = Config.PRIORITY_MAP
tasks = {}
task_queue = queue.PriorityQueue()
running_task_info = {"id": None, "priority": None, "process": None}
scheduler_lock = threading.Lock()


def _kill_process_tree_by_pid(pid):
    ProcessManager.kill_process_tree(pid)


def interrupt_current_task(reason="被高优先级任务打断"):
    global running_task_info
    current_id = running_task_info["id"]
    if not current_id or current_id not in tasks:
        return
    proc = running_task_info.get("process")
    if proc:
        print("!!! 正在打断任务 %s，原因: %s !!!" % (current_id, reason))
        _kill_process_tree_by_pid(proc.pid)
    if "control" in tasks[current_id]:
        tasks[current_id]["control"]["aborted"] = True
    current_p_score = running_task_info["priority"]
    original_priority_str = tasks[current_id].get("priority", "common")
    if current_p_score == PRIORITY_MAP["cutin"] or original_priority_str == "cutin":
        tasks[current_id]["status"] = "interrupted"
        tasks[current_id]["message"] = "任务被打断，等待恢复"
    else:
        tasks[current_id]["status"] = "queued"
        tasks[current_id]["message"] = "正在排队 (自动恢复中)..."
        tasks[current_id]["control"] = {"aborted": False, "process": None}
        tasks[current_id]["restarted"] = True
        task_data = tasks[current_id].get("original_data")
        created_at = tasks[current_id].get("created_at")
        task_queue.put((current_p_score, created_at, current_id, task_data))
    running_task_info["id"] = None
    running_task_info["priority"] = None
    running_task_info["process"] = None


def is_name_duplicate(new_name):
    if not os.path.exists(MERGE_DIR):
        return False
    for task_id in os.listdir(MERGE_DIR):
        path = os.path.join(MERGE_DIR, task_id)
        if not os.path.isdir(path):
            continue
        meta_path = os.path.join(path, "metadata.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                if json.load(f).get("custom_name") == new_name:
                    return True
        except Exception:
            continue
    return False


def get_all_history():
    history_list = []
    if not os.path.exists(MERGE_DIR):
        return []
    for task_id in os.listdir(MERGE_DIR):
        task_path = os.path.join(MERGE_DIR, task_id)
        if not os.path.isdir(task_path):
            continue
        meta_path = os.path.join(task_path, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("type") == "eval_only":
                continue
            if "id" not in meta:
                meta["id"] = task_id
            history_list.append(meta)
        except Exception as e:
            print("Error reading metadata for %s: %s" % (task_id, e))
    return sorted(history_list, key=lambda x: x.get("created_at", 0), reverse=True)


def _status_from_disk(task_id):
    """从 merges/<task_id>/metadata.json 读取状态，用于 /api/status 回退。"""
    meta_path = os.path.join(MERGE_DIR, task_id, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "id": task_id,
            "status": meta.get("status", "unknown"),
            "progress": 100 if meta.get("status") == "success" else (0 if meta.get("status") == "error" else 50),
            "message": meta.get("error") or meta.get("message", ""),
            "created_at": meta.get("created_at"),
            "result": {"status": meta.get("status"), "metrics": meta.get("metrics")},
        }
    except Exception:
        return None


def worker():
    global running_task_info
    print("--- 任务处理 Worker 已启动 ---")
    while True:
        try:
            priority_score, created_at, task_id, data = task_queue.get()
        except Exception:
            continue
        if tasks.get(task_id, {}).get("status") == "stopped":
            task_queue.task_done()
            continue
        try:
            with scheduler_lock:
                tasks[task_id]["status"] = "running"
                tasks[task_id]["message"] = "正在初始化..."
                task_control = {"aborted": False, "process": None}
                tasks[task_id]["control"] = task_control
                running_task_info["id"] = task_id
                running_task_info["priority"] = priority_score

            def update_progress(p, msg):
                if tasks.get(task_id, {}).get("status") in ["interrupted", "stopped"]:
                    return
                tasks[task_id]["progress"] = p
                tasks[task_id]["message"] = msg
                if task_control.get("process") and running_task_info.get("id") == task_id:
                    running_task_info["process"] = task_control["process"]

            task_type = data.get("type", "merge")
            _app_logger.info(
                "[worker] 任务开始 task_id=%s type=%s custom_name=%s",
                task_id,
                task_type,
                data.get("custom_name", ""),
            )
            if task_type == "merge":
                _app_logger.info(
                    "[worker] 标准融合 模型数=%s method=%s",
                    len(data.get("model_paths") or data.get("models") or []),
                    data.get("method", ""),
                )
            elif task_type == "merge_evolutionary":
                _app_logger.info(
                    "[worker] 完全融合 模型路径=%s hf_subsets=%s pop_size=%s n_iter=%s",
                    data.get("model_paths", [])[:3],
                    data.get("hf_subsets", []),
                    data.get("pop_size"),
                    data.get("n_iter"),
                )
            elif task_type == "eval_only":
                _app_logger.info(
                    "[worker] 评估任务 model_path=%s dataset=%s",
                    data.get("model_path", ""),
                    data.get("dataset", ""),
                )

            if task_type == "merge_evolutionary":
                # VLM 进化搜索：启动 run_vlm_search_bridge 子进程（若存在）
                script_path = os.path.join(Config.PROJECT_ROOT, "scripts", "run_vlm_search_bridge.py")
                if not os.path.isfile(script_path):
                    tasks[task_id]["status"] = "error"
                    tasks[task_id]["message"] = "scripts/run_vlm_search_bridge.py 不存在"
                    result = {"status": "error", "error": "run_vlm_search_bridge.py 未找到"}
                else:
                    import subprocess
                    merge_dir = os.path.join(MERGE_DIR, task_id)
                    os.makedirs(merge_dir, exist_ok=True)
                    progress_path = os.path.join(merge_dir, "progress.json")
                    meta_path = os.path.join(merge_dir, "metadata.json")
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump({"id": task_id, "type": "merge_evolutionary", "status": "running", **data}, f, ensure_ascii=False, indent=2)
                    proc = subprocess.Popen(
                        [Config.MERGENETIC_PYTHON, script_path, "--task-id", task_id],
                        cwd=Config.PROJECT_ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        **_popen_group_kwargs(),
                    )
                    task_control["process"] = proc
                    try:
                        while True:
                            if task_control.get("aborted"):
                                ProcessManager.kill_process_tree(proc)
                                raise Exception("任务已停止")
                            line = proc.stdout.readline() if proc.stdout else ""
                            if not line and proc.poll() is not None:
                                break
                            if line:
                                update_progress(50, line.strip()[:80])
                        if proc.returncode != 0:
                            raise RuntimeError("子进程退出码: %s" % proc.returncode)
                        result = {"status": "success"}
                        _app_logger.info("[worker] 完全融合完成 task_id=%s", task_id)
                    except Exception as e:
                        _app_logger.exception("[worker] 完全融合失败 task_id=%s error=%s", task_id, e)
                        with open(progress_path, "w", encoding="utf-8") as f:
                            json.dump({"status": "error", "message": str(e)}, f, ensure_ascii=False)
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                m = json.load(f)
                            m["status"] = "error"
                            m["error"] = str(e)
                            with open(meta_path, "w", encoding="utf-8") as f:
                                json.dump(m, f, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
                        result = {"status": "error", "error": str(e)}
            elif task_type == "eval_only":
                result = run_eval_only_task(task_id, data, update_progress, task_control)
            else:
                result = run_merge_task(task_id, data, update_progress, task_control)
                if result.get("status") == "success":
                    _app_logger.info("[worker] 标准融合完成 task_id=%s output=%s", task_id, result.get("output_dir", ""))
                else:
                    _app_logger.warning("[worker] 标准融合失败 task_id=%s error=%s", task_id, result.get("error", ""))

            if task_type == "eval_only" and result:
                if result.get("status") == "success":
                    _app_logger.info("[worker] 评估完成 task_id=%s", task_id)
                else:
                    _app_logger.warning("[worker] 评估失败 task_id=%s error=%s", task_id, result.get("error", ""))

            with scheduler_lock:
                if running_task_info.get("id") == task_id:
                    running_task_info["id"] = None
                    running_task_info["process"] = None

            if tasks[task_id]["status"] not in ["interrupted", "queued", "stopped"]:
                tasks[task_id]["result"] = result
                if result.get("status") == "success":
                    tasks[task_id]["status"] = "completed"
                    tasks[task_id]["progress"] = 100
                    tasks[task_id]["message"] = "任务完成"
                else:
                    tasks[task_id]["status"] = "error"
                    tasks[task_id]["message"] = "失败: %s" % result.get("error", "unknown")
        except Exception as e:
            _app_logger.exception("[worker] 任务异常 task_id=%s: %s", task_id, e)
            print("Worker Error:", e)
            with scheduler_lock:
                if running_task_info.get("id") == task_id:
                    running_task_info["id"] = None
            if tasks.get(task_id, {}).get("status") not in ["interrupted", "queued", "stopped"]:
                tasks[task_id]["status"] = "error"
                tasks[task_id]["message"] = "系统内部错误: %s" % str(e)
        finally:
            task_queue.task_done()


def _popen_group_kwargs():
    return ProcessManager.create_process_group_kwargs()


threading.Thread(target=worker, daemon=True).start()


# ---------- 前端页面 ----------
@app.route("/")
def index():
    return render_template("index.html") if os.path.isdir(app.template_folder) else ("<p>mergeKit_beta</p><p>请将 static 与 templates 从 mergeKit_alpha 拷贝到本目录。</p>", 200)


@app.route("/evaluation")
def evaluation_page():
    if os.path.isfile(os.path.join(app.template_folder or "", "evaluation.html")):
        return render_template("evaluation.html")
    return "<p>evaluation</p>", 200


@app.route("/testsets")
def testsets_page():
    return render_template("testsets.html") if (app.template_folder and os.path.isfile(os.path.join(app.template_folder, "testsets.html"))) else "<p>testsets</p>", 200


@app.route("/test_history")
def test_history_page():
    return render_template("test_history.html") if (app.template_folder and os.path.isfile(os.path.join(app.template_folder, "test_history.html"))) else "<p>test_history</p>", 200


@app.route("/model_repo")
def model_repo_page():
    return render_template("model_repo.html") if (app.template_folder and os.path.isfile(os.path.join(app.template_folder, "model_repo.html"))) else "<p>model_repo</p>", 200


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder or "static", filename)


# ---------- API ----------
def _list_models_from_dir(root_path):
    """从指定目录扫描可融合模型（含 config.json 的目录），返回 name/size/details/path。"""
    out = []
    if not root_path or not os.path.isdir(root_path):
        return out
    for item in os.listdir(root_path):
        full_path = os.path.join(root_path, item)
        if not os.path.isdir(full_path) or not os.path.isfile(os.path.join(full_path, "config.json")):
            continue
        size_bytes = 0
        try:
            for f in os.listdir(full_path):
                if f.endswith(".safetensors") or f.endswith(".bin"):
                    size_bytes += os.path.getsize(os.path.join(full_path, f))
        except OSError:
            pass
        out.append({
            "name": item,
            "size": size_bytes,
            "details": {"family": "HuggingFace Local"},
            "path": os.path.abspath(full_path),
        })
    return out


@app.route("/api/models")
def get_models():
    """本地基座模型列表：优先从 LOCAL_MODELS_PATH 读取，并标注 is_vlm 便于完全融合选择。"""
    try:
        base_path = getattr(Config, "LOCAL_MODELS_PATH", None) or MODEL_POOL_PATH
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
            return jsonify({"status": "success", "models": [], "base_path": base_path})
        models = _list_models_from_dir(base_path)
        for m in models:
            m["is_vlm"] = _model_is_vlm(m.get("path") or "")
        return jsonify({"status": "success", "models": models, "base_path": base_path})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/models_pool")
def get_models_pool():
    return get_models()


@app.route("/api/merged_models")
def get_merged_models():
    models = []
    if not os.path.exists(MERGE_DIR):
        return jsonify({"status": "success", "models": []})
    for task_id in os.listdir(MERGE_DIR):
        output_path = os.path.join(MERGE_DIR, task_id, "output")
        meta_path = os.path.join(MERGE_DIR, task_id, "metadata.json")
        if os.path.isfile(meta_path) and os.path.isdir(output_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if meta.get("status") == "success":
                        models.append({"name": meta.get("custom_name", task_id), "path": os.path.abspath(output_path), "type": "merged"})
            except Exception:
                continue
    return jsonify({"status": "success", "models": models})


@app.route("/api/merge", methods=["POST"])
def start_merge():
    data = request.json or {}
    priority_str = data.get("priority", "common")
    priority_score = PRIORITY_MAP.get(priority_str, 10)
    custom_name = (data.get("custom_name") or "").strip()
    if not custom_name:
        return jsonify({"status": "error", "message": "模型名称不能为空"}), 400
    if is_name_duplicate(custom_name):
        return jsonify({"status": "error", "message": "名称 '%s' 已存在。" % custom_name}), 409
    # 支持 model_paths（绝对路径）或 models（名称，相对 MODEL_POOL_PATH）
    model_paths = data.get("model_paths") or []
    models = data.get("models") or []
    if model_paths:
        resolved = []
        for p in model_paths:
            path = p if isinstance(p, str) else str(p)
            if not os.path.isabs(path) or not os.path.isdir(path):
                return jsonify({"status": "error", "message": "模型路径不存在或无效: %s" % p}), 400
            resolved.append(os.path.abspath(path))
        data["model_paths"] = resolved
    elif models:
        data["model_paths"] = None  # 由 merge_manager 用 MODEL_POOL_PATH + name 解析
    else:
        return jsonify({"status": "error", "message": "请提供 model_paths 或 models"}), 400
    task_id = str(uuid.uuid4())[:8]
    created_at = time.time()
    data["created_at"] = created_at
    data["type"] = "merge"
    _app_logger.info(
        "[API] 提交标准融合 task_id=%s custom_name=%s model_paths=%s method=%s",
        task_id, custom_name, data.get("model_paths") or data.get("models"), data.get("method"),
    )
    with scheduler_lock:
        if running_task_info["id"] and priority_score < (running_task_info["priority"] or 99):
            interrupt_current_task(reason="被 VIP 任务抢占")
        tasks[task_id] = {
            "progress": 0,
            "message": "正在加入优先级队列...",
            "status": "queued",
            "created_at": created_at,
            "original_data": data,
            "priority": priority_str,
        }
        task_queue.put((priority_score, created_at, task_id, data))
    return jsonify({"status": "success", "task_id": task_id})


@app.route("/api/merge_evolutionary", methods=["POST"])
def start_merge_evolutionary():
    """完全融合：进化迭代，使用 mergenetic，需种群大小、迭代次数、最大样本量、MMLU 子集等。"""
    data = request.json or {}
    model_paths = data.get("model_paths") or []
    if len(model_paths) < 2:
        return jsonify({"status": "error", "message": "至少需要 2 个模型"}), 400
    resolved = []
    for p in model_paths:
        if isinstance(p, str) and os.path.isabs(p) and os.path.isdir(p):
            resolved.append(p)
        else:
            path = os.path.join(MODEL_POOL_PATH, p) if isinstance(p, str) else os.path.join(MODEL_POOL_PATH, str(p))
            if not os.path.isdir(path):
                return jsonify({"status": "error", "message": "模型路径不存在: %s" % p}), 400
            resolved.append(os.path.abspath(path))
    dataset_type = "cmmmu" if "CMMMU" in (data.get("hf_dataset") or "") else "mmlu"
    hf_subset_raw = data.get("hf_subset") or data.get("hf_subset_group") or ("health_and_medicine" if dataset_type == "cmmmu" else "college_medicine")
    hf_subsets = data.get("hf_subsets")
    if not hf_subsets or not isinstance(hf_subsets, list):
        hf_subsets = _resolve_hf_subsets(dataset_type, hf_subset_raw)
    if not hf_subsets:
        hf_subsets = [hf_subset_raw]
    task_id = str(uuid.uuid4())[:8]
    created_at = time.time()
    task_data = {
        "type": "merge_evolutionary",
        "task_id": task_id,
        "custom_name": data.get("custom_name", "进化融合-%s" % task_id),
        "model_paths": resolved,
        "vlm_path": data.get("vlm_path", ""),
        "eval_mode": data.get("eval_mode", "text"),
        "hf_dataset": data.get("hf_dataset", "cais/mmlu"),
        "hf_subset": hf_subsets[0] if hf_subsets else "college_medicine",
        "hf_subsets": hf_subsets,
        "hf_split": data.get("hf_split", "test"),
        "pop_size": int(data.get("pop_size", 20)),
        "n_iter": int(data.get("n_iter", 15)),
        "max_samples": int(data.get("max_samples", 64)),
        "dtype": data.get("dtype", "bfloat16"),
        "ray_num_gpus": int(data.get("ray_num_gpus", 1)),
        "created_at": created_at,
    }
    merge_dir = os.path.join(MERGE_DIR, task_id)
    os.makedirs(merge_dir, exist_ok=True)
    with open(os.path.join(merge_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"id": task_id, "status": "pending", **task_data}, f, ensure_ascii=False, indent=2)
    _app_logger.info(
        "[API] 提交完全融合 task_id=%s custom_name=%s model_paths=%s hf_subsets=%s pop_size=%s n_iter=%s",
        task_id, task_data.get("custom_name"), resolved[:3], hf_subsets, task_data.get("pop_size"), task_data.get("n_iter"),
    )
    with scheduler_lock:
        tasks[task_id] = {
            "progress": 0,
            "message": "正在排队...",
            "status": "queued",
            "created_at": created_at,
            "original_data": task_data,
            "priority": data.get("priority", "common"),
        }
        task_queue.put((PRIORITY_MAP.get(data.get("priority", "common"), 10), created_at, task_id, task_data))
    return jsonify({"status": "success", "task_id": task_id})


@app.route("/api/evaluate", methods=["POST"])
def start_evaluation_task():
    data = request.json or {}
    model_path = data.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return jsonify({"status": "error", "message": "模型路径无效"}), 400
    task_id = str(uuid.uuid4())[:8]
    created_at = time.time()
    task_data = {
        "id": task_id,
        "type": "eval_only",
        "model_path": model_path,
        "model_name": data.get("model_name", "Unknown Model"),
        "dataset": data.get("dataset", "hellaswag"),
        "created_at": created_at,
        "status": "queued",
        "progress": 0,
        "message": "正在加入测试队列...",
    }
    _app_logger.info(
        "[API] 提交评估任务 task_id=%s model_path=%s dataset=%s",
        task_id, model_path, task_data.get("dataset"),
    )
    with scheduler_lock:
        tasks[task_id] = task_data
        task_queue.put((10, created_at, task_id, task_data))
    return jsonify({"status": "success", "task_id": task_id})


@app.route("/api/history", methods=["GET"])
def list_history():
    return jsonify({"status": "success", "history": get_all_history()})


@app.route("/api/history/<task_id>", methods=["DELETE"])
def delete_history(task_id):
    path = os.path.join(MERGE_DIR, task_id)
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "Not found"}), 404


@app.route("/api/history/<task_id>", methods=["GET"])
def get_history_detail(task_id):
    path = os.path.join(MERGE_DIR, task_id, "metadata.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return jsonify({"status": "success", "data": json.load(f)})
    return jsonify({"status": "error", "message": "Not found"}), 404


@app.route("/api/status/<task_id>")
def get_status(task_id):
    task = tasks.get(task_id)
    if task is not None:
        resp = {k: v for k, v in task.items() if k not in ("control", "original_data")}
        if task.get("status") == "queued":
            my_priority = PRIORITY_MAP.get(task.get("priority", "common"), 10)
            pos = 0
            running = 1 if running_task_info["id"] else 0
            for tid, t in tasks.items():
                if t.get("status") == "queued" and tid != task_id:
                    op = PRIORITY_MAP.get(t.get("priority", "common"), 10)
                    if op < my_priority or (op == my_priority and t.get("created_at", 0) < task.get("created_at", 0)):
                        pos += 1
            resp["queue_position"] = pos + running
        return jsonify(resp)
    # 回退：从磁盘 metadata 读取
    disk = _status_from_disk(task_id)
    if disk is not None:
        return jsonify(disk)
    return jsonify({"status": "error"}), 404


@app.route("/api/stop/<task_id>", methods=["POST"])
def stop_task(task_id):
    if task_id not in tasks:
        return jsonify({"status": "error", "message": "任务不存在"}), 404
    with scheduler_lock:
        tasks[task_id]["status"] = "stopped"
        tasks[task_id]["message"] = "任务已手动停止"
        if running_task_info["id"] == task_id:
            tasks[task_id].get("control", {})["aborted"] = True
            if running_task_info.get("process"):
                _kill_process_tree_by_pid(running_task_info["process"].pid)
            running_task_info["id"] = None
    return jsonify({"status": "success"})


@app.route("/api/resume/<task_id>", methods=["POST"])
def resume_task(task_id):
    if task_id not in tasks:
        return jsonify({"status": "error", "message": "不存在"}), 404
    with scheduler_lock:
        task = tasks[task_id]
        if task.get("status") != "interrupted":
            return jsonify({"status": "error", "message": "不可恢复"}), 400
        priority = PRIORITY_MAP.get(task.get("priority", "cutin"), 20)
        task["status"] = "queued"
        task["message"] = "已手动恢复..."
        task_queue.put((priority, task["created_at"], task_id, task["original_data"]))
    return jsonify({"status": "success"})


# ---------- model_repo / testset 占位 API（无 model_repo.api 时返回空或简单实现）----------
def _merge_metadata_by_output_path(output_path: str):
    """根据融合输出路径查找 merges/<task_id>/metadata.json，返回 meta 或 None。"""
    if not output_path or not os.path.isdir(MERGE_DIR):
        return None
    output_path = os.path.abspath(output_path)
    for tid in os.listdir(MERGE_DIR):
        out_dir = os.path.abspath(os.path.join(MERGE_DIR, tid, "output"))
        if out_dir != output_path:
            continue
        meta_path = os.path.join(MERGE_DIR, tid, "metadata.json")
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _model_repo_list():
    """返回已注册的融合模型列表，并补充父代、融合参数、数据集等详情（基座模型由 model_repo/list 的 base_models 单独返回）。"""
    data_path = os.path.join(Config.PROJECT_ROOT, "model_repo", "data", "models.json")
    if not os.path.isfile(data_path):
        return []
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models", data) if isinstance(data.get("models"), dict) else data
        if not isinstance(models, dict):
            return []
        out = []
        for m in models.values():
            m = dict(m)
            if not m.get("path") and os.path.isdir(MERGE_DIR):
                for tid in os.listdir(MERGE_DIR):
                    meta_path = os.path.join(MERGE_DIR, tid, "metadata.json")
                    if not os.path.isfile(meta_path):
                        continue
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        if meta.get("status") == "success" and meta.get("custom_name") == m.get("name"):
                            m["path"] = os.path.abspath(os.path.join(MERGE_DIR, tid, "output"))
                            break
                    except Exception:
                        continue
            m["is_base"] = False
            path = m.get("path")
            if path and os.path.isdir(path):
                meta = _merge_metadata_by_output_path(path)
                if meta:
                    recipe = m.get("recipe") or {}
                    m["parent_models"] = (
                        meta.get("models")
                        or recipe.get("parent_models")
                        or [os.path.basename(p) for p in (meta.get("model_paths") or [])]
                    )
                    m["recipe"] = {
                        "weights": recipe.get("weights"),
                        "method": recipe.get("method"),
                        "dtype": recipe.get("dtype"),
                        "library": recipe.get("library"),
                    }
                    m["dataset"] = {
                        "hf_dataset": meta.get("hf_dataset"),
                        "hf_subset": meta.get("hf_subset"),
                        "hf_subsets": meta.get("hf_subsets"),
                        "hf_split": meta.get("hf_split"),
                    }
            out.append(m)
        return out
    except Exception:
        return []


def _base_models_list():
    """基座模型列表（仅名称、路径等简要信息，并标注 is_vlm，不包含父代/参数/数据集）。"""
    base_path = getattr(Config, "LOCAL_MODELS_PATH", None) or MODEL_POOL_PATH
    models = _list_models_from_dir(base_path)
    for m in models:
        m["is_vlm"] = _model_is_vlm(m.get("path") or "")
    return models


def _model_repo_path(model_id):
    for m in _model_repo_list():
        if m.get("model_id") == model_id:
            return m.get("path")
    return None


@app.route("/api/model_repo/list")
def api_model_repo_list():
    """返回基座模型（简要）与已注册融合模型（含父代、融合参数、数据集详情）。"""
    base = _base_models_list()
    for b in base:
        b["is_base"] = True
    merged = _model_repo_list()
    return jsonify({
        "status": "success",
        "base_models": base,
        "merged_models": merged,
        "models": merged,
    })


@app.route("/api/model_repo/<model_id>/path")
def api_model_repo_path(model_id):
    path = _model_repo_path(model_id)
    if path and os.path.isdir(path):
        return jsonify({"status": "success", "path": path})
    return jsonify({"status": "error", "message": "Not found"}), 404


def _testset_list():
    data_path = os.path.join(Config.PROJECT_ROOT, "testset_repo", "data", "testsets.json")
    if not os.path.isfile(data_path):
        return []
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        testsets = data.get("testsets", data) if isinstance(data.get("testsets"), dict) else data
        if isinstance(testsets, dict):
            return list(testsets.values())
        return testsets
    except Exception:
        return []


@app.route("/api/testset/list")
def api_testset_list():
    return jsonify({"status": "success", "testsets": _testset_list()})


@app.route("/api/testset/search")
def api_testset_search():
    q = (request.args.get("q") or "").strip().lower()
    limit = int(request.args.get("limit", 20))
    all_list = _testset_list()
    if q:
        all_list = [t for t in all_list if q in (t.get("name") or "").lower() or q in (t.get("hf_dataset") or "").lower()]
    return jsonify({"status": "success", "results": all_list[:limit], "total": len(all_list)})


@app.route("/api/testset/create", methods=["POST"])
def api_testset_create():
    data = request.json or {}
    name = (data.get("name") or "").strip() or (data.get("dataset_name") or "").strip()
    if not name:
        return jsonify({"status": "error", "message": "name 或 dataset_name 必填"}), 400
    # 占位：不实际写入 testset_repo，仅返回成功
    return jsonify({"status": "success", "testset_id": str(uuid.uuid4())})


@app.route("/api/testset/<testset_id>")
def api_testset_get(testset_id):
    for t in _testset_list():
        if t.get("testset_id") == testset_id:
            return jsonify({"status": "success", "testset": t})
    return jsonify({"status": "error", "message": "Not found"}), 404


@app.route("/api/test_history")
def api_test_history():
    return jsonify({"status": "success", "history": []})


# MMLU 子集列表（保留用于兼容）
MMLU_SUBSETS = [
    "college_medicine", "college_biology", "college_chemistry", "college_physics",
    "clinical_knowledge", "professional_medicine", "anatomy", "abstract_algebra",
    "astronomy", "business_ethics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "high_school_biology", "high_school_chemistry", "high_school_physics",
    "high_school_mathematics", "high_school_computer_science", "high_school_government",
    "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions",
]

# MMLU 按领域分组：一个选项对应多个子集，训练/测试时使用更大数据集
MMLU_SUBSET_GROUPS = [
    {"id": "biology_medicine", "label": "生物/医学", "subsets": [
        "college_medicine", "college_biology", "clinical_knowledge", "professional_medicine",
        "anatomy", "medical_genetics", "virology", "high_school_biology", "human_aging", "human_sexuality",
    ]},
    {"id": "stem", "label": "STEM（数理/工程/计算机）", "subsets": [
        "college_chemistry", "college_physics", "abstract_algebra", "astronomy", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_chemistry",
        "high_school_physics", "high_school_mathematics", "high_school_computer_science", "machine_learning",
    ]},
    {"id": "humanities_social", "label": "人文/法律/社科", "subsets": [
        "high_school_government", "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
        "international_law", "jurisprudence", "philosophy", "prehistory", "sociology", "us_foreign_policy", "world_religions",
    ]},
    {"id": "business_economics", "label": "经济/商科", "subsets": [
        "business_ethics", "econometrics", "management", "marketing", "professional_accounting",
    ]},
    {"id": "other", "label": "其他", "subsets": [
        "logical_fallacies", "miscellaneous", "moral_disputes", "moral_scenarios",
        "professional_law", "professional_psychology", "public_relations", "security_studies",
    ]},
]

# CMMMU 按领域分组（MMLU 与 CMMMU 分开）
CMMMU_SUBSETS = [
    "art_and_design", "business", "health_and_medicine",
    "humanities_and_social_sciences", "science", "technology_and_engineering",
]
CMMMU_SUBSET_GROUPS = [
    {"id": "health_medicine", "label": "健康与医学", "subsets": ["health_and_medicine"]},
    {"id": "stem", "label": "STEM", "subsets": ["science", "technology_and_engineering"]},
    {"id": "humanities_art", "label": "人文与艺术", "subsets": ["art_and_design", "humanities_and_social_sciences"]},
    {"id": "business", "label": "商学", "subsets": ["business"]},
]


def _resolve_hf_subsets(dataset_type: str, subset_group_or_single: str) -> list:
    """将数据集类型 + 子集组 id 或单子集名解析为子集列表。"""
    if dataset_type == "cmmmu":
        for g in CMMMU_SUBSET_GROUPS:
            if g["id"] == subset_group_or_single:
                return list(g["subsets"])
        if subset_group_or_single in CMMMU_SUBSETS:
            return [subset_group_or_single]
    else:
        for g in MMLU_SUBSET_GROUPS:
            if g["id"] == subset_group_or_single:
                return list(g["subsets"])
        if subset_group_or_single in MMLU_SUBSETS:
            return [subset_group_or_single]
    return [subset_group_or_single] if subset_group_or_single else []


@app.route("/api/mmlu_subsets")
def api_mmlu_subsets():
    return jsonify({"status": "success", "subsets": MMLU_SUBSETS})


@app.route("/api/mmlu_subset_groups")
def api_mmlu_subset_groups():
    return jsonify({"status": "success", "groups": MMLU_SUBSET_GROUPS})


@app.route("/api/cmmmu_subsets")
def api_cmmmu_subsets():
    return jsonify({"status": "success", "subsets": CMMMU_SUBSETS})


@app.route("/api/cmmmu_subset_groups")
def api_cmmmu_subset_groups():
    return jsonify({"status": "success", "groups": CMMMU_SUBSET_GROUPS})


def _model_is_vlm(model_path: str) -> bool:
    """判断是否为视觉语言模型：config 内 vision 相关字段、model_type/architectures、目录名含 VL/Vision。"""
    if not model_path or not os.path.isdir(model_path):
        return False
    # 1. 目录名包含 VL/Vision/VLM 等（常见命名习惯，优先用于漏判）
    dir_name = os.path.basename(model_path.rstrip(os.sep)).lower()
    vlm_dir_keywords = (
        "vl", "vision", "vlm", "_vl-", "-vl-", "-vl_", "qwen2.5vl", "qwen2vl",
        "llava", "cogvlm", "minicpm-v", "minicpm_v", "visual", "qwen2_vl", "qwen2.5_vl",
    )
    if any(k in dir_name for k in vlm_dir_keywords):
        return True
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    # 2. 存在 vision_config / 图像相关 token（Qwen2.5-VL 等顶层 model_type 可能仍为 qwen2）
    if cfg.get("vision_config") is not None:
        return True
    if any(cfg.get(k) is not None for k in ("image_token_id", "vision_start_token_id", "vision_token_id")):
        return True
    # 3. model_type / architectures 中的 VLM 关键词
    model_type = (cfg.get("model_type") or "").lower()
    archs = cfg.get("architectures") or []
    arch_str = " ".join(str(a) for a in archs).lower()
    vlm_indicators = ("vision", "vl", "qwen2_vl", "qwen2.5_vl", "qwen2vl", "llava", "cogvlm", "minicpm-v", "visual")
    if any(v in model_type for v in vlm_indicators):
        return True
    if any(v in arch_str for v in vlm_indicators):
        return True
    return False


@app.route("/api/model_is_vlm")
def api_model_is_vlm():
    """查询指定路径的模型是否为 VLM，用于完全融合时自动切换数据集。"""
    path = request.args.get("path") or (request.json or {}).get("path") if request.is_json else None
    if not path or not os.path.isdir(path):
        return jsonify({"status": "error", "message": "path 无效"}), 400
    return jsonify({"status": "success", "is_vlm": _model_is_vlm(path)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("--- mergeKit_beta 后端启动 (port=%s) ---" % port)
    app.run(host="0.0.0.0", debug=True, port=port, use_reloader=False)
