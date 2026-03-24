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

from merge_manager import run_merge_task, run_eval_only_task, run_recipe_apply_task, MODEL_POOL_PATH, MERGE_DIR
from core.process_manager import ProcessManager

app = Flask(__name__, static_folder="static", template_folder="templates")


def _resolve_model_path(name_or_path: str):
    """
    将前端传来的模型名或路径解析为绝对路径。优先 LOCAL_MODELS_PATH，再 MODEL_POOL_PATH，
    支持绝对路径、相对名称、以及遍历两池下目录名匹配（便于名称一致时找到路径）。
    """
    if not name_or_path or not isinstance(name_or_path, str):
        return None
    s = name_or_path.strip()
    if not s:
        return None
    local_path = getattr(Config, "LOCAL_MODELS_PATH", None) or MODEL_POOL_PATH
    if os.path.isabs(s) and os.path.isdir(s):
        return os.path.abspath(s)
    # 先试直接拼接：LOCAL 优先（用户模型通常在此）
    for base in (local_path, MODEL_POOL_PATH):
        if not base:
            continue
        candidate = os.path.join(base, s)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    # 再试按目录名匹配：遍历两池的一级子目录
    name = os.path.basename(s)
    for base in (local_path, MODEL_POOL_PATH):
        if not base or not os.path.isdir(base):
            continue
        try:
            for item in os.listdir(base):
                full = os.path.join(base, item)
                if os.path.isdir(full) and item == name:
                    return os.path.abspath(full)
        except OSError:
            continue
    return None

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


def get_all_eval_history():
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
            if meta.get("type") != "eval_only":
                continue
            if "id" not in meta:
                meta["id"] = task_id
            if "task_id" not in meta:
                meta["task_id"] = task_id
            if not meta.get("model_name"):
                meta["model_name"] = meta.get("custom_name")
            history_list.append(meta)
        except Exception as e:
            print("Error reading eval metadata for %s: %s" % (task_id, e))
    return sorted(history_list, key=lambda x: x.get("created_at", 0), reverse=True)


def _status_from_disk(task_id):
    """从 merges/<task_id>/metadata.json 读取状态，用于 /api/status 回退。"""
    meta_path = os.path.join(MERGE_DIR, task_id, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        raw_status = meta.get("status", "unknown")
        # 前端轮询期望 status 为 "completed" 才展示结果，与 worker 内存中一致
        api_status = "completed" if raw_status == "success" else ("error" if raw_status == "error" else raw_status)
        return {
            "id": task_id,
            "status": api_status,
            "progress": 100 if raw_status == "success" else (0 if raw_status == "error" else 50),
            "message": meta.get("error") or meta.get("message", ""),
            "created_at": meta.get("created_at"),
            "result": {"status": raw_status, "metrics": meta.get("metrics")},
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
            elif task_type == "recipe_apply":
                _app_logger.info(
                    "[worker] 配方应用 recipe_id=%s",
                    data.get("recipe_id", ""),
                )

            if task_type == "merge_evolutionary":
                import merge_manager as _mm
                _data = dict(data)
                _temp_suffixes = []
                _result = None
                if _data.get("items") and len(_data["items"]) == 2:
                    resolved_paths = []
                    for idx, it in enumerate(_data["items"]):
                        if it.get("type") == "path" and it.get("path"):
                            resolved_paths.append(it["path"])
                        elif it.get("type") == "recipe" and it.get("recipe_id"):
                            suffix = "a" if idx == 0 else "b"
                            _temp_suffixes.append(suffix)
                            update_progress(5 + idx * 10, "正在根据配方物化模型…")
                            path, err = _mm.materialize_recipe_to_temp(
                                it["recipe_id"], task_id, suffix, update_progress, task_control
                            )
                            if err:
                                _result = {"status": "error", "error": "物化配方失败: %s" % err}
                                break
                            resolved_paths.append(path)
                        else:
                            _result = {"status": "error", "error": "items[%d] 格式无效" % idx}
                            break
                    else:
                        _data["model_paths"] = resolved_paths
                if _result is None:
                    script_path = os.path.join(Config.PROJECT_ROOT, "scripts", "run_vlm_search_bridge.py")
                    if not os.path.isfile(script_path):
                        tasks[task_id]["status"] = "error"
                        tasks[task_id]["message"] = "scripts/run_vlm_search_bridge.py 不存在"
                        _result = {"status": "error", "error": "run_vlm_search_bridge.py 未找到"}
                    else:
                        import subprocess
                        merge_dir = os.path.join(MERGE_DIR, task_id)
                        os.makedirs(merge_dir, exist_ok=True)
                        progress_path = os.path.join(merge_dir, "progress.json")
                        meta_path = os.path.join(merge_dir, "metadata.json")
                        _mm._write_metadata(task_id, merge_dir, {"id": task_id, "type": "merge_evolutionary", "status": "running", **_data})
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
                            _result = {"status": "success"}
                            _app_logger.info("[worker] 完全融合完成 task_id=%s", task_id)
                            if _temp_suffixes:
                                _mm.cleanup_recipe_temp_dirs(task_id, _temp_suffixes)
                        except Exception as e:
                            _app_logger.exception("[worker] 完全融合失败 task_id=%s error=%s", task_id, e)
                            with open(progress_path, "w", encoding="utf-8") as f:
                                json.dump({"status": "error", "message": str(e)}, f, ensure_ascii=False)
                            try:
                                with open(meta_path, "r", encoding="utf-8") as f:
                                    m = json.load(f)
                                m["status"] = "error"
                                m["error"] = str(e)
                                _mm._write_metadata(task_id, merge_dir, m)
                            except Exception:
                                pass
                            _result = {"status": "error", "error": str(e)}
                            if _temp_suffixes:
                                _mm.cleanup_recipe_temp_dirs(task_id, _temp_suffixes)
                result = _result
            elif task_type == "eval_only":
                # 每次评估前重新加载 merge_manager，确保使用最新 run_eval_only_task 实现（避免进程未重启时仍跑旧占位逻辑）
                import importlib
                import merge_manager as _mm
                importlib.reload(_mm)
                result = _mm.run_eval_only_task(task_id, data, update_progress, task_control)
            elif task_type == "recipe_apply":
                result = run_recipe_apply_task(task_id, data, update_progress, task_control)
                if result.get("status") == "success":
                    _app_logger.info("[worker] 配方融合完成 task_id=%s output=%s", task_id, result.get("output_path", ""))
                else:
                    _app_logger.warning("[worker] 配方融合失败 task_id=%s error=%s", task_id, result.get("error", ""))
            else:
                # 标准融合：若存在 items（含 recipe），先物化再融合并清理临时目录
                _data = dict(data)
                _merge_temp_suffixes = []
                if _data.get("items") and len(_data["items"]) == 2:
                    import merge_manager as _mm_merge
                    paths = []
                    for idx, it in enumerate(_data["items"]):
                        if it.get("type") == "path" and it.get("path"):
                            paths.append(it["path"])
                        elif it.get("type") == "recipe" and it.get("recipe_id"):
                            suf = "m1" if idx == 0 else "m2"
                            _merge_temp_suffixes.append(suf)
                            path, err = _mm_merge.materialize_recipe_to_temp(
                                it["recipe_id"], task_id, suf, update_progress, task_control
                            )
                            if err:
                                result = {"status": "error", "error": "物化配方失败: %s" % err}
                                break
                            paths.append(path)
                        else:
                            result = {"status": "error", "error": "items[%d] 格式无效" % idx}
                            break
                    else:
                        _data["model_paths"] = paths
                        result = run_merge_task(task_id, _data, update_progress, task_control)
                    if _merge_temp_suffixes:
                        _mm_merge.cleanup_recipe_temp_dirs(task_id, _merge_temp_suffixes)
                else:
                    result = run_merge_task(task_id, data, update_progress, task_control)
                if result and result.get("status") == "success":
                    _app_logger.info("[worker] 标准融合完成 task_id=%s output=%s", task_id, result.get("output_dir", ""))
                elif result and result.get("status") != "success":
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


def _output_has_safetensors(output_path):
    """判断 output 目录是否包含完整权重（至少一个 .safetensors）。"""
    if not output_path or not os.path.isdir(output_path):
        return False
    try:
        for f in os.listdir(output_path):
            if f.endswith(".safetensors"):
                return True
    except OSError:
        pass
    return False


@app.route("/api/models")
def get_models():
    """本地基座模型列表：同时扫描 LOCAL_MODELS_PATH 与 MODEL_POOL_PATH，去重并标注 is_vlm。"""
    try:
        seen_paths = set()
        models = []
        for base_path in (
            getattr(Config, "LOCAL_MODELS_PATH", None),
            MODEL_POOL_PATH,
        ):
            if not base_path or not os.path.exists(base_path):
                if base_path:
                    os.makedirs(base_path, exist_ok=True)
                continue
            for m in _list_models_from_dir(base_path):
                path = (m.get("path") or "").strip()
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    m["is_vlm"] = _model_is_vlm(path)
                    m["source"] = "base"
                    models.append(m)
        base_path = getattr(Config, "LOCAL_MODELS_PATH", None) or MODEL_POOL_PATH
        return jsonify({"status": "success", "models": models, "base_path": base_path})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/models_pool")
def get_models_pool():
    return get_models()


@app.route("/api/merged_models")
def get_merged_models():
    """已融合模型列表：仅包含 output 目录下存在 .safetensors 的条目，并标注 is_vlm。"""
    models = []
    if not os.path.exists(MERGE_DIR):
        return jsonify({"status": "success", "models": models})
    for task_id in os.listdir(MERGE_DIR):
        output_path = os.path.join(MERGE_DIR, task_id, "output")
        meta_path = os.path.join(MERGE_DIR, task_id, "metadata.json")
        if not os.path.isfile(meta_path) or not os.path.isdir(output_path):
            continue
        if not _output_has_safetensors(output_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("status") != "success":
                continue
            path = os.path.abspath(output_path)
            models.append({
                "name": meta.get("custom_name", task_id),
                "path": path,
                "type": "merged",
                "is_vlm": _model_is_vlm(path),
            })
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
    # 支持 model_paths（两个路径）、models（名称）、或 items（path/recipe 混合）
    model_paths = data.get("model_paths") or []
    models = data.get("models") or []
    items = data.get("items")
    if items and len(items) == 2:
        resolved = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                return jsonify({"status": "error", "message": "items[%d] 格式无效" % i}), 400
            if it.get("type") == "recipe" and (it.get("recipe_id") or "").strip():
                resolved.append({"type": "recipe", "recipe_id": (it.get("recipe_id") or "").strip()})
            elif it.get("type") == "path" or it.get("path"):
                path = _resolve_model_path(it.get("path") or "")
                if not path:
                    return jsonify({"status": "error", "message": "items[%d] 路径无效" % i}), 400
                resolved.append({"type": "path", "path": path})
            else:
                return jsonify({"status": "error", "message": "items[%d] 需为 path 或 recipe" % i}), 400
        data["items"] = resolved
        data["model_paths"] = [x["path"] for x in resolved if x.get("type") == "path"]
    elif model_paths:
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
        return jsonify({"status": "error", "message": "请提供 model_paths、models 或 items"}), 400
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
    """完全融合：进化迭代。支持 model_paths（两个路径）或 items（两个 path/recipe 项）。"""
    data = request.json or {}
    items = data.get("items")
    model_paths = data.get("model_paths") or []
    resolved = []
    if items and len(items) == 2:
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                return jsonify({"status": "error", "message": "items[%d] 格式无效" % i}), 400
            if it.get("type") == "recipe":
                rid = (it.get("recipe_id") or "").strip()
                if not rid:
                    return jsonify({"status": "error", "message": "items[%d] 缺少 recipe_id" % i}), 400
                resolved.append({"type": "recipe", "recipe_id": rid})
            elif it.get("type") == "path" or it.get("path"):
                path = _resolve_model_path(it.get("path") or "")
                if not path:
                    return jsonify({"status": "error", "message": "items[%d] 路径无效: %s" % (i, it.get("path", ""))}), 400
                resolved.append({"type": "path", "path": path})
            else:
                return jsonify({"status": "error", "message": "items[%d] 需为 type:path 或 type:recipe" % i}), 400
    else:
        if len(model_paths) < 2:
            return jsonify({"status": "error", "message": "至少需要 2 个模型（或提供 items 数组）"}), 400
        for p in model_paths:
            path = _resolve_model_path(p)
            if not path:
                return jsonify({
                    "status": "error",
                    "message": "模型路径不存在: %s（已尝试 LOCAL_MODELS_PATH 与 MODEL_POOL_PATH 及目录名匹配）" % (p,),
                }), 400
            resolved.append({"type": "path", "path": path})
    dataset_type = "cmmmu" if "CMMMU" in (data.get("hf_dataset") or "") else "mmlu"
    hf_subset_raw = data.get("hf_subset") or data.get("hf_subset_group") or ("health_and_medicine" if dataset_type == "cmmmu" else "college_medicine")
    hf_subsets = data.get("hf_subsets")
    if not hf_subsets or not isinstance(hf_subsets, list):
        hf_subsets = _resolve_hf_subsets(dataset_type, hf_subset_raw)
    if not hf_subsets:
        hf_subsets = [hf_subset_raw]
    task_id = str(uuid.uuid4())[:8]
    created_at = time.time()
    has_recipe = any(x.get("type") == "recipe" for x in resolved)
    task_data = {
        "type": "merge_evolutionary",
        "task_id": task_id,
        "custom_name": data.get("custom_name", "进化融合-%s" % task_id),
        "model_paths": [x.get("path") for x in resolved if x.get("type") == "path"] if not has_recipe else [],
        "items": resolved if has_recipe else None,
        "vlm_path": data.get("vlm_path", ""),
        "eval_mode": data.get("eval_mode", "text"),
        "hf_dataset": data.get("hf_dataset", "cais/mmlu"),
        "hf_subset": hf_subsets[0] if hf_subsets else "college_medicine",
        "hf_subsets": hf_subsets,
        "hf_subset_group": hf_subset_raw if hf_subset_raw in [g["id"] for g in (MMLU_SUBSET_GROUPS + CMMMU_SUBSET_GROUPS)] else "",  # 领域 id（如 stem），供本地按领域目录加载
        "hf_split": data.get("hf_split", "test"),
        "hf_split_final": (data.get("hf_split_final") or "").strip() or None,  # 最终评测用 split，如 test
        "pop_size": max(2, min(128, int(data.get("pop_size", 20)))),
        "n_iter": max(1, min(50, int(data.get("n_iter", 15)))),
        "max_samples": max(4, min(512, int(data.get("max_samples", 64)))),
        "dtype": data.get("dtype", "bfloat16"),
        "ray_num_gpus": int(data.get("ray_num_gpus", 1)),
        "created_at": created_at,
    }
    merge_dir = os.path.join(MERGE_DIR, task_id)
    os.makedirs(merge_dir, exist_ok=True)
    _mm._write_metadata(task_id, merge_dir, {"id": task_id, "status": "pending", **task_data})
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
    dataset = data.get("dataset", "hellaswag")
    testset_id = (data.get("testset_id") or "").strip()
    hf_dataset = (data.get("hf_dataset") or "").strip() or None
    hf_subset = (data.get("hf_subset") or "").strip() or None
    hf_split = (data.get("hf_split") or "").strip() or "test"
    lm_eval_task = ""
    if testset_id:
        testset = None
        for t in _testset_list():
            if t.get("testset_id") == testset_id:
                testset = t
                break
        if testset:
            dataset = testset.get("hf_dataset") or dataset
            hf_dataset = hf_dataset or testset.get("hf_dataset")
            hf_subset = (hf_subset or "").strip() or testset.get("hf_subset") or None
            hf_split = (hf_split or testset.get("hf_split") or "test").strip() or "test"
            lm_eval_task = (testset.get("lm_eval_task") or "").strip()
            if not lm_eval_task:
                # 尝试推断：优先使用 hf_subset，如果没有则尝试从 hf_dataset 推断
                if hf_dataset and hf_subset:
                    lm_eval_task = _infer_lm_eval_task(hf_dataset, hf_subset)
                elif hf_dataset:
                    # 对于没有 subset 的数据集，尝试从映射文件或内置映射中查找
                    try:
                        import merge_manager as mm
                        extra = mm._load_eval_task_mapping()
                        hf_key = hf_dataset.strip().lower()
                        if hf_dataset in extra:
                            lm_eval_task = extra[hf_dataset]
                        elif hf_key in mm.HF_DATASET_TO_LM_EVAL_TASK:
                            lm_eval_task = mm.HF_DATASET_TO_LM_EVAL_TASK[hf_key]
                        # 如果仍然找不到，尝试自动发现
                        if not lm_eval_task:
                            discovered_task = mm._auto_discover_lm_eval_task(hf_dataset, hf_subset)
                            if discovered_task:
                                lm_eval_task = discovered_task
                                # 保存到映射文件和测试集记录
                                exact_key = "%s|%s" % (hf_dataset.strip(), hf_subset.strip()) if hf_subset else hf_dataset.strip()
                                extra[exact_key] = discovered_task
                                if not hf_subset:
                                    extra[hf_dataset.strip()] = discovered_task
                                mm._save_eval_task_mapping(extra)
                                _app_logger.info("[eval] 已自动发现并保存映射: %s -> %s", exact_key, discovered_task)
                    except Exception:
                        pass  # 如果导入失败，保持 lm_eval_task 为空，让 merge_manager 处理
                if lm_eval_task:
                    testsets = _load_testsets_dict()
                    for tid, t in list(testsets.items()):
                        if t.get("testset_id") == testset_id:
                            testsets[tid] = {**t, "lm_eval_task": lm_eval_task}
                            _save_testsets_dict(testsets)
                            break
        else:
            if not hf_dataset and "/" in testset_id:
                hf_dataset = testset_id
            if hf_dataset:
                dataset = hf_dataset
            testset_id = ""
    elif hf_dataset:
        dataset = hf_dataset
    task_data = {
        "id": task_id,
        "type": "eval_only", 
        "model_path": model_path,
        "model_name": data.get("model_name", "Unknown Model"),
        "dataset": dataset,
        "created_at": created_at,
        "status": "queued",
        "progress": 0,
        "message": "正在加入测试队列...",
        "limit": data.get("limit", "0.5"),
        "sampling": data.get("sampling", "sequential"),
        "testset_id": testset_id or None,
        "hf_dataset": hf_dataset,
        "hf_subset": hf_subset,
        "hf_split": hf_split,
        "lm_eval_task": lm_eval_task or None,
    }
    _app_logger.info(
        "[API] 提交评估任务 task_id=%s model_path=%s dataset=%s testset_id=%s hf_dataset=%s hf_subset=%s",
        task_id, model_path, task_data.get("dataset"), task_data.get("testset_id"),
        task_data.get("hf_dataset"), task_data.get("hf_subset"),
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


def _read_evolution_progress(task_id):
    """完全融合任务：读取 merges/<task_id>/progress.json 中的迭代/种群/当前最优等，供前端展示。"""
    progress_path = os.path.join(MERGE_DIR, task_id, "progress.json")
    if not os.path.isfile(progress_path):
        return None
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status") == "error":
            return {"error": data.get("message", ""), "step": 0}
        result = {
            "step": data.get("step", 0),
            "current_best": data.get("current_best"),
            "global_best": data.get("global_best"),
            "best_genotype": data.get("best_genotype"),
        }
        # 添加 ETA 和进度信息
        if "eta_seconds" in data:
            result["eta_seconds"] = data.get("eta_seconds")
            result["estimated_completion"] = data.get("estimated_completion")
        if "current_step" in data:
            result["current_step"] = data.get("current_step")
            result["total_expected_steps"] = data.get("total_expected_steps")
        return result
    except Exception:
        return None


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
        # 完全融合运行中或刚完成：附带 progress.json 的详细进度（迭代/种群/当前最优）及 n_iter/pop_size 供前端进度条
        if task.get("original_data", {}).get("type") == "merge_evolutionary" or task.get("type") == "merge_evolutionary":
            evo = _read_evolution_progress(task_id)
            if evo is not None:
                resp["evolution_progress"] = evo
            od = task.get("original_data") or {}
            resp["original_data"] = {"n_iter": od.get("n_iter"), "pop_size": od.get("pop_size")}
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


def _model_repo_load_raw():
    """从 model_repo 存储文件加载 models 字典（用于删除等写操作）。"""
    data_path = os.path.join(Config.PROJECT_ROOT, "model_repo", "data", "models.json")
    if not os.path.isfile(data_path):
        return {}
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("models", data) if isinstance(data.get("models"), dict) else {}


def _model_repo_save_raw(models):
    data_path = os.path.join(Config.PROJECT_ROOT, "model_repo", "data", "models.json")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump({"models": models}, f, ensure_ascii=False, indent=2)


@app.route("/api/model_repo/<model_id>", methods=["DELETE"])
def api_model_repo_delete(model_id):
    """删除模型：从仓库移除并删除磁盘上的模型目录。二次确认由前端完成。"""
    model_id = (model_id or "").strip()
    if not model_id:
        return jsonify({"status": "error", "message": "模型 ID 无效"}), 400
    models = _model_repo_load_raw()
    if not isinstance(models, dict):
        return jsonify({"status": "error", "message": "仓库数据异常"}), 500
    m = models.get(model_id)
    if not m:
        return jsonify({"status": "error", "message": "模型不存在"}), 404
    path = m.get("path")
    if path and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            _app_logger.info("[model_repo] 已删除磁盘目录: %s", path)
        except Exception as e:
            _app_logger.exception("[model_repo] 删除目录失败: %s", path)
            return jsonify({"status": "error", "message": "删除磁盘文件失败: %s" % str(e)}), 500
    try:
        del models[model_id]
        _model_repo_save_raw(models)
    except Exception as e:
        _app_logger.exception("[model_repo] 保存仓库失败")
        return jsonify({"status": "error", "message": "更新仓库失败: %s" % str(e)}), 500
    return jsonify({"status": "success"})


@app.route("/api/resolve_model_path")
def api_resolve_model_path():
    """根据模型名或路径解析为绝对路径，供前端校验或展示。"""
    name_or_path = request.args.get("name") or request.args.get("path") or ""
    path = _resolve_model_path(name_or_path)
    if path:
        return jsonify({"status": "success", "path": path})
    return jsonify({"status": "error", "message": "未找到对应模型目录"}), 404


def _testset_list(refresh=True):
    testsets = _load_testsets_dict()
    return _refresh_testsets_counts(testsets) if refresh else list(testsets.values())


def _refresh_single_testset_count(entry):
    if not entry or not (entry.get("hf_dataset") or "").strip():
        return entry
    if int(entry.get("sample_count") or 0) > 0:
        return entry
    cache_dir = getattr(Config, "HF_DATASETS_CACHE", None) or os.environ.get("HF_DATASETS_CACHE")
    n, actual_split, actual_subset = _resolve_dataset_sample_count(
        entry.get("hf_dataset"),
        entry.get("hf_subset"),
        entry.get("hf_split"),
        cache_dir,
    )
    if n > 0:
        entry["sample_count"] = n
        if actual_split:
            entry["hf_split"] = actual_split
        if actual_subset and not (entry.get("hf_subset") or "").strip():
            entry["hf_subset"] = actual_subset
    return entry


def _get_testset_by_id(testset_id, refresh=False):
    testsets = _load_testsets_dict()
    entry = testsets.get(testset_id)
    if not entry:
        return None
    if refresh:
        entry = _refresh_single_testset_count(entry)
        testsets[testset_id] = entry
        _save_testsets_dict(testsets)
    return entry


@app.route("/api/testset/list")
def api_testset_list():
    refresh = request.args.get("refresh", "0") in ("1", "true", "yes")
    return jsonify({"status": "success", "testsets": _testset_list(refresh=refresh)})


@app.route("/api/testset/search")
def api_testset_search():
    q = (request.args.get("q") or "").strip().lower()
    limit = int(request.args.get("limit", 20))
    refresh = request.args.get("refresh", "0") in ("1", "true", "yes")
    all_list = _testset_list(refresh=refresh)
    if q:
        all_list = [t for t in all_list if q in (t.get("name") or "").lower() or q in (t.get("hf_dataset") or "").lower()]
    return jsonify({"status": "success", "results": all_list[:limit], "total": len(all_list)})


@app.route("/api/hf/datasets/search", methods=["GET", "POST"])
def api_hf_datasets_search():
    """在 HuggingFace 镜像上搜索数据集。GET/POST 均可，q 为搜索关键词。"""
    q = (request.args.get("q") or (request.get_json(silent=True) or {}).get("q") or "").strip()
    limit = int(request.args.get("limit") or (request.get_json(silent=True) or {}).get("limit", 20) or 20)
    if not q:
        return jsonify({"status": "error", "message": "q 必填"}), 400
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # 使用镜像
        endpoint = getattr(Config, "HF_ENDPOINT", None) or os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        ds_list = list(api.list_datasets(search=q, limit=min(limit, 50)))
        out = [{"id": d.id, "author": getattr(d, "author", None), "downloads": getattr(d, "downloads", None)} for d in ds_list]
        return jsonify({"status": "success", "results": out})
    except Exception as e:
        _app_logger.exception("api_hf_datasets_search: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


def _testset_repo_path():
    return getattr(Config, "TESTSET_DATA_PATH", None) or os.path.join(Config.PROJECT_ROOT, "testset_repo", "data", "testsets.json")


def _resolve_dataset_sample_count(hf_dataset, hf_subset, hf_split, cache_dir):
    if not (hf_dataset or "").strip():
        return 0, None, None

    # [Patch] Handle MMLU custom domain subsets by using a proxy subset to get splits
    if hf_dataset == "cais/mmlu" and hf_subset in ["stem", "biology", "law", "society", "humanities", "other"]:
        hf_subset = "abstract_algebra"

    try:
        from datasets import load_dataset_builder, get_dataset_config_names
        subset_candidates = []
        if (hf_subset or "").strip():
            subset_candidates = [hf_subset]
        else:
            subset_candidates = [None]
            try:
                configs = get_dataset_config_names(hf_dataset, trust_remote_code=True) or []
                subset_candidates.extend([c for c in configs if c])
            except Exception:
                pass
        for subset_name in subset_candidates:
            try:
                builder = load_dataset_builder(hf_dataset, name=subset_name or None, trust_remote_code=True, cache_dir=cache_dir)
                splits = getattr(builder.info, "splits", None) or {}
                if splits:
                    preferred = []
                    if hf_split:
                        preferred.append(hf_split.split("[")[0].strip())
                    preferred.extend(["test", "validation", "dev", "val", "train"])
                    for split_name in preferred:
                        if split_name in splits:
                            return int(getattr(splits[split_name], "num_examples", 0) or 0), split_name, subset_name
                    first_key = list(splits.keys())[0]
                    return int(getattr(splits[first_key], "num_examples", 0) or 0), first_key, subset_name
            except Exception:
                continue
    except Exception as e:
        _app_logger.debug("testset count metadata failed: %s", e)
    return 0, None, None


def _refresh_testsets_counts(testsets_dict):
    if not testsets_dict:
        return []
    cache_dir = getattr(Config, "HF_DATASETS_CACHE", None) or os.environ.get("HF_DATASETS_CACHE")
    changed = False
    for tid, entry in list(testsets_dict.items()):
        if not entry:
            continue
        if not (entry.get("hf_dataset") or "").strip():
            continue
        if int(entry.get("sample_count") or 0) > 0:
            continue
        n, actual_split, actual_subset = _resolve_dataset_sample_count(
            entry.get("hf_dataset"),
            entry.get("hf_subset"),
            entry.get("hf_split"),
            cache_dir,
        )
        if n > 0:
            entry["sample_count"] = n
            if actual_split:
                entry["hf_split"] = actual_split
            if actual_subset and not (entry.get("hf_subset") or "").strip():
                entry["hf_subset"] = actual_subset
            testsets_dict[tid] = entry
            changed = True
    if changed:
        _save_testsets_dict(testsets_dict)
    return list(testsets_dict.values())


def _infer_lm_eval_task(hf_dataset, hf_subset):
    """根据 hf_dataset / hf_subset 推断 lm_eval 任务名，与 merge_manager 逻辑一致。用于测试集创建或首次选用时自动写入映射。"""
    if not (hf_dataset or "").strip():
        return ""
    key = (hf_dataset or "").strip().lower()
    if key in ("cais/mmlu", "mmlu") and (hf_subset or "").strip():
        return "mmlu_" + (hf_subset or "").strip().replace("-", "_")
    return ""


def _load_testsets_dict():
    path = _testset_repo_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("testsets", data) if isinstance(data.get("testsets"), dict) else {}
    except Exception:
        return {}


def _save_testsets_dict(testsets_dict):
    """DB-first：先写 DB，再写文件缓存。"""
    if testsets_dict:
        try:
            from app.repositories import testset_upsert as _ts_upsert
            for tid, entry in testsets_dict.items():
                if not entry or not isinstance(entry, dict):
                    continue
                testset_id = entry.get("testset_id") or tid
                _ts_upsert(
                    testset_id=str(testset_id),
                    name=entry.get("name") or str(testset_id) or "未命名",
                    hf_dataset=entry.get("hf_dataset"),
                    hf_subset=entry.get("hf_subset"),
                    hf_split=entry.get("hf_split"),
                    lm_eval_task=entry.get("lm_eval_task"),
                    benchmark_config=entry.get("benchmark_config"),
                    version=entry.get("version"),
                    sample_count=int(entry.get("sample_count") or 0),
                    is_local=bool(entry.get("is_local")),
                    local_path=entry.get("local_path"),
                    yaml_template_path=entry.get("yaml_template_path"),
                    created_by=entry.get("created_by"),
                    notes=entry.get("notes"),
                    question_type=entry.get("question_type"),
                    type=entry.get("type"),
                )
        except Exception as e:
            _app_logger.warning("[_save_testsets_dict] DB upsert 失败: %s", e)
    try:
        path = _testset_repo_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"testsets": testsets_dict}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _app_logger.warning("[_save_testsets_dict] 文件缓存写入失败: %s", e)


@app.route("/api/testset/create", methods=["POST"])
def api_testset_create():
    """创建测试集：从 HuggingFace 下载并登记到 testset_repo，测试集仓库可见。"""
    data = request.json or {}
    name = (data.get("name") or "").strip() or (data.get("dataset_name") or "").strip()
    hf_dataset = (data.get("hf_dataset") or "").strip()
    if not hf_dataset:
        return jsonify({"status": "error", "message": "hf_dataset 必填"}), 400
    if not name:
        name = hf_dataset
    hf_subset_raw = (data.get("hf_subset") or "").strip()
    hf_subset = hf_subset_raw if hf_subset_raw else None  # 空字符串转为 None
    hf_split_raw = (data.get("hf_split") or "").strip()
    hf_split = hf_split_raw if hf_split_raw else None
    cache_dir = getattr(Config, "HF_DATASETS_CACHE", None) or os.environ.get("HF_DATASETS_CACHE")
    # 尝试多个 split（test/validation/train），优先 test/validation 用于评估
    split_candidates = []
    if hf_split:
        split_candidates.append(hf_split.split("[")[0].strip())
    split_candidates.extend(["test", "validation", "dev", "val", "train"])
    split_candidates = list(dict.fromkeys(split_candidates))  # 去重保持顺序
    n = 0
    actual_split = None
    load_error = None
    try:
        from datasets import load_dataset
        for split_name in split_candidates:
            try:
                if hf_subset:
                    ds = load_dataset(hf_dataset, hf_subset, split=split_name, trust_remote_code=True, cache_dir=cache_dir)
                else:
                    ds = load_dataset(hf_dataset, split=split_name, trust_remote_code=True, cache_dir=cache_dir)
                # 处理 DatasetDict（返回所有 splits）和 Dataset（单个 split）两种情况
                if ds:
                    try:
                        n = 0
                        # 优先尝试按 split_name 访问（DatasetDict 情况）
                        if hasattr(ds, "__getitem__") and hasattr(ds, "keys"):
                            try:
                                keys_list = list(ds.keys())
                                if split_name in keys_list:
                                    n = len(ds[split_name]) if hasattr(ds[split_name], "__len__") else 0
                                elif len(keys_list) > 0:
                                    # 如果请求的 split 不存在，取第一个可用的
                                    first_key = keys_list[0]
                                    n = len(ds[first_key]) if hasattr(ds[first_key], "__len__") else 0
                                    if n > 0:
                                        actual_split = first_key  # 使用实际存在的 split
                            except (KeyError, AttributeError, TypeError) as dict_err:
                                _app_logger.debug("testset create 访问 DatasetDict 失败: %s", dict_err)
                        # 如果 DatasetDict 访问失败或 n 仍为 0，尝试直接取长度（Dataset 情况）
                        if n == 0 and hasattr(ds, "__len__"):
                            try:
                                n = len(ds)
                            except Exception as len_err:
                                _app_logger.debug("testset create 获取 Dataset 长度失败: %s", len_err)
                        # 如果成功获取到样本数，记录并跳出循环
                        if n > 0:
                            if not actual_split:
                                actual_split = split_name
                            break
                    except Exception as parse_err:
                        _app_logger.debug("testset create 解析数据集失败: %s", parse_err)
                        continue
            except Exception as split_err:
                load_error = str(split_err)
                _app_logger.debug("testset create 尝试 split=%s 失败: %s", split_name, split_err)
                continue
        if n == 0:
            _app_logger.warning("testset create 所有 split 均失败: hf_dataset=%s hf_subset=%s splits=%s error=%s", hf_dataset, hf_subset, split_candidates, load_error)
            # 尝试最后一次：不指定 split，让 load_dataset 返回所有 splits
            if not hf_subset:
                try:
                    ds_all = load_dataset(hf_dataset, trust_remote_code=True, cache_dir=cache_dir)
                    if ds_all:
                        if hasattr(ds_all, "__getitem__") and hasattr(ds_all, "keys"):
                            keys_list = list(ds_all.keys())
                            if len(keys_list) > 0:
                                first_key = keys_list[0]
                                n = len(ds_all[first_key]) if hasattr(ds_all[first_key], "__len__") else 0
                                if n > 0 and not actual_split:
                                    actual_split = first_key
                        elif hasattr(ds_all, "__len__"):
                            n = len(ds_all)
                except Exception as fallback_err:
                    _app_logger.debug("testset create 回退加载失败: %s", fallback_err)
    except Exception as e:
        load_error = str(e)
        _app_logger.warning("testset create load_dataset 异常: %s", e)
    if n == 0:
        meta_n, meta_split, meta_subset = _resolve_dataset_sample_count(hf_dataset, hf_subset, hf_split, cache_dir)
        if meta_n > 0:
            n = meta_n
            if not actual_split:
                actual_split = meta_split
            if not hf_subset and meta_subset:
                hf_subset = meta_subset
    # 使用实际成功的 split，否则用第一个候选
    hf_split = actual_split or (split_candidates[0] if split_candidates else "test")
    testset_id = str(uuid.uuid4())
    prompt_path = os.path.join(Config.PROJECT_ROOT, "yaml_template", "prompt.yaml")
    if not os.path.isfile(prompt_path):
        prompt_path = ""
    lm_eval_task = _infer_lm_eval_task(hf_dataset, hf_subset)
    # 如果无法推断，尝试自动发现
    if not lm_eval_task and hf_dataset:
        try:
            import merge_manager as mm
            discovered_task = mm._auto_discover_lm_eval_task(hf_dataset, hf_subset)
            if discovered_task:
                lm_eval_task = discovered_task
                # 自动保存映射到配置文件
                try:
                    extra = mm._load_eval_task_mapping()
                    hf_key = hf_dataset.strip()
                    subset_key = hf_subset.strip() if hf_subset else None
                    exact_key = "%s|%s" % (hf_key, subset_key) if subset_key else hf_key
                    extra[exact_key] = discovered_task
                    if not subset_key:
                        extra[hf_key] = discovered_task
                    mm._save_eval_task_mapping(extra)
                    _app_logger.info("[testset] 已自动发现并保存映射: %s -> %s", exact_key, discovered_task)
                except Exception as save_err:
                    _app_logger.warning("[testset] 自动保存映射失败: %s", save_err)
        except Exception as discover_err:
            _app_logger.debug("[testset] 自动发现任务失败: %s", discover_err)
    if not lm_eval_task and hf_dataset:
        try:
            import merge_manager as mm
            available = mm._get_available_lm_eval_tasks()
            hf_key = hf_dataset.strip().lower()
            base_name = hf_key.split("/")[-1]
            target = None
            for t in available:
                t_lower = t.lower()
                if t_lower == hf_key or t_lower == base_name:
                    target = t
                    break
            if target:
                lm_eval_task = target
                try:
                    extra = mm._load_eval_task_mapping()
                    exact_key = "%s|%s" % (hf_dataset.strip(), hf_subset.strip()) if hf_subset else hf_dataset.strip()
                    extra[exact_key] = target
                    if not hf_subset:
                        extra[hf_dataset.strip()] = target
                    mm._save_eval_task_mapping(extra)
                except Exception as save_err:
                    _app_logger.warning("[testset] 自动保存映射失败: %s", save_err)
        except Exception as e:
            _app_logger.debug("[testset] 任务精确匹配失败: %s", e)
    
    entry = {
        "testset_id": testset_id,
        "name": name,
        "hf_dataset": hf_dataset,
        "hf_subset": hf_subset,
        "hf_split": hf_split,
        "lm_eval_task": lm_eval_task or "",  # 即使为空也保存，后续评估时会再次尝试自动发现
        "yaml_template_path": prompt_path,
        "sample_count": n,
        "created_at": time.time(),
        "created_by": "user",
        "notes": "从 HuggingFace 添加: %s" % hf_dataset,
        "question_type": "未知",
    }
    testsets = _load_testsets_dict()
    testsets[testset_id] = entry
    _save_testsets_dict(testsets)
    return jsonify({"status": "success", "testset_id": testset_id, "testset": entry})





@app.route("/api/test_history")
def api_test_history():
    return jsonify({"status": "success", "history": get_all_eval_history()})


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
    return jsonify({
        "status": "success",
        "subsets": CMMMU_SUBSETS,
        "count": len(CMMMU_SUBSETS),
        "note": "不选子集或选「全部」时，评估将优先使用 cmmmu_*；若环境无该任务，将回退到 mmmu_val_* 兼容任务",
    })


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


def _get_model_type(model_path: str) -> str | None:
    """从 config.json 读取 model_type。"""
    if not model_path or not os.path.isdir(model_path):
        return None
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return (cfg.get("model_type") or "").strip().lower()
    except Exception:
        return None


def _get_model_arch(model_path: str) -> tuple[int | None, int | None]:
    """
    从 config.json 读取 hidden_size 与 num_hidden_layers，用于判断是否同一架构可配对融合。
    返回 (hidden_size, num_hidden_layers)，若缺失则对应为 None。
    """
    if not model_path or not os.path.isdir(model_path):
        return None, None
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None, None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        hs = cfg.get("hidden_size")
        nhl = cfg.get("num_hidden_layers")
        if hs is not None:
            try:
                hs = int(hs)
            except (TypeError, ValueError):
                hs = None
        if nhl is not None:
            try:
                nhl = int(nhl)
            except (TypeError, ValueError):
                nhl = None
        return hs, nhl
    except Exception:
        return None, None


def _check_merge_compatible(model_paths: list) -> tuple[bool, str, list]:
    """
    判断多模型是否可融合：要求同一架构，即 config 的 hidden_size 与 num_hidden_layers 均一致。
    返回 (compatible, reason, model_types)。model_types 仍返回各 model_type 供前端展示。
    """
    if not model_paths or len(model_paths) < 2:
        return True, "", []
    archs = []
    types = []
    for p in model_paths:
        hs, nhl = _get_model_arch(p)
        t = _get_model_type(p)
        types.append(t or "")
        if hs is None or nhl is None:
            return False, "无法读取模型 config 的 hidden_size/num_hidden_layers：%s" % (os.path.basename(p) if p else p), types
        archs.append((hs, nhl))
    if len(set(archs)) != 1:
        return False, "模型架构不一致（hidden_size 或 num_hidden_layers 不同），无法融合。请参见 DEVELOPMENT.md 的兼容性说明。", types
    return True, "", types


@app.route("/api/dataset/hf_info", methods=["POST"])
def api_dataset_hf_info():
    """
    根据 HuggingFace 数据集名称拉取信息并返回子集(config)与 split 列表，供完全融合界面选择。
    优先从本地已加载的数据集读取真实的 splits 和 configs，而不是使用固定字段。
    请求体: { "hf_dataset": "cais/mmlu" }。会触发缓存/下载到 HF_DATASETS_CACHE 或默认缓存。
    """
    data = request.json or {}
    hf_dataset = (data.get("hf_dataset") or "").strip()
    if not hf_dataset:
        return jsonify({"status": "error", "message": "hf_dataset 必填"}), 400
    try:
        from datasets import get_dataset_config_names, load_dataset_builder, load_dataset
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        import glob
        cache_dir = getattr(Config, "EVAL_HF_DATASETS_CACHE", None) or os.environ.get("HF_DATASETS_CACHE")
        
        configs = []
        splits = []

        def _load_ds():
            try:
                ds_local = load_dataset(hf_dataset, trust_remote_code=True, cache_dir=cache_dir)
                if ds_local and hasattr(ds_local, "keys") and callable(getattr(ds_local, "keys", None)):
                    return list(ds_local.keys())
            except Exception as e:
                _app_logger.debug("[hf_info] 本地数据集读取失败: %s", e)
            return []

        def _get_configs():
            try:
                return get_dataset_config_names(hf_dataset, trust_remote_code=True)
            except Exception as e:
                _app_logger.debug("[hf_info] 获取 configs 失败: %s", e)
                return []

        def _get_splits_from_builder(first_config):
            try:
                b = load_dataset_builder(hf_dataset, first_config or None, trust_remote_code=True, cache_dir=cache_dir)
                return list(b.info.splits.keys()) if b.info.splits else []
            except Exception as e:
                _app_logger.debug("[hf_info] builder 读取 splits 失败: %s", e)
                return []
        
        def _get_default_splits():
            return ["train", "validation", "test"]
        
        def _collect_local_infos(cache_root):
            if not cache_root or not os.path.isdir(cache_root):
                return {}
            base_name = hf_dataset.split("/")[-1]
            norm_name = hf_dataset.replace("/", "___")
            roots = [cache_root, os.path.join(cache_root, "datasets")]
            files = []
            for r in roots:
                if not r or not os.path.isdir(r):
                    continue
                files.extend(glob.glob(os.path.join(r, norm_name, "**", "dataset_info.json"), recursive=True))
                files.extend(glob.glob(os.path.join(r, base_name, "**", "dataset_info.json"), recursive=True))
                files.extend(glob.glob(os.path.join(r, norm_name, "**", "dataset_infos.json"), recursive=True))
                files.extend(glob.glob(os.path.join(r, base_name, "**", "dataset_infos.json"), recursive=True))
            def _parse_readme_dataset_info(text):
                lines = text.splitlines()
                if not lines:
                    return {}
                idx = 0
                if lines[0].strip() == "---":
                    idx = 1
                else:
                    return {}
                in_dataset_info = False
                in_splits = False
                infos = {}
                current_cfg = ""
                current_splits = []
                while idx < len(lines):
                    line = lines[idx]
                    if line.strip() == "---":
                        break
                    stripped = line.strip()
                    if stripped == "dataset_info:":
                        in_dataset_info = True
                        in_splits = False
                        idx += 1
                        continue
                    if not in_dataset_info:
                        idx += 1
                        continue
                    if stripped.startswith("- config_name:"):
                        if current_cfg:
                            infos[current_cfg] = current_splits
                        current_cfg = stripped.split(":", 1)[1].strip()
                        current_splits = []
                        in_splits = False
                        idx += 1
                        continue
                    if stripped == "splits:":
                        in_splits = True
                        idx += 1
                        continue
                    if in_splits and stripped.startswith("- name:"):
                        split_name = stripped.split(":", 1)[1].strip()
                        if split_name:
                            current_splits.append(split_name)
                        idx += 1
                        continue
                    if stripped.startswith("download_size:") or stripped.startswith("dataset_size:") or stripped.startswith("features:"):
                        in_splits = False
                    idx += 1
                if current_cfg:
                    infos[current_cfg] = current_splits
                return infos
            infos = {}
            for p in files:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    if isinstance(info, dict) and "splits" in info:
                        ds_name = (info.get("dataset_name") or "").strip().lower()
                        if ds_name and ds_name != base_name.lower():
                            continue
                        cfg = (info.get("config_name") or "").strip()
                        split_info = info.get("splits") or {}
                        split_keys = list(split_info.keys()) if isinstance(split_info, dict) else []
                        if cfg:
                            infos[cfg] = split_keys
                    else:
                        for cfg_key, cfg_info in (info or {}).items():
                            if not isinstance(cfg_info, dict):
                                continue
                            ds_name = (cfg_info.get("dataset_name") or "").strip().lower()
                            if ds_name and ds_name != base_name.lower():
                                continue
                            cfg = (cfg_info.get("config_name") or cfg_key or "").strip()
                            split_info = cfg_info.get("splits") or {}
                            split_keys = list(split_info.keys()) if isinstance(split_info, dict) else []
                            if cfg:
                                infos[cfg] = split_keys
                except Exception:
                    continue
            downloads_dir = os.path.join(cache_root, "downloads")
            if os.path.isdir(downloads_dir):
                meta_files = glob.glob(os.path.join(downloads_dir, "*.json"))
                for meta_path in meta_files:
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        url = (meta.get("url") or "").strip()
                        if not url:
                            continue
                        if ("/datasets/" + hf_dataset.strip("/") + "/") not in url:
                            continue
                        if not url.endswith("README.md"):
                            continue
                        content_path = meta_path[:-5]
                        if not os.path.isfile(content_path):
                            continue
                        with open(content_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        yaml_infos = _parse_readme_dataset_info(text)
                        for cfg, splits_list in (yaml_infos or {}).items():
                            if cfg and cfg not in infos:
                                infos[cfg] = splits_list or []
                    except Exception:
                        continue
            return infos

        def _collect_lm_task_infos():
            base_name = hf_dataset.split("/")[-1].lower()
            project_root = getattr(Config, "PROJECT_ROOT", None)
            if not project_root:
                return {}
            base_root = os.path.abspath(os.path.join(project_root, "..", ".."))
            lm_root = os.path.join(base_root, "Packages", "mergenetic", "lm_tasks")
            if not os.path.isdir(lm_root):
                return {}
            target_dir = os.path.join(lm_root, base_name)
            if not os.path.isdir(target_dir):
                return {}
            yaml_files = glob.glob(os.path.join(target_dir, "*.yaml")) + glob.glob(os.path.join(target_dir, "*.yml"))
            infos = {}
            for p in yaml_files:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    dataset_path = ""
                    dataset_name = ""
                    training_split = ""
                    test_split = ""
                    fewshot_split = ""
                    for raw in lines:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("dataset_path:"):
                            dataset_path = line.split(":", 1)[1].strip()
                            continue
                        if line.startswith("dataset_name:"):
                            dataset_name = line.split(":", 1)[1].strip()
                            continue
                        if line.startswith("training_split:"):
                            training_split = line.split(":", 1)[1].strip()
                            continue
                        if line.startswith("test_split:"):
                            test_split = line.split(":", 1)[1].strip()
                            continue
                        if line.startswith("fewshot_split:"):
                            fewshot_split = line.split(":", 1)[1].strip()
                            continue
                    if dataset_path and dataset_path.lower() != base_name:
                        continue
                    cfg = dataset_name or ""
                    splits = []
                    for s in [training_split, fewshot_split, test_split]:
                        if s and s not in splits:
                            splits.append(s)
                    if cfg and cfg not in infos:
                        infos[cfg] = splits
                except Exception:
                    continue
            return infos

        with ThreadPoolExecutor(max_workers=2) as ex:
            try:
                configs = ex.submit(_get_configs).result(timeout=5)
            except TimeoutError:
                _app_logger.warning("[hf_info] 获取 configs 超时 5s")
                configs = []
            try:
                splits = ex.submit(_load_ds).result(timeout=5)
            except TimeoutError:
                _app_logger.warning("[hf_info] 读取本地数据集 splits 超时 5s")
                splits = []

        local_infos = {}
        default_hf_cache = os.path.join(str(Path.home()), ".cache", "huggingface", "datasets")
        cache_candidates = [
            cache_dir,
            getattr(Config, "HF_DATASETS_CACHE", None),
            getattr(Config, "EVAL_HF_DATASETS_CACHE", None),
            default_hf_cache,
        ]
        for c in cache_candidates:
            local_infos.update(_collect_local_infos(c))
        lm_task_infos = _collect_lm_task_infos()
        for cfg, split_list in (lm_task_infos or {}).items():
            if cfg and cfg not in local_infos:
                local_infos[cfg] = split_list

        configs_set = set([c for c in (configs or []) if c])
        configs_set.update(local_infos.keys())
        configs = list(configs_set)
        configs.sort()

        requested_subset = (data.get("hf_subset") or "").strip()
        target_config = requested_subset if requested_subset and requested_subset in configs_set else (configs[0] if configs else None)
        
        if configs:
            splits = local_infos.get(target_config) or splits
            if not splits:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    try:
                        splits = ex.submit(_get_splits_from_builder, target_config).result(timeout=5)
                    except TimeoutError:
                        _app_logger.warning("[hf_info] builder 读取 splits 超时 5s")
                        splits = []
        else:
            if not splits:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    try:
                        splits = ex.submit(_get_splits_from_builder, None).result(timeout=5)
                    except TimeoutError:
                        _app_logger.warning("[hf_info] builder 读取 splits 超时 5s")
                        splits = []
        
        if not splits:
            splits = _get_default_splits()
            _app_logger.warning("[hf_info] 无法读取 splits，使用默认值: %s", splits)
        
        return jsonify({
            "status": "success",
            "hf_dataset": hf_dataset,
            "configs": configs or [],
            "splits": splits,
        })
    except Exception as e:
        _app_logger.exception("api_dataset_hf_info: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


def _get_recipe_arch_path(recipe_id: str) -> str | None:
    """
    获取配方的架构代表路径（父代模型之一）。
    配方物化后的模型架构与其父代模型一致，用第一个父代的路径来判断架构。
    """
    recipe_path = os.path.join(RECIPES_DIR, f"{recipe_id}.json")
    if not os.path.isfile(recipe_path):
        return None
    try:
        with open(recipe_path, "r", encoding="utf-8") as f:
            r = json.load(f)
        model_paths = r.get("model_paths") or []
        if model_paths and os.path.isdir(model_paths[0]):
            return model_paths[0]
    except Exception:
        pass
    return None


@app.route("/api/merge_evolutionary_check", methods=["POST"])
def api_merge_evolutionary_check():
    """
    检查所选模型是否可融合，用于前端在选择时弹窗提示。
    支持 model_paths（纯路径）或 items（包含 path/recipe 类型）。
    """
    data = request.json or {}
    items = data.get("items") or []
    paths = data.get("model_paths") or []

    # 如果是 items 格式（可能包含配方），转换为路径列表
    if items and len(items) >= 2:
        resolved = []
        for it in items:
            if it.get("type") == "path" and it.get("path"):
                ab = _resolve_model_path(it["path"])
                if not ab:
                    return jsonify({"status": "error", "message": "模型路径无效: %s" % it["path"]}), 400
                resolved.append(ab)
            elif it.get("type") == "recipe" and it.get("recipe_id"):
                arch_path = _get_recipe_arch_path(it["recipe_id"])
                if not arch_path:
                    return jsonify({"status": "error", "message": "配方无效或父代模型缺失: %s" % it["recipe_id"]}), 400
                resolved.append(arch_path)
            else:
                return jsonify({"status": "error", "message": "items 格式无效"}), 400
    elif len(paths) >= 2:
        resolved = []
        for p in paths:
            ab = _resolve_model_path(p)
            if not ab:
                return jsonify({"status": "error", "message": "模型路径无效: %s" % (p,)}), 400
            resolved.append(ab)
    else:
        return jsonify({"status": "success", "compatible": True, "reason": ""})

    compatible, reason, model_types = _check_merge_compatible(resolved)
    return jsonify({
        "status": "success",
        "compatible": compatible,
        "reason": reason,
        "model_types": model_types,
    })


RECIPES_DIR = getattr(Config, "RECIPES_DIR", None) or os.path.join(Config.PROJECT_ROOT, "recipes")


@app.route("/api/recipes")
def api_recipes_list():
    """列出所有已保存的完全融合配方，供前端查询。每个配方带 is_vlm 字段，根据父代模型判断。"""
    if not os.path.isdir(RECIPES_DIR):
        return jsonify({"status": "success", "recipes": []})
    recipes = []
    for f in os.listdir(RECIPES_DIR):
        if not f.endswith(".json"):
            continue
        path = os.path.join(RECIPES_DIR, f)
        try:
            with open(path, "r", encoding="utf-8") as fp:
                r = json.load(fp)
            r["recipe_id"] = os.path.splitext(f)[0]
            # 根据父代模型判断是否 VLM（任一父代是 VLM 则为 VLM）
            model_paths = r.get("model_paths") or []
            is_vlm = any(_model_is_vlm(p) for p in model_paths if p and os.path.isdir(p))
            r["is_vlm"] = is_vlm
            recipes.append(r)
        except Exception:
            continue
    recipes.sort(key=lambda x: (x.get("completed_at") or ""), reverse=True)
    return jsonify({"status": "success", "recipes": recipes})


@app.route("/api/recipes/<recipe_id>")
def api_recipe_get(recipe_id):
    """获取单个配方详情（含完整参数与 best_genotype）。"""
    path = os.path.join(RECIPES_DIR, "%s.json" % recipe_id)
    if not os.path.isfile(path):
        return jsonify({"status": "error", "message": "配方不存在"}), 404
    try:
        with open(path, "r", encoding="utf-8") as f:
            recipe = json.load(f)
        recipe["recipe_id"] = recipe_id
        return jsonify({"status": "success", "recipe": recipe})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/recipes/apply", methods=["POST"])
def api_recipes_apply():
    """根据配方直接融合出最终模型（固定 genotype，只跑一次合并）。"""
    data = request.json or {}
    recipe_id = (data.get("recipe_id") or "").strip()
    if not recipe_id:
        return jsonify({"status": "error", "message": "缺少 recipe_id"}), 400
    path = os.path.join(RECIPES_DIR, "%s.json" % recipe_id)
    if not os.path.isfile(path):
        return jsonify({"status": "error", "message": "配方不存在"}), 404

    task_id = str(uuid.uuid4())[:8]
    created_at = time.time()
    task_data = {
        "type": "recipe_apply",
        "task_id": task_id,
        "recipe_id": recipe_id,
        "custom_name": (data.get("custom_name") or "").strip() or None,
        "created_at": created_at,
    }
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
    _app_logger.info("[API] 提交配方应用 task_id=%s recipe_id=%s", task_id, recipe_id)
    return jsonify({"status": "success", "task_id": task_id})


@app.route("/api/search", methods=["GET"])
def api_search():
    """搜索功能：支持按任务ID、配方、模型名称搜索融合记录和配方。"""
    query = (request.args.get("q") or "").strip().lower()
    if not query:
        return jsonify({"status": "error", "message": "缺少搜索关键词"}), 400

    results = {"recipes": [], "tasks": [], "models": []}

    # 搜索配方
    if os.path.isdir(RECIPES_DIR):
        for f in os.listdir(RECIPES_DIR):
            if not f.endswith(".json"):
                continue
            recipe_id = os.path.splitext(f)[0]
            if query in recipe_id.lower():
                try:
                    path = os.path.join(RECIPES_DIR, f)
                    with open(path, "r", encoding="utf-8") as fp:
                        r = json.load(fp)
                    r["recipe_id"] = recipe_id
                    if query in (r.get("custom_name") or "").lower() or query in (r.get("task_id") or "").lower():
                        results["recipes"].append(r)
                except Exception:
                    continue

    # 搜索任务记录
    if os.path.isdir(MERGE_DIR):
        for tid in os.listdir(MERGE_DIR):
            if query not in tid.lower():
                continue
            meta_path = os.path.join(MERGE_DIR, tid, "metadata.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if query in (meta.get("custom_name") or "").lower() or query in tid.lower():
                    meta["task_id"] = tid
                    # 读取进度信息
                    progress_path = os.path.join(MERGE_DIR, tid, "progress.json")
                    if os.path.isfile(progress_path):
                        try:
                            with open(progress_path, "r", encoding="utf-8") as pf:
                                prog = json.load(pf)
                            meta["evolution_progress"] = {
                                "step": prog.get("step", 0),
                                "current_best": prog.get("current_best"),
                                "global_best": prog.get("global_best"),
                                "best_genotype": prog.get("best_genotype"),
                            }
                        except Exception:
                            pass
                    results["tasks"].append(meta)
            except Exception:
                continue

    # 搜索模型仓库（通过模型名称）
    try:
        repo_models = _model_repo_list()
        for m in repo_models:
            name = (m.get("name") or "").lower()
            if query in name:
                results["models"].append(m)
    except Exception:
        pass

    return jsonify({"status": "success", "query": query, "results": results})


@app.route("/api/fusion_history", methods=["GET"])
def api_fusion_history():
    """获取所有完全融合任务的历史记录，包含性能指标。"""
    history = []
    if not os.path.isdir(MERGE_DIR):
        return jsonify({"status": "success", "history": history})

    for tid in os.listdir(MERGE_DIR):
        meta_path = os.path.join(MERGE_DIR, tid, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # 只返回完全融合任务
            if meta.get("type") != "merge_evolutionary":
                continue
            meta["task_id"] = tid
            # 读取进度信息（包含 current_best, global_best）
            progress_path = os.path.join(MERGE_DIR, tid, "progress.json")
            evo_progress = {}
            if os.path.isfile(progress_path):
                try:
                    with open(progress_path, "r", encoding="utf-8") as pf:
                        prog = json.load(pf)
                    evo_progress = {
                        "step": prog.get("step", 0),
                        "current_best": prog.get("current_best"),
                        "global_best": prog.get("global_best"),
                        "best_genotype": prog.get("best_genotype"),
                    }
                    # 添加 ETA 和进度信息
                    if "eta_seconds" in prog:
                        evo_progress["eta_seconds"] = prog.get("eta_seconds")
                        evo_progress["estimated_completion"] = prog.get("estimated_completion")
                    if "current_step" in prog:
                        evo_progress["current_step"] = prog.get("current_step")
                        evo_progress["total_expected_steps"] = prog.get("total_expected_steps")
                except Exception:
                    pass
            # 优先从 fusion_info.json 或 recipe 读取准确率（如果 progress.json 中为0）
            recipe_path = os.path.join(RECIPES_DIR, "%s.json" % tid)
            if os.path.isfile(recipe_path):
                try:
                    with open(recipe_path, "r", encoding="utf-8") as rf:
                        recipe = json.load(rf)
                    # 如果 progress 中的准确率为0，尝试从 recipe 读取
                    if evo_progress.get("current_best") == 0.0 and recipe.get("final_test_acc") is not None:
                        evo_progress["current_best"] = recipe.get("final_test_acc")
                    if evo_progress.get("global_best") == 0.0 and recipe.get("final_test_acc") is not None:
                        evo_progress["global_best"] = recipe.get("final_test_acc")
                    if evo_progress.get("current_best") == 0.0 and recipe.get("current_best_acc") is not None:
                        evo_progress["current_best"] = recipe.get("current_best_acc")
                except Exception:
                    pass
            # 也检查 fusion_info.json（在命名目录中）
            for item in os.listdir(os.path.join(MERGE_DIR, tid)):
                fusion_info_path = os.path.join(MERGE_DIR, tid, item, "fusion_info.json")
                if os.path.isfile(fusion_info_path):
                    try:
                        with open(fusion_info_path, "r", encoding="utf-8") as ff:
                            fusion_info = json.load(ff)
                        if evo_progress.get("current_best") == 0.0 and fusion_info.get("final_test_acc") is not None:
                            evo_progress["current_best"] = fusion_info.get("final_test_acc")
                        if evo_progress.get("global_best") == 0.0 and fusion_info.get("final_test_acc") is not None:
                            evo_progress["global_best"] = fusion_info.get("final_test_acc")
                        break
                    except Exception:
                        pass
            meta["evolution_progress"] = evo_progress
            # 对于 recipe_apply 类型，使用 recipe_id 而不是 task_id 来检查配方文件
            if meta.get("type") == "recipe_apply" and meta.get("recipe_id"):
                recipe_path = os.path.join(RECIPES_DIR, "%s.json" % meta.get("recipe_id"))
            meta["has_recipe"] = os.path.isfile(recipe_path)
            # 添加 pop_size 和 n_iter 信息供前端计算迭代次数
            meta["pop_size"] = meta.get("pop_size")
            meta["n_iter"] = meta.get("n_iter")
            history.append(meta)
        except Exception:
            continue

    # 按创建时间倒序
    history.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"status": "success", "history": history})


@app.route("/fusion_history")
def fusion_history_page():
    """融合历史页面路由。"""
    return render_template("fusion_history.html")


if __name__ == "__main__":
    try:
        from app import app as modular_app
        app_to_run = modular_app
        print("--- 使用模块化应用入口 ---")
    except Exception:
        app_to_run = app
        print("--- 模块化入口不可用，回退到本文件内置应用 ---")
    port = int(os.environ.get("PORT", 5000))
    print("--- mergeKit_beta 后端启动 (port=%s) ---" % port)
    app_to_run.run(host="0.0.0.0", debug=True, port=port, use_reloader=False)
