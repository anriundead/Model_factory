import json
import os
import shutil
import time
import uuid

from flask import render_template, jsonify, request, send_from_directory

from .dataset_info import DataRequest


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


def resolve_hf_subsets(dataset_type: str, subset_group_or_single: str) -> list:
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


def register_routes(app, state, services, dataset_service):
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

    @app.route("/api/models")
    def get_models():
        try:
            seen_paths = set()
            models = []
            for base_path in (
                getattr(state.config, "LOCAL_MODELS_PATH", None),
                state.model_pool_path,
            ):
                if not base_path or not os.path.exists(base_path):
                    if base_path:
                        os.makedirs(base_path, exist_ok=True)
                    continue
                for m in services.list_models_from_dir(base_path):
                    path = (m.get("path") or "").strip()
                    if path and path not in seen_paths:
                        seen_paths.add(path)
                        m["is_vlm"] = services.model_is_vlm(path)
                        m["source"] = "base"
                        models.append(m)
            base_path = getattr(state.config, "LOCAL_MODELS_PATH", None) or state.model_pool_path
            return jsonify({"status": "success", "models": models, "base_path": base_path})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/models_pool")
    def get_models_pool():
        return get_models()

    @app.route("/api/merged_models")
    def get_merged_models():
        models = []
        if not os.path.exists(state.merge_dir):
            return jsonify({"status": "success", "models": models})
        for task_id in os.listdir(state.merge_dir):
            output_path = os.path.join(state.merge_dir, task_id, "output")
            meta_path = os.path.join(state.merge_dir, task_id, "metadata.json")
            if not os.path.isfile(meta_path) or not os.path.isdir(output_path):
                continue
            if not services.output_has_safetensors(output_path):
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
                    "is_vlm": services.model_is_vlm(path),
                })
            except Exception:
                continue
        return jsonify({"status": "success", "models": models})

    @app.route("/api/merge", methods=["POST"])
    def start_merge():
        data = request.json or {}
        priority_str = data.get("priority", "common")
        priority_score = state.priority_map.get(priority_str, 10)
        custom_name = (data.get("custom_name") or "").strip()
        if not custom_name:
            return jsonify({"status": "error", "message": "模型名称不能为空"}), 400
        if services.is_name_duplicate(custom_name):
            return jsonify({"status": "error", "message": "名称 '%s' 已存在。" % custom_name}), 409
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
                    path = services.resolve_model_path(it.get("path") or "")
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
            data["model_paths"] = None
        else:
            return jsonify({"status": "error", "message": "请提供 model_paths、models 或 items"}), 400
        task_id = str(uuid.uuid4())[:8]
        created_at = time.time()
        data["created_at"] = created_at
        data["type"] = "merge"
        state.logger.info(
            "[API] 提交标准融合 task_id=%s custom_name=%s model_paths=%s method=%s",
            task_id, custom_name, data.get("model_paths") or data.get("models"), data.get("method"),
        )
        with state.scheduler_lock:
            if state.running_task_info["id"] and priority_score < (state.running_task_info["priority"] or 99):
                services.interrupt_current_task(reason="被 VIP 任务抢占")
            state.tasks[task_id] = {
                "progress": 0,
                "message": "正在加入优先级队列...",
                "status": "queued",
                "created_at": created_at,
                "original_data": data,
                "priority": priority_str,
            }
            state.task_queue.put((priority_score, created_at, task_id, data))
        return jsonify({"status": "success", "task_id": task_id})

    @app.route("/api/merge_evolutionary", methods=["POST"])
    def start_merge_evolutionary():
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
                    path = services.resolve_model_path(it.get("path") or "")
                    if not path:
                        return jsonify({"status": "error", "message": "items[%d] 路径无效: %s" % (i, it.get("path", ""))}), 400
                    resolved.append({"type": "path", "path": path})
                else:
                    return jsonify({"status": "error", "message": "items[%d] 需为 type:path 或 type:recipe" % i}), 400
        else:
            if len(model_paths) < 2:
                return jsonify({"status": "error", "message": "至少需要 2 个模型（或提供 items 数组）"}), 400
            for p in model_paths:
                path = services.resolve_model_path(p)
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
            hf_subsets = resolve_hf_subsets(dataset_type, hf_subset_raw)
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
            "hf_subset_group": hf_subset_raw if hf_subset_raw in [g["id"] for g in (MMLU_SUBSET_GROUPS + CMMMU_SUBSET_GROUPS)] else "",
            "hf_split": data.get("hf_split", "test"),
            "hf_split_final": (data.get("hf_split_final") or "").strip() or None,
            "pop_size": max(2, min(128, int(data.get("pop_size", 20)))),
            "n_iter": max(1, min(50, int(data.get("n_iter", 15)))),
            "max_samples": max(4, min(512, int(data.get("max_samples", 64)))),
            "dtype": data.get("dtype", "bfloat16"),
            "ray_num_gpus": int(data.get("ray_num_gpus", 1)),
            "created_at": created_at,
        }
        merge_dir = os.path.join(state.merge_dir, task_id)
        os.makedirs(merge_dir, exist_ok=True)
        with open(os.path.join(merge_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"id": task_id, "status": "pending", **task_data}, f, ensure_ascii=False, indent=2)
        state.logger.info(
            "[API] 提交完全融合 task_id=%s custom_name=%s model_paths=%s hf_subsets=%s pop_size=%s n_iter=%s",
            task_id, task_data.get("custom_name"), resolved[:3], hf_subsets, task_data.get("pop_size"), task_data.get("n_iter"),
        )
        with state.scheduler_lock:
            state.tasks[task_id] = {
                "progress": 0,
                "message": "正在排队...",
                "status": "queued",
                "created_at": created_at,
                "original_data": task_data,
                "priority": data.get("priority", "common"),
            }
            state.task_queue.put((state.priority_map.get(data.get("priority", "common"), 10), created_at, task_id, task_data))
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
            for t in services.testset_list():
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
                    if hf_dataset and hf_subset:
                        lm_eval_task = services.infer_lm_eval_task(hf_dataset, hf_subset)
                    elif hf_dataset:
                        try:
                            import merge_manager as mm
                            extra = mm._load_eval_task_mapping()
                            hf_key = hf_dataset.strip().lower()
                            if hf_dataset in extra:
                                lm_eval_task = extra[hf_dataset]
                            elif hf_key in mm.HF_DATASET_TO_LM_EVAL_TASK:
                                lm_eval_task = mm.HF_DATASET_TO_LM_EVAL_TASK[hf_key]
                            if not lm_eval_task:
                                discovered_task = mm._auto_discover_lm_eval_task(hf_dataset, hf_subset)
                                if discovered_task:
                                    lm_eval_task = discovered_task
                                    exact_key = "%s|%s" % (hf_dataset.strip(), hf_subset.strip()) if hf_subset else hf_dataset.strip()
                                    extra[exact_key] = discovered_task
                                    if not hf_subset:
                                        extra[hf_dataset.strip()] = discovered_task
                                    mm._save_eval_task_mapping(extra)
                                    state.logger.info("[eval] 已自动发现并保存映射: %s -> %s", exact_key, discovered_task)
                        except Exception:
                            pass
                    if lm_eval_task:
                        testsets = services.load_testsets_dict()
                        for tid, t in list(testsets.items()):
                            if t.get("testset_id") == testset_id:
                                testsets[tid] = {**t, "lm_eval_task": lm_eval_task}
                                services.save_testsets_dict(testsets)
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
        state.logger.info(
            "[API] 提交评估任务 task_id=%s model_path=%s dataset=%s testset_id=%s hf_dataset=%s hf_subset=%s",
            task_id, model_path, task_data.get("dataset"), task_data.get("testset_id"),
            task_data.get("hf_dataset"), task_data.get("hf_subset"),
        )
        with state.scheduler_lock:
            state.tasks[task_id] = task_data
            state.task_queue.put((10, created_at, task_id, task_data))
        return jsonify({"status": "success", "task_id": task_id})

    @app.route("/api/history", methods=["GET"])
    def list_history():
        return jsonify({"status": "success", "history": services.get_all_history()})

    @app.route("/api/history/<task_id>", methods=["DELETE"])
    def delete_history(task_id):
        path = os.path.join(state.merge_dir, task_id)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                return jsonify({"status": "success"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500
        return jsonify({"status": "error", "message": "Not found"}), 404

    @app.route("/api/history/<task_id>", methods=["GET"])
    def get_history_detail(task_id):
        path = os.path.join(state.merge_dir, task_id, "metadata.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return jsonify({"status": "success", "data": json.load(f)})
        return jsonify({"status": "error", "message": "Not found"}), 404

    @app.route("/api/status/<task_id>")
    def get_status(task_id):
        task = state.tasks.get(task_id)
        if task is not None:
            resp = {k: v for k, v in task.items() if k not in ("control", "original_data")}
            if task.get("status") == "queued":
                my_priority = state.priority_map.get(task.get("priority", "common"), 10)
                pos = 0
                running = 1 if state.running_task_info["id"] else 0
                for tid, t in state.tasks.items():
                    if t.get("status") == "queued" and tid != task_id:
                        op = state.priority_map.get(t.get("priority", "common"), 10)
                        if op < my_priority or (op == my_priority and t.get("created_at", 0) < task.get("created_at", 0)):
                            pos += 1
                resp["queue_position"] = pos + running
            if task.get("original_data", {}).get("type") == "merge_evolutionary" or task.get("type") == "merge_evolutionary":
                evo = services.read_evolution_progress(task_id)
                if evo is not None:
                    resp["evolution_progress"] = evo
                od = task.get("original_data") or {}
                resp["original_data"] = {"n_iter": od.get("n_iter"), "pop_size": od.get("pop_size")}
            return jsonify(resp)
        disk = services.status_from_disk(task_id)
        if disk is not None:
            return jsonify(disk)
        return jsonify({"status": "error"}), 404

    @app.route("/api/stop/<task_id>", methods=["POST"])
    def stop_task(task_id):
        if task_id not in state.tasks:
            return jsonify({"status": "error", "message": "任务不存在"}), 404
        with state.scheduler_lock:
            state.tasks[task_id]["status"] = "stopped"
            state.tasks[task_id]["message"] = "任务已手动停止"
            if state.running_task_info["id"] == task_id:
                state.tasks[task_id].get("control", {})["aborted"] = True
                if state.running_task_info.get("process"):
                    services.kill_process_tree_by_pid(state.running_task_info["process"].pid)
                state.running_task_info["id"] = None
        return jsonify({"status": "success"})

    @app.route("/api/resume/<task_id>", methods=["POST"])
    def resume_task(task_id):
        if task_id not in state.tasks:
            return jsonify({"status": "error", "message": "不存在"}), 404
        with state.scheduler_lock:
            task = state.tasks[task_id]
            if task.get("status") != "interrupted":
                return jsonify({"status": "error", "message": "不可恢复"}), 400
            priority = state.priority_map.get(task.get("priority", "cutin"), 20)
            task["status"] = "queued"
            task["message"] = "已手动恢复..."
            state.task_queue.put((priority, task["created_at"], task_id, task["original_data"]))
        return jsonify({"status": "success"})

    @app.route("/api/model_repo/list")
    def api_model_repo_list():
        base = services.base_models_list()
        for b in base:
            b["is_base"] = True
        merged = services.model_repo_list()
        return jsonify({
            "status": "success",
            "base_models": base,
            "merged_models": merged,
            "models": merged,
        })

    @app.route("/api/model_repo/<model_id>/path")
    def api_model_repo_path(model_id):
        path = services.model_repo_path(model_id)
        if path and os.path.isdir(path):
            return jsonify({"status": "success", "path": path})
        return jsonify({"status": "error", "message": "Not found"}), 404

    @app.route("/api/model_repo/<model_id>", methods=["DELETE"])
    def api_model_repo_delete(model_id):
        model_id = (model_id or "").strip()
        if not model_id:
            return jsonify({"status": "error", "message": "模型 ID 无效"}), 400
        models = services.model_repo_load_raw()
        if not isinstance(models, dict):
            return jsonify({"status": "error", "message": "仓库数据异常"}), 500
        m = models.get(model_id)
        if not m:
            return jsonify({"status": "error", "message": "模型不存在"}), 404
        path = m.get("path")
        if path and os.path.isdir(path):
            try:
                shutil.rmtree(path)
                state.logger.info("[model_repo] 已删除磁盘目录: %s", path)
            except Exception as e:
                state.logger.exception("[model_repo] 删除目录失败: %s", path)
                return jsonify({"status": "error", "message": "删除磁盘文件失败: %s" % str(e)}), 500
        try:
            del models[model_id]
            services.model_repo_save_raw(models)
        except Exception as e:
            state.logger.exception("[model_repo] 保存仓库失败")
            return jsonify({"status": "error", "message": "更新仓库失败: %s" % str(e)}), 500
        return jsonify({"status": "success"})

    @app.route("/api/resolve_model_path")
    def api_resolve_model_path():
        name_or_path = request.args.get("name") or request.args.get("path") or ""
        path = services.resolve_model_path(name_or_path)
        if path:
            return jsonify({"status": "success", "path": path})
        return jsonify({"status": "error", "message": "未找到对应模型目录"}), 404

    @app.route("/api/testset/list")
    def api_testset_list():
        refresh = request.args.get("refresh", "0") in ("1", "true", "yes")
        return jsonify({"status": "success", "testsets": services.testset_list(refresh=refresh)})

    @app.route("/api/testset/search")
    def api_testset_search():
        q = (request.args.get("q") or "").strip().lower()
        limit = int(request.args.get("limit", 20))
        refresh = request.args.get("refresh", "0") in ("1", "true", "yes")
        all_list = services.testset_list(refresh=refresh)
        if q:
            all_list = [t for t in all_list if q in (t.get("name") or "").lower() or q in (t.get("hf_dataset") or "").lower()]
        return jsonify({"status": "success", "results": all_list[:limit], "total": len(all_list)})

    @app.route("/api/hf/datasets/search", methods=["GET", "POST"])
    def api_hf_datasets_search():
        q = (request.args.get("q") or (request.get_json(silent=True) or {}).get("q") or "").strip()
        limit = int(request.args.get("limit") or (request.get_json(silent=True) or {}).get("limit", 20) or 20)
        if not q:
            return jsonify({"status": "error", "message": "q 必填"}), 400
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            endpoint = getattr(state.config, "HF_ENDPOINT", None) or os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
            ds_list = list(api.list_datasets(search=q, limit=min(limit, 50)))
            out = [{"id": d.id, "author": getattr(d, "author", None), "downloads": getattr(d, "downloads", None)} for d in ds_list]
            return jsonify({"status": "success", "results": out})
        except Exception as e:
            state.logger.exception("api_hf_datasets_search: %s", e)
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/testset/create", methods=["POST"])
    def api_testset_create():
        data = request.json or {}
        name = (data.get("name") or "").strip() or (data.get("dataset_name") or "").strip()
        hf_dataset = (data.get("hf_dataset") or "").strip()
        if not hf_dataset:
            return jsonify({"status": "error", "message": "hf_dataset 必填"}), 400
        if not name:
            name = hf_dataset
        hf_subset_raw = (data.get("hf_subset") or "").strip()
        hf_subset = hf_subset_raw if hf_subset_raw else None
        hf_split_raw = (data.get("hf_split") or "").strip()
        hf_split = hf_split_raw if hf_split_raw else None
        cache_dir = getattr(state.config, "HF_DATASETS_CACHE", None) or os.environ.get("HF_DATASETS_CACHE")
        split_candidates = []
        if hf_split:
            split_candidates.append(hf_split.split("[")[0].strip())
        split_candidates.extend(["test", "validation", "dev", "val", "train"])
        split_candidates = list(dict.fromkeys(split_candidates))
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
                    if ds:
                        try:
                            n = 0
                            if hasattr(ds, "__getitem__") and hasattr(ds, "keys"):
                                try:
                                    keys_list = list(ds.keys())
                                    if split_name in keys_list:
                                        n = len(ds[split_name]) if hasattr(ds[split_name], "__len__") else 0
                                    elif len(keys_list) > 0:
                                        first_key = keys_list[0]
                                        n = len(ds[first_key]) if hasattr(ds[first_key], "__len__") else 0
                                        if n > 0:
                                            actual_split = first_key
                                except (KeyError, AttributeError, TypeError) as dict_err:
                                    state.logger.debug("testset create 访问 DatasetDict 失败: %s", dict_err)
                            if n == 0 and hasattr(ds, "__len__"):
                                try:
                                    n = len(ds)
                                except Exception as len_err:
                                    state.logger.debug("testset create 获取 Dataset 长度失败: %s", len_err)
                            if n > 0:
                                if not actual_split:
                                    actual_split = split_name
                                break
                        except Exception as parse_err:
                            state.logger.debug("testset create 解析数据集失败: %s", parse_err)
                            continue
                except Exception as split_err:
                    load_error = str(split_err)
                    state.logger.debug("testset create 尝试 split=%s 失败: %s", split_name, split_err)
                    continue
            if n == 0:
                state.logger.warning("testset create 所有 split 均失败: hf_dataset=%s hf_subset=%s splits=%s error=%s", hf_dataset, hf_subset, split_candidates, load_error)
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
                        state.logger.debug("testset create 回退加载失败: %s", fallback_err)
        except Exception as e:
            load_error = str(e)
            state.logger.warning("testset create load_dataset 异常: %s", e)
        if n == 0:
            meta_n, meta_split, meta_subset = services.resolve_dataset_sample_count(hf_dataset, hf_subset, hf_split, cache_dir)
            if meta_n > 0:
                n = meta_n
                if not actual_split:
                    actual_split = meta_split
                if not hf_subset and meta_subset:
                    hf_subset = meta_subset
        hf_split = actual_split or (split_candidates[0] if split_candidates else "test")
        testset_id = str(uuid.uuid4())
        prompt_path = os.path.join(state.project_root, "yaml_template", "prompt.yaml")
        if not os.path.isfile(prompt_path):
            prompt_path = ""
        lm_eval_task = services.infer_lm_eval_task(hf_dataset, hf_subset)
        if not lm_eval_task and hf_dataset:
            try:
                import merge_manager as mm
                discovered_task = mm._auto_discover_lm_eval_task(hf_dataset, hf_subset)
                if discovered_task:
                    lm_eval_task = discovered_task
                    try:
                        extra = mm._load_eval_task_mapping()
                        hf_key = hf_dataset.strip()
                        subset_key = hf_subset.strip() if hf_subset else None
                        exact_key = "%s|%s" % (hf_key, subset_key) if subset_key else hf_key
                        extra[exact_key] = discovered_task
                        if not subset_key:
                            extra[hf_key] = discovered_task
                        mm._save_eval_task_mapping(extra)
                        state.logger.info("[testset] 已自动发现并保存映射: %s -> %s", exact_key, discovered_task)
                    except Exception as save_err:
                        state.logger.warning("[testset] 自动保存映射失败: %s", save_err)
            except Exception as discover_err:
                state.logger.debug("[testset] 自动发现任务失败: %s", discover_err)
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
                        state.logger.warning("[testset] 自动保存映射失败: %s", save_err)
            except Exception as e:
                state.logger.debug("[testset] 任务精确匹配失败: %s", e)

        entry = {
            "testset_id": testset_id,
            "name": name,
            "hf_dataset": hf_dataset,
            "hf_subset": hf_subset,
            "hf_split": hf_split,
            "lm_eval_task": lm_eval_task or "",
            "yaml_template_path": prompt_path,
            "sample_count": n,
            "created_at": time.time(),
            "created_by": "user",
            "notes": "从 HuggingFace 添加: %s" % hf_dataset,
            "question_type": "未知",
        }
        testsets = services.load_testsets_dict()
        testsets[testset_id] = entry
        services.save_testsets_dict(testsets)
        return jsonify({"status": "success", "testset_id": testset_id, "testset": entry})

    @app.route("/api/testset/<testset_id>")
    def api_testset_get(testset_id):
        refresh = request.args.get("refresh", "0") in ("1", "true", "yes")
        testset = services.get_testset_by_id(testset_id, refresh=refresh)
        if not testset:
            return jsonify({"status": "error", "message": "Not found"}), 404
        leaderboards = services.load_leaderboard()
        lb = leaderboards.get(testset_id, [])
        lb_norm = []
        for item in lb:
            acc = item.get("accuracy", None)
            acc_val = None
            try:
                if acc is not None:
                    acc_val = float(acc)
            except Exception:
                acc_val = None
            if acc_val is not None and acc_val <= 1:
                acc_val = acc_val * 100
            new_item = dict(item)
            if acc_val is not None:
                new_item["accuracy"] = round(acc_val, 4)
            lb_norm.append(new_item)
        lb_sorted = sorted(lb_norm, key=lambda x: x.get("accuracy", 0) or 0, reverse=True)
        return jsonify({
            "status": "success",
            "testset": testset,
            "leaderboard": lb_sorted,
        })

    @app.route("/api/test_history")
    def api_test_history():
        return jsonify({"status": "success", "history": services.get_all_eval_history()})

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

    @app.route("/api/model_is_vlm")
    def api_model_is_vlm():
        path = request.args.get("path") or (request.json or {}).get("path") if request.is_json else None
        if not path or not os.path.isdir(path):
            return jsonify({"status": "error", "message": "path 无效"}), 400
        return jsonify({"status": "success", "is_vlm": services.model_is_vlm(path)})

    @app.route("/api/dataset/hf_info", methods=["POST"])
    def api_dataset_hf_info():
        data = request.json or {}
        hf_dataset = (data.get("hf_dataset") or "").strip()
        if not hf_dataset:
            return jsonify({"status": "error", "message": "hf_dataset 必填"}), 400
        info = dataset_service.get_info(DataRequest(hf_dataset=hf_dataset, hf_subset=(data.get("hf_subset") or "").strip() or None))
        if info.get("status") != "success":
            return jsonify(info), 500
        return jsonify(info)

    @app.route("/api/merge_evolutionary_check", methods=["POST"])
    def api_merge_evolutionary_check():
        data = request.json or {}
        items = data.get("items") or []
        paths = data.get("model_paths") or []

        if items and len(items) >= 2:
            resolved = []
            for it in items:
                if it.get("type") == "path" and it.get("path"):
                    ab = services.resolve_model_path(it["path"])
                    if not ab:
                        return jsonify({"status": "error", "message": "模型路径无效: %s" % it["path"]}), 400
                    resolved.append(ab)
                elif it.get("type") == "recipe" and it.get("recipe_id"):
                    arch_path = services.get_recipe_arch_path(it["recipe_id"])
                    if not arch_path:
                        return jsonify({"status": "error", "message": "配方无效或父代模型缺失: %s" % it["recipe_id"]}), 400
                    resolved.append(arch_path)
                else:
                    return jsonify({"status": "error", "message": "items 格式无效"}), 400
        elif len(paths) >= 2:
            resolved = []
            for p in paths:
                ab = services.resolve_model_path(p)
                if not ab:
                    return jsonify({"status": "error", "message": "模型路径无效: %s" % (p,)}), 400
                resolved.append(ab)
        else:
            return jsonify({"status": "success", "compatible": True, "reason": ""})

        compatible, reason, model_types = services.check_merge_compatible(resolved)
        return jsonify({
            "status": "success",
            "compatible": compatible,
            "reason": reason,
            "model_types": model_types,
        })

    @app.route("/api/recipes")
    def api_recipes_list():
        if not os.path.isdir(state.recipes_dir):
            return jsonify({"status": "success", "recipes": []})
        recipes = []
        for f in os.listdir(state.recipes_dir):
            if not f.endswith(".json"):
                continue
            path = os.path.join(state.recipes_dir, f)
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    r = json.load(fp)
                r["recipe_id"] = os.path.splitext(f)[0]
                model_paths = r.get("model_paths") or []
                is_vlm = any(services.model_is_vlm(p) for p in model_paths if p and os.path.isdir(p))
                r["is_vlm"] = is_vlm
                recipes.append(r)
            except Exception:
                continue
        recipes.sort(key=lambda x: (x.get("completed_at") or ""), reverse=True)
        return jsonify({"status": "success", "recipes": recipes})

    @app.route("/api/recipes/<recipe_id>")
    def api_recipe_get(recipe_id):
        path = os.path.join(state.recipes_dir, "%s.json" % recipe_id)
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
        data = request.json or {}
        recipe_id = (data.get("recipe_id") or "").strip()
        if not recipe_id:
            return jsonify({"status": "error", "message": "缺少 recipe_id"}), 400
        path = os.path.join(state.recipes_dir, "%s.json" % recipe_id)
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
        with state.scheduler_lock:
            state.tasks[task_id] = {
                "progress": 0,
                "message": "正在排队...",
                "status": "queued",
                "created_at": created_at,
                "original_data": task_data,
                "priority": data.get("priority", "common"),
            }
            state.task_queue.put((state.priority_map.get(data.get("priority", "common"), 10), created_at, task_id, task_data))
        state.logger.info("[API] 提交配方应用 task_id=%s recipe_id=%s", task_id, recipe_id)
        return jsonify({"status": "success", "task_id": task_id})

    @app.route("/api/search", methods=["GET"])
    def api_search():
        query = (request.args.get("q") or "").strip().lower()
        if not query:
            return jsonify({"status": "error", "message": "缺少搜索关键词"}), 400

        results = {"recipes": [], "tasks": [], "models": []}

        if os.path.isdir(state.recipes_dir):
            for f in os.listdir(state.recipes_dir):
                if not f.endswith(".json"):
                    continue
                recipe_id = os.path.splitext(f)[0]
                if query in recipe_id.lower():
                    try:
                        path = os.path.join(state.recipes_dir, f)
                        with open(path, "r", encoding="utf-8") as fp:
                            r = json.load(fp)
                        r["recipe_id"] = recipe_id
                        if query in (r.get("custom_name") or "").lower() or query in (r.get("task_id") or "").lower():
                            results["recipes"].append(r)
                    except Exception:
                        continue

        if os.path.isdir(state.merge_dir):
            for tid in os.listdir(state.merge_dir):
                if query not in tid.lower():
                    continue
                meta_path = os.path.join(state.merge_dir, tid, "metadata.json")
                if not os.path.isfile(meta_path):
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if query in (meta.get("custom_name") or "").lower() or query in tid.lower():
                        meta["task_id"] = tid
                        progress_path = os.path.join(state.merge_dir, tid, "progress.json")
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

        try:
            repo_models = services.model_repo_list()
            for m in repo_models:
                name = (m.get("name") or "").lower()
                if query in name:
                    results["models"].append(m)
        except Exception:
            pass

        return jsonify({"status": "success", "query": query, "results": results})

    @app.route("/api/fusion_history", methods=["GET"])
    def api_fusion_history():
        history = []
        if not os.path.isdir(state.merge_dir):
            return jsonify({"status": "success", "history": history})

        for tid in os.listdir(state.merge_dir):
            meta_path = os.path.join(state.merge_dir, tid, "metadata.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("type") != "merge_evolutionary":
                    continue
                meta["task_id"] = tid
                progress_path = os.path.join(state.merge_dir, tid, "progress.json")
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
                        if "eta_seconds" in prog:
                            evo_progress["eta_seconds"] = prog.get("eta_seconds")
                            evo_progress["estimated_completion"] = prog.get("estimated_completion")
                        if "current_step" in prog:
                            evo_progress["current_step"] = prog.get("current_step")
                            evo_progress["total_expected_steps"] = prog.get("total_expected_steps")
                    except Exception:
                        pass
                recipe_path = os.path.join(state.recipes_dir, "%s.json" % tid)
                if os.path.isfile(recipe_path):
                    try:
                        with open(recipe_path, "r", encoding="utf-8") as rf:
                            recipe = json.load(rf)
                        if evo_progress.get("current_best") == 0.0 and recipe.get("final_test_acc") is not None:
                            evo_progress["current_best"] = recipe.get("final_test_acc")
                        if evo_progress.get("global_best") == 0.0 and recipe.get("final_test_acc") is not None:
                            evo_progress["global_best"] = recipe.get("final_test_acc")
                        if evo_progress.get("current_best") == 0.0 and recipe.get("current_best_acc") is not None:
                            evo_progress["current_best"] = recipe.get("current_best_acc")
                    except Exception:
                        pass
                for item in os.listdir(os.path.join(state.merge_dir, tid)):
                    fusion_info_path = os.path.join(state.merge_dir, tid, item, "fusion_info.json")
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
                if meta.get("type") == "recipe_apply" and meta.get("recipe_id"):
                    recipe_path = os.path.join(state.recipes_dir, "%s.json" % meta.get("recipe_id"))
                meta["has_recipe"] = os.path.isfile(recipe_path)
                meta["pop_size"] = meta.get("pop_size")
                meta["n_iter"] = meta.get("n_iter")
                history.append(meta)
            except Exception:
                continue

        history.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return jsonify({"status": "success", "history": history})

    @app.route("/fusion_history")
    def fusion_history_page():
        return render_template("fusion_history.html")
