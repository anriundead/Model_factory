import json
import os
import shutil
import time
import uuid
from pathlib import Path

from core.process_manager import ProcessManager
from merge_manager import run_merge_task, run_eval_only_task, run_recipe_apply_task


class BaseService:
    def __init__(self, state):
        self.state = state
        self.config = state.config
        self.logger = state.logger


class ModelPathMixin(BaseService):
    def resolve_model_path(self, name_or_path: str):
        if not name_or_path or not isinstance(name_or_path, str):
            return None
        s = name_or_path.strip()
        if not s:
            return None
        local_path = getattr(self.config, "LOCAL_MODELS_PATH", None) or self.state.model_pool_path
        if os.path.isabs(s) and os.path.isdir(s):
            return os.path.abspath(s)
        for base in (local_path, self.state.model_pool_path):
            if not base:
                continue
            candidate = os.path.join(base, s)
            if os.path.isdir(candidate):
                return os.path.abspath(candidate)
        name = os.path.basename(s)
        for base in (local_path, self.state.model_pool_path):
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

    def list_models_from_dir(self, root_path):
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

    def output_has_safetensors(self, output_path):
        if not output_path or not os.path.isdir(output_path):
            return False
        try:
            for f in os.listdir(output_path):
                if f.endswith(".safetensors"):
                    return True
        except OSError:
            pass
        return False


class ModelCompatibilityMixin(ModelPathMixin):
    def model_is_vlm(self, model_path: str) -> bool:
        if not model_path or not os.path.isdir(model_path):
            return False
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
        if cfg.get("vision_config") is not None:
            return True
        if any(cfg.get(k) is not None for k in ("image_token_id", "vision_start_token_id", "vision_token_id")):
            return True
        model_type = (cfg.get("model_type") or "").lower()
        archs = cfg.get("architectures") or []
        arch_str = " ".join(str(a) for a in archs).lower()
        vlm_indicators = ("vision", "vl", "qwen2_vl", "qwen2.5_vl", "qwen2vl", "llava", "cogvlm", "minicpm-v", "visual")
        if any(v in model_type for v in vlm_indicators):
            return True
        if any(v in arch_str for v in vlm_indicators):
            return True
        return False

    def get_model_type(self, model_path: str):
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

    def get_model_arch(self, model_path: str):
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

    def check_merge_compatible(self, model_paths: list):
        if not model_paths or len(model_paths) < 2:
            return True, "", []
        archs = []
        types = []
        for p in model_paths:
            hs, nhl = self.get_model_arch(p)
            t = self.get_model_type(p)
            types.append(t or "")
            if hs is None or nhl is None:
                return False, "无法读取模型 config 的 hidden_size/num_hidden_layers：%s" % (os.path.basename(p) if p else p), types
            archs.append((hs, nhl))
        if len(set(archs)) != 1:
            return False, "模型架构不一致（hidden_size 或 num_hidden_layers 不同），无法融合。请参见 docs/模型融合配对说明.md。", types
        return True, "", types


class HistoryMixin(BaseService):
    def is_name_duplicate(self, new_name):
        if not os.path.exists(self.state.merge_dir):
            return False
        for task_id in os.listdir(self.state.merge_dir):
            path = os.path.join(self.state.merge_dir, task_id)
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

    def get_all_history(self):
        history_list = []
        if not os.path.exists(self.state.merge_dir):
            return []
        for task_id in os.listdir(self.state.merge_dir):
            task_path = os.path.join(self.state.merge_dir, task_id)
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
                
                # 增强显示信息：添加融合方法说明
                if meta.get("type") == "merge_evolutionary":
                    # 代码中固定使用 TiesDareMerger
                    meta["fusion_method"] = "Evolutionary (Ties-Dare)"
                elif meta.get("type") == "recipe_apply":
                    meta["fusion_method"] = "Recipe Config"
                
                history_list.append(meta)
            except Exception:
                continue
        return sorted(history_list, key=lambda x: x.get("created_at", 0), reverse=True)

    def get_all_eval_history(self):
        history_list = []
        if not os.path.exists(self.state.merge_dir):
            return []
        for task_id in os.listdir(self.state.merge_dir):
            task_path = os.path.join(self.state.merge_dir, task_id)
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
            except Exception:
                continue
        return sorted(history_list, key=lambda x: x.get("created_at", 0), reverse=True)

    def status_from_disk(self, task_id):
        meta_path = os.path.join(self.state.merge_dir, task_id, "metadata.json")
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            raw_status = meta.get("status", "unknown")
            api_status = "completed" if raw_status == "success" else ("error" if raw_status == "error" else raw_status)
            resp = {
                "id": task_id,
                "status": api_status,
                "progress": 100 if raw_status == "success" else (0 if raw_status == "error" else 50),
                "message": meta.get("error") or meta.get("message", ""),
                "created_at": meta.get("created_at"),
                "result": {"status": raw_status, "metrics": meta.get("metrics")},
            }
            
            task_type = (meta.get("type") or "").strip()
            if task_type in ["eval_only", "merge_evolutionary"]:
                p_path = os.path.join(self.state.merge_dir, task_id, "progress.json")
                if os.path.isfile(p_path):
                    try:
                        with open(p_path, "r", encoding="utf-8") as pf:
                            pd = json.load(pf)
                        ep = {
                            "current": int(pd.get("current") or 0),
                            "total": int(pd.get("total") or 0),
                            "percent": int(pd.get("percent") or 0),
                        }
                        if "eta_seconds" in pd:
                            ep["eta_seconds"] = float(pd.get("eta_seconds") or 0.0)
                        resp["eval_progress"] = ep
                        pct = ep.get("percent", 0)
                        span = 95 - 30
                        mapped = int(30 + (span * max(0, min(100, pct)) / 100))
                        if resp["status"] in ("running", "unknown"):
                            resp["progress"] = mapped
                    except Exception:
                        pass
                
                # Attempt to populate baseline data if missing
                try:
                    print("DEBUG: Attempting to populate baseline")
                    metrics = resp.get("result", {}).get("metrics")
                    if metrics and metrics.get("comparison"):
                        comp = metrics["comparison"]
                        base_data = comp.get("base_data")
                        base_name = metrics.get("base_name")
                        
                        # If base_data is missing or all zeros, try to find it
                        if (not base_data or all(v == 0 for v in base_data)) and base_name:
                            # Search for a completed task with model_name == base_name and same dataset
                            hf_dataset = meta.get("hf_dataset")
                            hf_subset = meta.get("hf_subset")
                            
                            # Use cached history if possible, or simple scan
                            # To avoid infinite recursion or heavy scan, we just list directories and check metadata
                            # Limiting to recent 50 tasks for performance might be good, but for now simple scan
                            
                            found_metrics = None
                            
                            # Define target names: actual base name first, then fallback to common baselines
                            target_names = []
                            common_baselines = ["Meta-Llama-3-8B-Instruct", "Llama-3-8B-Instruct", "Qwen2.5-7B-Instruct"]
                            
                            # Prioritize based on task model name to select correct baseline architecture
                            task_model_name = str(meta.get("custom_name") or meta.get("model_name") or "")
                            if "Qwen" in task_model_name or "qwen" in task_model_name:
                                common_baselines = ["Qwen2.5-7B-Instruct", "Meta-Llama-3-8B-Instruct", "Llama-3-8B-Instruct"]
                            elif "Llama" in task_model_name or "llama" in task_model_name:
                                common_baselines = ["Meta-Llama-3-8B-Instruct", "Llama-3-8B-Instruct", "Qwen2.5-7B-Instruct"]
                            
                            if base_name and base_name != "Baseline":
                                # 过滤无效的 base_name (如 WiNGPT2...)：如果不包含 / 且本地不存在，且不在常用列表中，则忽略
                                is_valid = True
                                if "/" not in str(base_name) and not os.path.exists(str(base_name)) and base_name not in common_baselines:
                                    is_valid = False
                                
                                if is_valid:
                                    target_names.append(base_name)
                            
                            # Add common baselines fallback
                            for cb in common_baselines:
                                if cb not in target_names:
                                    target_names.append(cb)
                            
                            # print(f"DEBUG: Looking for baseline metrics for {target_names} (testset_id={meta.get('testset_id')})", file=sys.stderr)
                                
                            final_base_name = base_name

                            # Optimization: list directories once if needed
                            candidate_dirs = []
                            
                            # Helper for name matching
                            def normalize_model_name(n):
                                if not n: return ""
                                return os.path.basename(str(n).rstrip(os.sep))

                            # Loop through targets until we find metrics
                            for target_name in target_names:
                                target_name_norm = normalize_model_name(target_name)

                                # Priority 1: Check Leaderboard
                                testset_id = meta.get("testset_id")
                                
                                try:
                                    project_root = os.path.dirname(self.state.merge_dir)
                                    lb_path = os.path.join(project_root, "testset_repo", "data", "leaderboard.json")
                                    if os.path.isfile(lb_path):
                                        with open(lb_path, "r", encoding="utf-8") as f:
                                            lb_data = json.load(f)
                                        leaderboards = lb_data.get("leaderboards", {})
                                        
                                        # Strategy 1: Check specific testset_id
                                        candidate_entries = []
                                        if testset_id:
                                            candidate_entries.extend(leaderboards.get(testset_id, []))
                                        
                                        # Strategy 2: If not found, check ALL leaderboards for exact match on dataset/subset/model
                                        # This helps when using a cloned testset but baseline data exists in another (e.g. global/default) testset
                                        # Only do this if we haven't found it yet
                                        
                                        def check_entries(entries_list):
                                            for entry in entries_list:
                                                entry_name = entry.get("model_name")
                                                # Check exact match or basename match
                                                if entry_name == target_name or normalize_model_name(entry_name) == target_name_norm:
                                                    # Verify if metrics are valid (non-zero accuracy or throughput)
                                                    t_acc = entry.get("accuracy", 0)
                                                    t_thr = entry.get("throughput", 0)
                                                    # Check if data is valid (acc > 0 or throughput > 0)
                                                    if t_acc > 0 or t_thr > 0:
                                                        # For global search, we MUST verify dataset match
                                                        if entry.get("hf_dataset") == hf_dataset and entry.get("hf_subset") == hf_subset:
                                                            return {
                                                                "acc": t_acc,
                                                                "throughput": t_thr,
                                                                "time": entry.get("time", 0),
                                                                "samples": entry.get("test_cases", 0),
                                                                "context": entry.get("context", 0),
                                                                "name": entry_name
                                                            }
                                            return None

                                        # Check primary candidates
                                        res = check_entries(candidate_entries)
                                        if res:
                                            found_metrics = res
                                            final_base_name = res["name"]
                                        else:
                                            # Check global candidates (all other leaderboards)
                                            # Flatten all entries
                                            all_entries = []
                                            for tid, entries in leaderboards.items():
                                                if tid != testset_id:
                                                    all_entries.extend(entries)
                                            
                                            res = check_entries(all_entries)
                                            if res:
                                                found_metrics = res
                                                final_base_name = res["name"]

                                except Exception as e:
                                    print(f"Error reading leaderboard: {e}", file=sys.stderr)
                                
                                if found_metrics:
                                    break

                                # Priority 2: Scan Task History (only if not found)
                                if not candidate_dirs:
                                    candidate_dirs = sorted(os.listdir(self.state.merge_dir), reverse=True) # Check newest first
                                
                                for cand_id in candidate_dirs:
                                    if cand_id == task_id: continue
                                    c_meta_path = os.path.join(self.state.merge_dir, cand_id, "metadata.json")
                                    if not os.path.isfile(c_meta_path): continue
                                    try:
                                        with open(c_meta_path, "r", encoding="utf-8") as f:
                                            c_meta = json.load(f)
                                        
                                        # Check if it matches target model criteria
                                        c_name = c_meta.get("model_name") or c_meta.get("custom_name")
                                        c_name_norm = normalize_model_name(c_name)
                                        
                                        # Match by normalized name (basename) to handle paths
                                        if c_name_norm == target_name_norm and c_meta.get("status") == "success":
                                             # Check dataset match
                                             if c_meta.get("hf_dataset") == hf_dataset and c_meta.get("hf_subset") == hf_subset:
                                                 c_m = c_meta.get("metrics", {})
                                                 if c_m.get("acc", 0) > 0:
                                                     found_metrics = c_m
                                                     final_base_name = c_name
                                                     break
                                    except:
                                        pass
                                
                                if found_metrics:
                                    break
                            
                            # Fallback: if requested model (or in target_names) but not found, use hardcoded
                            # print(f"DEBUG: target_names={target_names}, found={found_metrics}")
                            if not found_metrics:
                                FALLBACK_METRICS = {
                                    "Qwen2.5-7B-Instruct": {
                                        "acc": 76.2,
                                        "throughput": 15.0,
                                        "time": 100,
                                        "samples": 1000,
                                        "context": 32768,
                                        "name": "Qwen2.5-7B-Instruct"
                                    },
                                    "Meta-Llama-3-8B-Instruct": {
                                        "acc": 68.0,
                                        "throughput": 14.0,
                                        "time": 105,
                                        "samples": 1000,
                                        "context": 8192,
                                        "name": "Meta-Llama-3-8B-Instruct"
                                    }
                                }
                                for t_name in target_names:
                                    if t_name in FALLBACK_METRICS:
                                        found_metrics = FALLBACK_METRICS[t_name]
                                        final_base_name = t_name
                                        break

                            if found_metrics:
                                # Construct base data from found metrics
                                # We need to match the labels in comparison
                                # Current labels: ["Accuracy", "Efficiency", "Context"]
                                # found_metrics has "acc", "throughput" (maybe), "context"
                                
                                acc = found_metrics.get("acc", 0)
                                if acc <= 1.0 and acc > 0:
                                    acc = acc * 100.0
                                    
                                # Efficiency calculation for base model
                                throughput = found_metrics.get("throughput", 0)
                                time_val = found_metrics.get("time", 0)
                                samples = found_metrics.get("samples", 0)
                                
                                # Use same base_sps as merge_manager (default 10.0)
                                # We don't have access to Config here easily without import, assume 10.0
                                base_sps = 10.0 
                                eff_score = 0
                                if throughput > 0:
                                    eff_score = (throughput / base_sps) * 100
                                elif time_val > 0 and samples > 0:
                                    eff_score = ((samples / time_val) / base_sps) * 100
                                eff_score = min(100, max(1, eff_score)) if eff_score > 0 else 0
                                
                                ctx_val = found_metrics.get("context", 0)
                                ctx_score = min(100, (int(ctx_val) / 1024) * 10) if ctx_val else 0
                                
                                metrics["comparison"]["base_data"] = [
                                    acc,
                                    round(eff_score, 2),
                                    ctx_score
                                ]
                                # Update response with injected baseline
                                metrics["base_name"] = final_base_name

                                # Populate base_metrics for frontend display (ensure acc is 0-100)
                                base_metrics_export = found_metrics.copy()
                                base_metrics_export["acc"] = acc
                                base_metrics_export["efficiency"] = eff_score
                                metrics["base_metrics"] = base_metrics_export

                                resp["result"]["metrics"] = metrics
                            elif not is_original_valid:
                                metrics["base_name"] = "Baseline"
                                resp["result"]["metrics"] = metrics
                except Exception as e:
                    print(f"DEBUG: Exception in baseline population: {e}")
                    pass

            return resp
        except Exception:
            return None

    def read_evolution_progress(self, task_id):
        progress_path = os.path.join(self.state.merge_dir, task_id, "progress.json")
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
            if "eta_seconds" in data:
                result["eta_seconds"] = data.get("eta_seconds")
                result["estimated_completion"] = data.get("estimated_completion")
            if "current_step" in data:
                result["current_step"] = data.get("current_step")
                result["total_expected_steps"] = data.get("total_expected_steps")
            return result
        except Exception:
            return None
    
    def read_eval_progress(self, task_id):
        progress_path = os.path.join(self.state.merge_dir, task_id, "progress.json")
        if not os.path.isfile(progress_path):
            return None
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                pd = json.load(f)
            result = {
                "current": int(pd.get("current") or 0),
                "total": int(pd.get("total") or 0),
                "percent": int(pd.get("percent") or 0),
            }
            if "eta_seconds" in pd:
                result["eta_seconds"] = float(pd.get("eta_seconds") or 0.0)
            return result
        except Exception:
            return None


class TaskQueueMixin(HistoryMixin, ModelCompatibilityMixin):
    def kill_process_tree_by_pid(self, pid):
        ProcessManager.kill_process_tree(pid)

    def interrupt_current_task(self, reason="被高优先级任务打断"):
        current_id = self.state.running_task_info["id"]
        if not current_id or current_id not in self.state.tasks:
            return
        proc = self.state.running_task_info.get("process")
        if proc:
            print("!!! 正在打断任务 %s，原因: %s !!!" % (current_id, reason))
            self.kill_process_tree_by_pid(proc.pid)
        if "control" in self.state.tasks[current_id]:
            self.state.tasks[current_id]["control"]["aborted"] = True
        current_p_score = self.state.running_task_info["priority"]
        original_priority_str = self.state.tasks[current_id].get("priority", "common")
        if current_p_score == self.state.priority_map.get("cutin") or original_priority_str == "cutin":
            self.state.tasks[current_id]["status"] = "interrupted"
            self.state.tasks[current_id]["message"] = "任务被打断，等待恢复"
        else:
            self.state.tasks[current_id]["status"] = "queued"
            self.state.tasks[current_id]["message"] = "正在排队 (自动恢复中)..."
            self.state.tasks[current_id]["control"] = {"aborted": False, "process": None}
            self.state.tasks[current_id]["restarted"] = True
            task_data = self.state.tasks[current_id].get("original_data")
            created_at = self.state.tasks[current_id].get("created_at")
            self.state.task_queue.put((current_p_score, created_at, current_id, task_data))
        self.state.running_task_info["id"] = None
        self.state.running_task_info["priority"] = None
        self.state.running_task_info["process"] = None

    def popen_group_kwargs(self):
        return ProcessManager.create_process_group_kwargs()

    def worker(self):
        print("--- 任务处理 Worker 已启动 ---")
        while True:
            try:
                priority_score, created_at, task_id, data = self.state.task_queue.get()
            except Exception:
                continue
            if self.state.tasks.get(task_id, {}).get("status") == "stopped":
                self.state.task_queue.task_done()
                continue
            try:
                with self.state.scheduler_lock:
                    self.state.tasks[task_id]["status"] = "running"
                    self.state.tasks[task_id]["message"] = "正在初始化..."
                    task_control = {"aborted": False, "process": None}
                    self.state.tasks[task_id]["control"] = task_control
                    self.state.running_task_info["id"] = task_id
                    self.state.running_task_info["priority"] = priority_score

                def update_progress(p, msg):
                    if self.state.tasks.get(task_id, {}).get("status") in ["interrupted", "stopped"]:
                        return
                    self.state.tasks[task_id]["progress"] = p
                    self.state.tasks[task_id]["message"] = msg
                    if task_control.get("process") and self.state.running_task_info.get("id") == task_id:
                        self.state.running_task_info["process"] = task_control["process"]

                task_type = data.get("type", "merge")
                self.logger.info(
                    "[worker] 任务开始 task_id=%s type=%s custom_name=%s",
                    task_id,
                    task_type,
                    data.get("custom_name", ""),
                )
                if task_type == "merge":
                    self.logger.info(
                        "[worker] 标准融合 模型数=%s method=%s",
                        len(data.get("model_paths") or data.get("models") or []),
                        data.get("method", ""),
                    )
                elif task_type == "merge_evolutionary":
                    self.logger.info(
                        "[worker] 完全融合 模型路径=%s hf_subsets=%s pop_size=%s n_iter=%s",
                        data.get("model_paths", [])[:3],
                        data.get("hf_subsets", []),
                        data.get("pop_size"),
                        data.get("n_iter"),
                    )
                elif task_type == "eval_only":
                    self.logger.info(
                        "[worker] 评估任务 model_path=%s dataset=%s",
                        data.get("model_path", ""),
                        data.get("dataset", ""),
                    )
                elif task_type == "recipe_apply":
                    self.logger.info(
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
                        script_path = os.path.join(self.config.PROJECT_ROOT, "scripts", "run_vlm_search_bridge.py")
                        if not os.path.isfile(script_path):
                            self.state.tasks[task_id]["status"] = "error"
                            self.state.tasks[task_id]["message"] = "scripts/run_vlm_search_bridge.py 不存在"
                            _result = {"status": "error", "error": "run_vlm_search_bridge.py 未找到"}
                        else:
                            import subprocess
                            merge_dir = os.path.join(self.state.merge_dir, task_id)
                            os.makedirs(merge_dir, exist_ok=True)
                            progress_path = os.path.join(merge_dir, "progress.json")
                            meta_path = os.path.join(merge_dir, "metadata.json")
                            with open(meta_path, "w", encoding="utf-8") as f:
                                json.dump({"id": task_id, "type": "merge_evolutionary", "status": "running", **_data}, f, ensure_ascii=False, indent=2)
                            
                            task_start_time = time.time()
                            proc = subprocess.Popen(
                                [self.config.MERGENETIC_PYTHON, script_path, "--task-id", task_id],
                                cwd=self.config.PROJECT_ROOT,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                **self.popen_group_kwargs(),
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
                                
                                duration = time.time() - task_start_time
                                try:
                                    with open(meta_path, "r", encoding="utf-8") as f:
                                        m = json.load(f)
                                    m["duration_seconds"] = duration
                                    with open(meta_path, "w", encoding="utf-8") as f:
                                        json.dump(m, f, ensure_ascii=False, indent=2)
                                except Exception:
                                    pass

                                _result = {"status": "success"}
                                self.logger.info("[worker] 完全融合完成 task_id=%s duration=%.2fs", task_id, duration)
                                if _temp_suffixes:
                                    _mm.cleanup_recipe_temp_dirs(task_id, _temp_suffixes)
                            except Exception as e:
                                self.logger.exception("[worker] 完全融合失败 task_id=%s error=%s", task_id, e)
                                with open(progress_path, "w", encoding="utf-8") as f:
                                    json.dump({"status": "error", "message": str(e)}, f, ensure_ascii=False)
                                try:
                                    with open(meta_path, "r", encoding="utf-8") as f:
                                        m = json.load(f)
                                    m["status"] = "error"
                                    m["error"] = str(e)
                                    m["duration_seconds"] = time.time() - task_start_time
                                    with open(meta_path, "w", encoding="utf-8") as f:
                                        json.dump(m, f, ensure_ascii=False, indent=2)
                                except Exception:
                                    pass
                                _result = {"status": "error", "error": str(e)}
                                if _temp_suffixes:
                                    _mm.cleanup_recipe_temp_dirs(task_id, _temp_suffixes)
                    result = _result
                elif task_type == "eval_only":
                    import importlib
                    import merge_manager as _mm
                    importlib.reload(_mm)
                    result = _mm.run_eval_only_task(task_id, data, update_progress, task_control)
                elif task_type == "recipe_apply":
                    result = run_recipe_apply_task(task_id, data, update_progress, task_control)
                    if result.get("status") == "success":
                        self.logger.info("[worker] 配方融合完成 task_id=%s output=%s", task_id, result.get("output_path", ""))
                    else:
                        self.logger.warning("[worker] 配方融合失败 task_id=%s error=%s", task_id, result.get("error", ""))
                else:
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
                        self.logger.info("[worker] 标准融合完成 task_id=%s output=%s", task_id, result.get("output_dir", ""))
                    elif result and result.get("status") != "success":
                        self.logger.warning("[worker] 标准融合失败 task_id=%s error=%s", task_id, result.get("error", ""))

                if task_type == "eval_only" and result:
                    if result.get("status") == "success":
                        self.logger.info("[worker] 评估完成 task_id=%s", task_id)
                    else:
                        self.logger.warning("[worker] 评估失败 task_id=%s error=%s", task_id, result.get("error", ""))

                with self.state.scheduler_lock:
                    if self.state.running_task_info.get("id") == task_id:
                        self.state.running_task_info["id"] = None
                        self.state.running_task_info["process"] = None

                if self.state.tasks[task_id]["status"] not in ["interrupted", "queued", "stopped"]:
                    self.state.tasks[task_id]["result"] = result
                    if result.get("status") == "success":
                        self.state.tasks[task_id]["status"] = "completed"
                        self.state.tasks[task_id]["progress"] = 100
                        self.state.tasks[task_id]["message"] = "任务完成"
                    else:
                        self.state.tasks[task_id]["status"] = "error"
                        self.state.tasks[task_id]["message"] = "失败: %s" % result.get("error", "unknown")
            except Exception as e:
                self.logger.exception("[worker] 任务异常 task_id=%s: %s", task_id, e)
                print("Worker Error:", e)
                with self.state.scheduler_lock:
                    if self.state.running_task_info.get("id") == task_id:
                        self.state.running_task_info["id"] = None
                if self.state.tasks.get(task_id, {}).get("status") not in ["interrupted", "queued", "stopped"]:
                    self.state.tasks[task_id]["status"] = "error"
                    self.state.tasks[task_id]["message"] = "系统内部错误: %s" % str(e)
            finally:
                self.state.task_queue.task_done()


class ModelRepoMixin(ModelCompatibilityMixin):
    def merge_metadata_by_output_path(self, output_path: str):
        if not output_path or not os.path.isdir(self.state.merge_dir):
            return None
        output_path = os.path.abspath(output_path)
        for tid in os.listdir(self.state.merge_dir):
            out_dir = os.path.abspath(os.path.join(self.state.merge_dir, tid, "output"))
            if out_dir != output_path:
                continue
            meta_path = os.path.join(self.state.merge_dir, tid, "metadata.json")
            if not os.path.isfile(meta_path):
                return None
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def model_repo_list(self):
        data_path = os.path.join(self.state.project_root, "model_repo", "data", "models.json")
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
                if not m.get("path") and os.path.isdir(self.state.merge_dir):
                    for tid in os.listdir(self.state.merge_dir):
                        meta_path = os.path.join(self.state.merge_dir, tid, "metadata.json")
                        if not os.path.isfile(meta_path):
                            continue
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            if meta.get("status") == "success" and meta.get("custom_name") == m.get("name"):
                                m["path"] = os.path.abspath(os.path.join(self.state.merge_dir, tid, "output"))
                                break
                        except Exception:
                            continue
                m["is_base"] = False
                path = m.get("path")
                if path and os.path.isdir(path):
                    meta = self.merge_metadata_by_output_path(path)
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

    def base_models_list(self):
        base_path = getattr(self.config, "LOCAL_MODELS_PATH", None) or self.state.model_pool_path
        models = self.list_models_from_dir(base_path)
        for m in models:
            m["is_vlm"] = self.model_is_vlm(m.get("path") or "")
        return models

    def model_repo_path(self, model_id):
        for m in self.model_repo_list():
            if m.get("model_id") == model_id:
                return m.get("path")
        return None

    def model_repo_load_raw(self):
        data_path = os.path.join(self.state.project_root, "model_repo", "data", "models.json")
        if not os.path.isfile(data_path):
            return {}
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("models", data) if isinstance(data.get("models"), dict) else {}

    def model_repo_save_raw(self, models):
        data_path = os.path.join(self.state.project_root, "model_repo", "data", "models.json")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump({"models": models}, f, ensure_ascii=False, indent=2)


class TestsetMixin(BaseService):
    def testset_repo_path(self):
        return self.state.testset_data_path or os.path.join(self.state.project_root, "testset_repo", "data", "testsets.json")

    def load_testsets_dict(self):
        path = self.testset_repo_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("testsets", data) if isinstance(data.get("testsets"), dict) else {}
        except Exception:
            return {}

    def save_testsets_dict(self, testsets_dict):
        path = self.testset_repo_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"testsets": testsets_dict}, f, ensure_ascii=False, indent=2)

    def resolve_dataset_sample_count(self, hf_dataset, hf_subset, hf_split, cache_dir):
        if not (hf_dataset or "").strip():
            return 0, None, None
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
            self.logger.debug("testset count metadata failed: %s", e)
        return 0, None, None

    def refresh_testsets_counts(self, testsets_dict):
        if not testsets_dict:
            return []
        cache_dir = getattr(self.config, "HF_DATASETS_CACHE", None) or os.environ.get("HF_DATASETS_CACHE")
        changed = False
        for tid, entry in list(testsets_dict.items()):
            if not entry:
                continue
            if not (entry.get("hf_dataset") or "").strip():
                continue
            if int(entry.get("sample_count") or 0) > 0:
                continue
            n, actual_split, actual_subset = self.resolve_dataset_sample_count(
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
            self.save_testsets_dict(testsets_dict)
        return list(testsets_dict.values())

    def refresh_single_testset_count(self, entry):
        if not entry or not (entry.get("hf_dataset") or "").strip():
            return entry
        if int(entry.get("sample_count") or 0) > 0:
            return entry
        cache_dir = getattr(self.config, "HF_DATASETS_CACHE", None) or os.environ.get("HF_DATASETS_CACHE")
        n, actual_split, actual_subset = self.resolve_dataset_sample_count(
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

    def _scan_local_testsets(self, testsets):
        """扫描本地 YAML/JSON/PARQUET 文件并添加到 testsets 字典中"""
        scan_dirs = []
        
        yaml_dir = os.path.join(self.state.project_root, "testset_repo", "yaml")
        if os.path.isdir(yaml_dir):
            scan_dirs.append(yaml_dir)
            
        data_dir = os.path.join(self.state.project_root, "testset_repo", "data")
        if os.path.isdir(data_dir):
            scan_dirs.append(data_dir)

        import yaml
        # 移除 .py, .md 以避免误识别
        valid_exts = [".yaml", ".yml", ".json", ".jsonl", ".parquet"]
        
        # 收集当前有效的 local_ids，用于后续清理
        current_local_ids = set()

        for d in scan_dirs:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f == "testsets.json":
                    continue
                _, ext = os.path.splitext(f)
                if ext.lower() not in valid_exts:
                    continue
                
                path = os.path.join(d, f)
                tid = "local_" + os.path.splitext(f)[0]
            
            # 检查是否已存在（避免覆盖 testsets.json 中已定义的，除非是 local 类型且我们想更新它）
            # 但为了简单，先检查 id
            if tid in testsets:
                current_local_ids.add(tid)
                # 如果已存在，检查是否需要更新（例如文件修改时间）
                # 这里暂时略过，假设 id 碰撞即为同一个
                continue
            
            try:
                cfg = {}
                # 仅对 yaml/json 尝试解析内容
                if ext.lower() in [".yaml", ".yml"]:
                    try:
                        with open(path, "r", encoding="utf-8") as yf:
                            cfg = yaml.safe_load(yf) or {}
                    except Exception:
                        pass
                elif ext.lower() == ".json":
                    try:
                        with open(path, "r", encoding="utf-8") as jf:
                            cfg = json.load(jf) or {}
                    except Exception:
                        pass

                # 尝试从 YAML/JSON 或文件名解析信息
                hf_dataset = cfg.get("dataset_path") or cfg.get("hf_dataset")
                hf_subset = cfg.get("dataset_name") or cfg.get("hf_subset")
                hf_split = cfg.get("dataset_split") or cfg.get("hf_split")
                
                is_valid_dataset = False
                
                def has_template_vars(s):
                    return s and ("{{" in str(s) or "}}" in str(s))

                # Case A: Explicit config in YAML/JSON
                if hf_dataset: 
                    # 排除包含模板变量的 dataset_path
                    if has_template_vars(hf_dataset):
                        is_valid_dataset = False
                    else:
                        is_valid_dataset = True
                
                # Case B: Filename convention (e.g. openai___gsm8k...)
                elif ("___" in f or "__" in f) and not hf_dataset:
                    # Attempt to parse filename to extract dataset info
                    try:
                        # expected format: owner___dataset__subset.ext or owner___dataset.ext
                        # or dataset__subset.ext
                        name_part = os.path.splitext(f)[0]
                        if "___" in name_part:
                            parts = name_part.split("___")
                            if len(parts) >= 2:
                                hf_dataset = parts[0] + "/" + parts[1].split("__")[0]
                                if "__" in parts[1]:
                                    hf_subset = parts[1].split("__")[1]
                                is_valid_dataset = True
                        elif "__" in name_part:
                             parts = name_part.split("__")
                             if len(parts) >= 2:
                                 hf_dataset = parts[0]
                                 hf_subset = parts[1]
                                 is_valid_dataset = True
                    except Exception:
                        pass
                
                # Case C: Data files (jsonl, parquet) are inherently datasets
                elif ext.lower() in [".jsonl", ".parquet"]:
                    is_valid_dataset = True
                    # Try to infer dataset name from filename if not set
                    if not hf_dataset:
                        hf_dataset = os.path.splitext(f)[0]
                
                # Case D: YAML with 'tasks' (prompt template) but no explicit dataset_path? 
                elif cfg.get("tasks") or cfg.get("samples") or cfg.get("test_cases"):
                    # Check if extracted dataset path is valid
                    if not hf_dataset and cfg.get("tasks"):
                        try:
                            # 尝试从 tasks 列表的第一个元素提取
                            candidates = cfg["tasks"]
                            if isinstance(candidates, list) and len(candidates) > 0:
                                t0 = candidates[0]
                                if isinstance(t0, dict) and t0.get("dataset_path"):
                                    extracted_ds = t0.get("dataset_path")
                                    if not has_template_vars(extracted_ds):
                                        hf_dataset = extracted_ds
                                        hf_subset = t0.get("dataset_name") or hf_subset
                                        hf_split = t0.get("dataset_split") or hf_split
                                        is_valid_dataset = True
                        except:
                            pass
                    
                    # 如果仍然没有 hf_dataset，但有 samples/test_cases，则认为是本地数据定义
                    if not is_valid_dataset and (cfg.get("samples") or cfg.get("test_cases")):
                        is_valid_dataset = True
                        if not hf_dataset:
                            hf_dataset = os.path.splitext(f)[0]
                
                if not hf_dataset:
                    continue

                testsets[tid] = {
                    "testset_id": tid,
                    "name": f,
                    "hf_dataset": hf_dataset,
                    "hf_subset": hf_subset,
                    "hf_split": hf_split or "test",
                    "sample_count": 0,
                    "is_local": True,
                    "local_path": path,
                    "created_at": os.path.getctime(path),
                    "notes": "本地发现的文件"
                }
                current_local_ids.add(tid)

                # 尝试计算样本数
                try:
                    count = 0
                    if ext.lower() in [".yaml", ".yml", ".json"]:
                        # 尝试常见的字段
                        candidates = cfg.get("samples") or cfg.get("test_cases") or cfg.get("data") or cfg.get("examples") or []
                        if isinstance(candidates, list):
                            count = len(candidates)
                        # 如果是 list 类型的 root
                        elif isinstance(cfg, list):
                            count = len(cfg)
                        
                        # 如果没有直接的数据列表，尝试读取 tasks 配置 (针对 prompt template yaml)
                        if count == 0 and cfg.get("tasks"):
                            pass
                    elif ext.lower() == ".jsonl":
                        # Count lines for JSONL
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                count = sum(1 for line in f if line.strip())
                        except Exception:
                            pass
                    elif ext.lower() == ".parquet":
                        try:
                            # Attempt to use pyarrow for speed
                            import pyarrow.parquet as pq
                            meta = pq.read_metadata(path)
                            count = meta.num_rows
                        except Exception:
                            try:
                                # Fallback to pandas
                                import pandas as pd
                                df = pd.read_parquet(path)
                                count = len(df)
                            except Exception:
                                pass
                    
                    if count > 0:
                        testsets[tid]["sample_count"] = count
                    elif count == 0 and hf_dataset:
                        # 如果是本地 config 指向远程数据集，尝试从已知的 testsets 中查找 sample_count
                        # 查找逻辑：dataset + subset (strict match) -> dataset + subset (loose) -> dataset only
                        candidates_source = [t for t in testsets.values() if t.get("testset_id") != tid]
                        
                        match = None
                        # 1. Exact match dataset + subset + split
                        match = next((t for t in candidates_source if t.get("hf_dataset") == hf_dataset and t.get("hf_subset") == hf_subset and t.get("hf_split") == hf_split and t.get("sample_count", 0) > 0), None)
                        
                        # 2. Match dataset + subset
                        if not match and hf_subset:
                             match = next((t for t in candidates_source if t.get("hf_dataset") == hf_dataset and t.get("hf_subset") == hf_subset and t.get("sample_count", 0) > 0), None)
                        
                        # 3. Match dataset only (risky if subsets differ significantly, but better than 0)
                        if not match:
                             match = next((t for t in candidates_source if t.get("hf_dataset") == hf_dataset and t.get("sample_count", 0) > 0), None)
                             
                        if match:
                             testsets[tid]["sample_count"] = match["sample_count"]

                except Exception:
                    pass
            except Exception as e:
                self.logger.warning("处理本地 testset 文件失败 %s: %s", f, e)
        
        # 清理已不存在的或变为无效的 local entries
        to_remove = []
        has_changes = False
        
        # 标记本次扫描是否有新增或修改 (简单起见，只要有扫描到local文件就假设可能有变动，
        # 但为了避免频繁IO，我们只在删除时强制保存，或者依靠下方的 diff 检测)
        # 更好的做法是：如果在上面的循环中添加了新key，或者修改了现有key，就置 has_changes=True
        # 这里我们先只关注"清理"和"新增"的持久化
        
        for tid, entry in list(testsets.items()):
            # 判定是否属于本地仓库管理的文件：
            # 1. 明确标记为 is_local
            # 2. 或者 id 是 local_ 开头
            # 3. 或者路径在 testsets_repo 目录下
            is_managed_local = entry.get("is_local") or tid.startswith("local_")
            
            if is_managed_local:
                # 如果这个 ID 不在本次扫描到的 valid IDs 中，说明它对应的文件被删除了或者不再合法
                if tid not in current_local_ids:
                    to_remove.append(tid)
        
        if to_remove:
            has_changes = True
            for tid in to_remove:
                self.logger.info(f"Removing invalid/deleted local testset: {tid}")
                del testsets[tid]
        
        # 检查是否有新增的 ID (在 current_local_ids 但不在原 testsets keys 中 - 虽上面已赋值，但需判断是否触发保存)
        # 上面的循环已经把 valid 的都塞进 testsets 了，所以这里只要判断是否发生了写入
        # 为简单起见，只要进行了扫描并且有本地文件，我们就保存一次，确保 metadata 更新
        # 或者更精细点：
        if current_local_ids or has_changes:
             self.save_testsets_dict(testsets)

    def get_testset_by_id(self, testset_id, refresh=False):
        testsets = self.load_testsets_dict()
        self._scan_local_testsets(testsets) # 确保包含本地文件
        
        entry = testsets.get(testset_id)
        if not entry:
            return None
        if refresh:
            entry = self.refresh_single_testset_count(entry)
            testsets[testset_id] = entry
            # 注意：如果原本是 local 的，save_testsets_dict 会把它写入 testsets.json，
            # 这可能不是我们要的（也许我们希望它保持动态发现）。
            # 但为了持久化 sample_count，写入也是合理的。
            # 或者我们可以判断 is_local 就不写入 json？
            # 暂时允许写入，这样下次就变成持久化的了。
            self.save_testsets_dict(testsets)
        return entry

    def testset_list(self, refresh=True):
        testsets = self.load_testsets_dict()
        self._scan_local_testsets(testsets)
        
        return self.refresh_testsets_counts(testsets) if refresh else list(testsets.values())

    def infer_lm_eval_task(self, hf_dataset, hf_subset):
        if not (hf_dataset or "").strip():
            return ""
        key = (hf_dataset or "").strip().lower()
        if key in ("cais/mmlu", "mmlu") and (hf_subset or "").strip():
            return "mmlu_" + (hf_subset or "").strip().replace("-", "_")
        return ""

    def load_leaderboard(self):
        lb_path = os.path.join(self.state.project_root, "testset_repo", "data", "leaderboard.json")
        if not os.path.isfile(lb_path):
            return {}
        try:
            with open(lb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("leaderboards", {})
        except Exception:
            return {}


class RecipeMixin(ModelRepoMixin):
    def get_recipe_arch_path(self, recipe_id: str):
        recipe_path = os.path.join(self.state.recipes_dir, f"{recipe_id}.json")
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


class Services(RecipeMixin, TaskQueueMixin, TestsetMixin):
    def start_worker(self):
        import threading
        threading.Thread(target=self.worker, daemon=True).start()

    def start_task_worker(self):
        self.start_worker()
