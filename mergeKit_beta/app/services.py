import json
import importlib
import os
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import or_ as db_or
from sqlalchemy.orm import Session as _SaSession

from core.process_manager import ProcessManager
import merge_manager
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
        extra_paths = getattr(self.config, "LOCAL_MODELS_EXTRA_PATHS", None) or []
        bases = [local_path, self.state.model_pool_path] + list(extra_paths)
        if os.path.isabs(s) and os.path.isdir(s):
            return os.path.abspath(s)
        for base in bases:
            if not base:
                continue
            candidate = os.path.join(base, s)
            if os.path.isdir(candidate):
                return os.path.abspath(candidate)
        name = os.path.basename(s)
        for base in bases:
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
        config_path = os.path.join(root_path, "config.json")
        has_weights = False
        if os.path.isfile(config_path):
            try:
                for f in os.listdir(root_path):
                    if f.endswith(".safetensors") or f.endswith(".bin"):
                        has_weights = True
                        break
            except OSError:
                pass
        if has_weights:
            size_bytes = 0
            try:
                for f in os.listdir(root_path):
                    if f.endswith(".safetensors") or f.endswith(".bin"):
                        size_bytes += os.path.getsize(os.path.join(root_path, f))
            except OSError:
                pass
            out.append({
                "name": os.path.basename(root_path.rstrip(os.sep)),
                "size": size_bytes,
                "details": {"family": "HuggingFace Local"},
                "path": os.path.abspath(root_path),
            })
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

    def scan_base_models_to_db(self):
        """扫描基座模型目录并 upsert 到 Model 表（source='base'）。"""
        app = getattr(self, "app", None)
        if not app:
            return 0
        extra_paths = getattr(self.config, "LOCAL_MODELS_EXTRA_PATHS", None) or []
        scan_dirs = [
            getattr(self.config, "LOCAL_MODELS_PATH", None),
            getattr(self.state, "model_pool_path", None),
        ] + list(extra_paths)
        seen_paths = set()
        count = 0
        for base_path in scan_dirs:
            if not base_path or not os.path.isdir(base_path):
                continue
            for m in self.list_models_from_dir(base_path):
                path = (m.get("path") or "").strip()
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                is_vlm = self.model_is_vlm(path)
                arch = None
                try:
                    hs, nhl = self.get_model_arch(path)
                    if hs is not None and nhl is not None:
                        arch = "hs%d_nhl%d" % (hs, nhl)
                except Exception:
                    pass
                try:
                    with app.app_context():
                        from app.repositories import model_register
                        model_register(
                            path=path,
                            name=m.get("name", os.path.basename(path)),
                            source="base",
                            is_vlm=is_vlm,
                            size_bytes=m.get("size"),
                            architecture=arch,
                        )
                        count += 1
                except Exception as e:
                    self.logger.debug("[scan_base_models_to_db] 跳过 %s: %s", path, e)
        return count

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


def _get_canonical_arch_from_config(cfg):
    """
    从任意 HuggingFace 风格 config 中解析 hidden_size / num_hidden_layers / model_type，
    支持顶层、text_config、decoder_config、encoder_config 及常见别名字段。
    返回 (hidden_size, num_hidden_layers, model_type)，缺失为 None。
    """
    if not isinstance(cfg, dict):
        return None, None, None
    hs = cfg.get("hidden_size")
    nhl = cfg.get("num_hidden_layers")
    mt = (cfg.get("model_type") or "").strip().lower() or None
    # 别名字段：部分架构用 depth / n_layer / n_layers / num_layers
    nhl_keys = ("num_hidden_layers", "depth", "n_layer", "n_layers", "num_layers")
    for sub_key in ("text_config", "decoder_config", "encoder_config"):
        sub = cfg.get(sub_key)
        if not isinstance(sub, dict):
            continue
        if hs is None:
            hs = sub.get("hidden_size")
        if nhl is None:
            for k in nhl_keys:
                if sub.get(k) is not None:
                    nhl = sub.get(k)
                    break
        if mt is None or mt == "":
            mt = (sub.get("model_type") or "").strip().lower() or None
        if hs is not None and nhl is not None and mt:
            break
    if hs is None or nhl is None:
        for sub_key in ("text_config", "decoder_config", "encoder_config"):
            sub = cfg.get(sub_key)
            if not isinstance(sub, dict):
                continue
            if hs is None:
                hs = sub.get("hidden_size")
            if nhl is None:
                for k in nhl_keys:
                    if sub.get(k) is not None:
                        nhl = sub.get(k)
                        break
            if hs is not None and nhl is not None:
                break
    if mt is None or mt == "":
        mt = (cfg.get("model_type") or "").strip().lower() or None
    if mt is None and isinstance(cfg.get("architectures"), list) and cfg["architectures"]:
        mt = str(cfg["architectures"][0]).lower()
    # 优先使用顶层 model_type（如 qwen3_5），子配置仅作回退
    top_mt = (cfg.get("model_type") or "").strip().lower()
    if top_mt:
        mt = top_mt
    try:
        hs = int(hs) if hs is not None else None
    except (TypeError, ValueError):
        hs = None
    try:
        nhl = int(nhl) if nhl is not None else None
    except (TypeError, ValueError):
        nhl = None
    return hs, nhl, (mt or None)


class ModelCompatibilityMixin(ModelPathMixin):
    def _load_model_config(self, model_path: str):
        """加载 config.json，失败返回 None。"""
        if not model_path or not os.path.isdir(model_path):
            return None
        config_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_path):
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def model_is_vlm(self, model_path: str) -> bool:
        if not model_path or not os.path.isdir(model_path):
            return False
        dir_name = os.path.basename(model_path.rstrip(os.sep)).lower()
        vlm_dir_keywords = (
            "vl", "vision", "vlm", "_vl-", "-vl-", "-vl_", "qwen2.5vl", "qwen2vl",
            "llava", "cogvlm", "minicpm-v", "minicpm_v", "visual", "qwen2_vl", "qwen2.5_vl", "qwen3_5",
        )
        if any(k in dir_name for k in vlm_dir_keywords):
            return True
        cfg = self._load_model_config(model_path)
        if not cfg:
            return False
        if cfg.get("vision_config") is not None:
            return True
        if any(cfg.get(k) is not None for k in ("image_token_id", "vision_start_token_id", "vision_token_id")):
            return True
        model_type = (cfg.get("model_type") or "").lower()
        for sub in ("text_config", "decoder_config"):
            sub_cfg = cfg.get(sub)
            if isinstance(sub_cfg, dict) and (sub_cfg.get("model_type") or ""):
                model_type = model_type or (sub_cfg.get("model_type") or "").lower()
        archs = cfg.get("architectures") or []
        arch_str = " ".join(str(a) for a in archs).lower()
        vlm_indicators = ("vision", "vl", "qwen2_vl", "qwen2.5_vl", "qwen2vl", "qwen3_5", "llava", "cogvlm", "minicpm-v", "visual")
        if any(v in model_type for v in vlm_indicators):
            return True
        if any(v in arch_str for v in vlm_indicators):
            return True
        return False

    def get_model_type(self, model_path: str):
        cfg = self._load_model_config(model_path)
        if not cfg:
            return None
        _, _, mt = _get_canonical_arch_from_config(cfg)
        if mt:
            return mt
        return (cfg.get("model_type") or "").strip().lower() or None

    def get_model_arch(self, model_path: str):
        cfg = self._load_model_config(model_path)
        if not cfg:
            return None, None
        hs, nhl, _ = _get_canonical_arch_from_config(cfg)
        return hs, nhl

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
            return False, "模型架构不一致（hidden_size 或 num_hidden_layers 不同），无法融合。请参见 DEVELOPMENT.md 的兼容性说明。", types
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

    def _get_all_history_from_file(self):
        """从 merges 目录扫描融合历史（文件回退）。"""
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
                if meta.get("type") == "merge_evolutionary":
                    meta["fusion_method"] = "Evolutionary (Ties-Dare)"
                elif meta.get("type") == "recipe_apply":
                    meta["fusion_method"] = "Recipe Config"
                history_list.append(meta)
            except Exception:
                continue
        return sorted(history_list, key=lambda x: x.get("created_at", 0), reverse=True)

    def get_all_history(self):
        """融合历史：优先 DB，无数据则回退文件。"""
        app = getattr(self, "app", None)
        if app:
            try:
                with app.app_context():
                    from app.db_read_layer import get_fusion_history_from_db
                    data = get_fusion_history_from_db()
                    if data is not None and len(data) > 0:
                        return data
            except Exception:
                pass
        return self._get_all_history_from_file()

    def _get_all_eval_history_from_file(self):
        """从 merges 目录扫描评估历史（文件回退）。"""
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

    def get_all_eval_history(self):
        """评估历史：优先 DB，无数据则回退文件。"""
        app = getattr(self, "app", None)
        if app:
            try:
                with app.app_context():
                    from app.db_read_layer import get_eval_history_from_db
                    data = get_eval_history_from_db()
                    if data is not None and len(data) > 0:
                        return data
            except Exception:
                pass
        return self._get_all_eval_history_from_file()

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
                "type": meta.get("type"),
                "is_active": False,
                "result": {"status": raw_status, "metrics": meta.get("metrics")},
            }
            
            task_type = (meta.get("type") or "").strip()
            if task_type == "eval_only":
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
            elif task_type == "merge_evolutionary":
                # 进化融合：从 progress.json 填充 evolution_progress，供前端/完成弹窗展示
                evo = self.read_evolution_progress(task_id)
                if evo is not None:
                    cur = evo.get("current_step")
                    tot = evo.get("total_expected_steps")
                    if isinstance(cur, (int, float)) and isinstance(tot, (int, float)) and cur > 0 and tot > 0 and cur > tot:
                        evo["total_expected_steps"] = int(cur)
                    resp["evolution_progress"] = evo
                    resp["original_data"] = {"n_iter": meta.get("n_iter"), "pop_size": meta.get("pop_size")}
                    pct = evo.get("percent")
                    if pct is not None and resp["status"] in ("running", "unknown"):
                        resp["progress"] = min(99, max(0, int(pct)))
                    if evo.get("message") and resp["status"] in ("running", "unknown"):
                        resp["message"] = evo["message"]
            
            # Attempt to populate baseline data if missing (eval_only / merge_evolutionary)
            if task_type in ["eval_only", "merge_evolutionary"]:
                try:
                    metrics = resp.get("result", {}).get("metrics")
                    if metrics and metrics.get("comparison"):
                        comp = metrics["comparison"]
                        base_data = comp.get("base_data")
                        base_name = (metrics.get("base_name") or "").strip() or "Baseline"
                        
                        # If base_data is missing or all zeros, try to find baseline (同架构基础模型 / leaderboard / 任务历史 / 硬编码)
                        if not base_data or all(v == 0 for v in (base_data or [])):
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
                                testset_id = meta.get("testset_id")

                                # Priority 0: Query DB EvaluationResult
                                app = getattr(self, "app", None)
                                if app and not found_metrics:
                                    try:
                                        with app.app_context():
                                            from app.extensions import db as _db
                                            from app.models import EvaluationResult as ER, Model as M
                                            q = _db.session.query(ER).join(M, ER.model_id == M.id)
                                            if testset_id:
                                                q = q.filter(ER.testset_id == testset_id)
                                            q = q.filter(
                                                db_or(
                                                    M.name == target_name,
                                                    M.path.endswith("/" + target_name_norm),
                                                    M.name == target_name_norm,
                                                )
                                            ).order_by(ER.created_at.desc())
                                            db_row = q.first()
                                            if db_row and db_row.accuracy and db_row.accuracy > 0:
                                                found_metrics = {
                                                    "acc": db_row.accuracy,
                                                    "throughput": db_row.throughput or 0,
                                                    "time": db_row.time or 0,
                                                    "samples": db_row.test_cases or 0,
                                                    "context": db_row.context or 0,
                                                    "name": target_name,
                                                }
                                                final_base_name = target_name
                                    except Exception:
                                        pass

                                if found_metrics:
                                    break

                                # Priority 1: Check Leaderboard (file fallback)
                                try:
                                    project_root = (self.state.project_root or os.path.dirname(self.state.merge_dir) or "").strip()
                                    if not project_root and self.state.merge_dir:
                                        project_root = os.path.dirname(os.path.abspath(self.state.merge_dir))
                                    lb_path = os.path.join(project_root, "testset_repo", "data", "leaderboard.json")
                                    if os.path.isfile(lb_path):
                                        with open(lb_path, "r", encoding="utf-8") as f:
                                            lb_data = json.load(f)
                                        leaderboards = lb_data.get("leaderboards", {})
                                        
                                        # Strategy 1: Check specific testset_id (and by hf_dataset if leaderboard keyed by dataset name)
                                        candidate_entries = []
                                        if testset_id:
                                            candidate_entries.extend(leaderboards.get(testset_id, []))
                                        if not candidate_entries and (hf_dataset or "").strip():
                                            candidate_entries.extend(leaderboards.get((hf_dataset or "").strip(), []))
                                        if not candidate_entries and hf_dataset and hf_subset:
                                            candidate_entries.extend(leaderboards.get("%s (%s)" % (hf_dataset, hf_subset), []))
                                        
                                        # Strategy 2: If not found, check ALL leaderboards for exact match on dataset/subset/model
                                        # This helps when using a cloned testset but baseline data exists in another (e.g. global/default) testset
                                        # Only do this if we haven't found it yet
                                        
                                        def check_entries(entries_list):
                                            for entry in entries_list:
                                                entry_name = entry.get("model_name")
                                                # Check exact match or basename match
                                                if entry_name == target_name or normalize_model_name(entry_name) == target_name_norm:
                                                    # Verify if metrics are valid (non-zero accuracy or throughput)
                                                    t_acc = entry.get("accuracy", 0) or entry.get("acc", 0)
                                                    t_thr = entry.get("throughput", 0)
                                                    if t_acc is not None and isinstance(t_acc, str):
                                                        t_acc = float((t_acc or "0").replace("%", "").strip()) or 0
                                                    t_acc = float(t_acc) if t_acc is not None else 0
                                                    if t_acc > 0 or t_thr > 0:
                                                        # Match dataset: treat None and "" as equal
                                                        ed = (entry.get("hf_dataset") or "").strip()
                                                        es = (entry.get("hf_subset") or "").strip()
                                                        md = (hf_dataset or "").strip()
                                                        ms = (hf_subset or "").strip()
                                                        if ed == md and es == ms:
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
                                    self.logger.debug("Error reading leaderboard: %s", e)
                                
                                if found_metrics:
                                    break

                                # Priority 2: Scan Task History (only if not found)
                                if not candidate_dirs:
                                    if os.path.isdir(self.state.merge_dir):
                                        try:
                                            candidate_dirs = sorted(os.listdir(self.state.merge_dir), reverse=True)
                                        except OSError:
                                            candidate_dirs = []
                                    else:
                                        candidate_dirs = []
                                
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
                                             # Check dataset match (None/empty treated equal)
                                             if (c_meta.get("hf_dataset") or "") == (hf_dataset or "") and (c_meta.get("hf_subset") or "") == (hf_subset or ""):
                                                 c_m = c_meta.get("metrics", {})
                                                 raw_acc = c_m.get("acc") or c_m.get("accuracy") or 0
                                                 if raw_acc is not None:
                                                     if isinstance(raw_acc, str):
                                                         raw_acc = float((raw_acc or "0").replace("%", "").strip()) or 0
                                                     else:
                                                         raw_acc = float(raw_acc)
                                                 else:
                                                     raw_acc = 0
                                                 if raw_acc > 0:
                                                     found_metrics = dict(c_m)
                                                     found_metrics["acc"] = raw_acc
                                                     found_metrics["samples"] = c_m.get("test_cases") or c_m.get("samples") or 0
                                                     found_metrics["time"] = c_m.get("time", 0)
                                                     found_metrics["throughput"] = c_m.get("throughput", 0)
                                                     found_metrics["context"] = c_m.get("context", 0)
                                                     final_base_name = c_name
                                                     break
                                    except:
                                        pass
                                
                                if found_metrics:
                                    break
                            
                            # 不使用跨数据集的硬编码基准，避免雷达图基准线与当前测评数据集不可比导致展示错误

                            if found_metrics:
                                try:
                                    # Construct base data from found metrics (leaderboard / 任务历史 / 硬编码)
                                    # comparison 轴: ["Accuracy", "Efficiency", "Context"]
                                    raw_acc = found_metrics.get("acc") or found_metrics.get("accuracy") or 0
                                    if isinstance(raw_acc, str):
                                        raw_acc = float((raw_acc or "0").replace("%", "").strip()) or 0
                                    else:
                                        raw_acc = float(raw_acc) if raw_acc is not None else 0
                                    acc = raw_acc
                                    if 0 < acc <= 1.0:
                                        acc = acc * 100.0
                                    
                                    throughput = found_metrics.get("throughput", 0)
                                    time_val = found_metrics.get("time", 0)
                                    samples = found_metrics.get("samples", 0) or found_metrics.get("test_cases", 0)
                                    
                                    base_sps = 10.0 
                                    eff_score = 0
                                    if throughput and float(throughput) > 0:
                                        eff_score = (float(throughput) / base_sps) * 100
                                    elif time_val and samples and float(time_val) > 0 and float(samples) > 0:
                                        eff_score = ((float(samples) / float(time_val)) / base_sps) * 100
                                    eff_score = min(100, max(1, eff_score)) if eff_score > 0 else 0
                                    
                                    ctx_val = found_metrics.get("context", 0)
                                    try:
                                        ctx_val = int(ctx_val) if ctx_val not in (None, "") else 0
                                    except (TypeError, ValueError):
                                        ctx_val = 0
                                    ctx_score = min(100, (ctx_val / 1024) * 10) if ctx_val else 0
                                    
                                    metrics["comparison"]["base_data"] = [
                                        acc,
                                        round(eff_score, 2),
                                        ctx_score
                                    ]
                                    metrics["base_name"] = final_base_name
                                    base_metrics_export = found_metrics.copy() if isinstance(found_metrics, dict) else {}
                                    base_metrics_export["acc"] = acc
                                    base_metrics_export["efficiency"] = eff_score
                                    metrics["base_metrics"] = base_metrics_export
                                    resp["result"]["metrics"] = metrics
                                except Exception as e:
                                    self.logger.debug("baseline base_data build: %s", e)
                            else:
                                is_original_valid = bool(
                                    metrics.get("base_name")
                                    or ((metrics.get("comparison") or {}).get("base_data"))
                                )
                                if not is_original_valid:
                                    metrics["base_name"] = "Baseline"
                                    resp["result"]["metrics"] = metrics
                except Exception as e:
                    self.logger.debug("baseline population: %s", e)

            return resp
        except Exception:
            return None

    def _sync_evolution_csv_to_db(self, task_id):
        """读取 evolution_stream.csv（或兼容旧 vlm_search.csv）并批量写入 evolution_steps 表。"""
        import csv as csv_mod
        from app.repositories import evolution_steps_bulk_insert
        csv_candidates = [
            os.path.join(self.state.merge_dir, task_id, "vlm_search_results", "evolution_stream.csv"),
            os.path.join(self.state.merge_dir, task_id, "vlm_search_results", "vlm_search.csv"),
            os.path.join(self.state.merge_dir, task_id, "vlm_search_results", "vlm_search", "vlm_search.csv"),
            os.path.join(self.state.merge_dir, task_id, "search_results.csv"),
        ]
        csv_path = None
        for p in csv_candidates:
            if os.path.isfile(p):
                csv_path = p
                break
        if not csv_path:
            return
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        if rows:
            evolution_steps_bulk_insert(task_id, rows)
            self.logger.info("[worker] 已同步 %d 条进化步骤到 DB task_id=%s", len(rows), task_id)

    def read_evolution_progress(self, task_id):
        """读取进化任务进度（progress.json），供 API 与前端展示；含 percent/message 以驱动进度条。"""
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
            if "total_expected_steps" in data:
                result["total_expected_steps"] = data.get("total_expected_steps")
            if "percent" in data:
                result["percent"] = int(data.get("percent", 0))
            elif result.get("current_step") is not None and result.get("total_expected_steps"):
                total = result["total_expected_steps"]
                if total > 0:
                    result["percent"] = min(99, round(100 * result["current_step"] / total))
            if "message" in data and data["message"]:
                result["message"] = str(data["message"])
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

    def _db_mark_running(self, task_id, log_path=None):
        app = getattr(self, "app", None)
        if not app:
            return
        try:
            with app.app_context():
                from app.repositories import task_mark_running
                task_mark_running(task_id, log_path=log_path)
        except Exception as e:
            self.logger.warning("[worker] DB task_mark_running 失败: %s", e)

    def _db_update_completion(self, task_id, status, task_type, result=None):
        app = getattr(self, "app", None)
        if not app:
            return
        gen_1 = avg_merge = final_eval = None
        if task_type == "merge_evolutionary" and status == "completed":
            progress_path = os.path.join(self.state.merge_dir, task_id, "progress.json")
            if os.path.isfile(progress_path):
                try:
                    with open(progress_path, "r", encoding="utf-8") as f:
                        prog = json.load(f)
                    gen_1 = prog.get("first_gen_time") or (prog.get("gen_durations") or {}).get(1)
                    avg_merge = prog.get("avg_step_time")
                except Exception:
                    pass
            for item in os.listdir(os.path.join(self.state.merge_dir, task_id)):
                fusion_path = os.path.join(self.state.merge_dir, task_id, item, "fusion_info.json")
                if os.path.isfile(fusion_path):
                    try:
                        with open(fusion_path, "r", encoding="utf-8") as f:
                            info = json.load(f)
                        final_eval = info.get("final_test_duration")
                        break
                    except Exception:
                        pass
        try:
            with app.app_context():
                from app.repositories import task_update_after_completion
                task_update_after_completion(
                    task_id,
                    "completed" if status == "completed" else "failed",
                    gen_1_duration=gen_1,
                    avg_merge_time=avg_merge,
                    final_eval_duration=final_eval,
                )
        except Exception as e:
            self.logger.warning("[worker] DB task_update_after_completion 失败: %s", e)

    def _db_register_model(self, output_path, task_id, task_type, result=None):
        """在 DB 中注册融合产物（merge / recipe_apply / merge_evolutionary 成功后调用）。"""
        if not output_path or not os.path.isdir(output_path):
            return
        output_path = os.path.abspath(output_path)
        if os.path.islink(output_path):
            output_path = os.path.realpath(output_path)
        name = None
        parent_model_ids = None
        if task_type == "merge_evolutionary":
            fusion_path = os.path.join(output_path, "fusion_info.json")
            if os.path.isfile(fusion_path):
                try:
                    with open(fusion_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    name = info.get("custom_name") or os.path.basename(output_path.rstrip("/"))
                    parent_model_ids = info.get("model_paths") or []
                except Exception:
                    pass
        if name is None:
            meta = self.merge_metadata_by_output_path(output_path)
            name = (meta or {}).get("display_name") or (meta or {}).get("custom_name") or os.path.basename(output_path.rstrip("/"))
            parent_model_ids = (meta or {}).get("model_paths")
        if not name:
            name = os.path.basename(output_path.rstrip("/")) or "merged"
        app = getattr(self, "app", None)
        if not app:
            return
        try:
            with app.app_context():
                from app.repositories import model_register
                model_register(
                    path=output_path,
                    name=name,
                    source="merged",
                    task_id=task_id,
                    parent_model_ids=parent_model_ids or [],
                )
        except Exception as e:
            self.logger.warning("[worker] DB model_register 失败: %s", e)

    def _db_insert_eval_result(self, task_id, data, result):
        """评估成功后双写 EvaluationResult 表（与 _update_leaderboard 并存）。
        若被评估模型路径尚未在 models 表中，则先注册为 source=base，保证排行榜能显示模型名。"""
        if result.get("status") != "success":
            return
        metrics = result.get("metrics") or {}
        testset_id = data.get("testset_id")
        if not testset_id:
            return
        model_path = (data.get("model_path") or "").strip().rstrip(os.sep)
        model_id = None
        app = getattr(self, "app", None)
        if not app:
            return
        try:
            with app.app_context():
                from app.repositories import (
                    evaluation_result_insert,
                    model_get_by_path,
                    model_register,
                    testset_get_by_id,
                    testset_upsert,
                )
                # 保证 testset 存在，满足 evaluation_results.testset_id 外键（文件侧 testset 可能未入 DB）
                if testset_get_by_id(testset_id) is None:
                    # 优先从 testsets.json 取完整信息，避免写入只有 id/name 的空记录
                    entry = (self.load_testsets_dict() or {}).get(testset_id)
                    if entry and isinstance(entry, dict):
                        testset_upsert(
                            testset_id=testset_id,
                            name=entry.get("name") or data.get("testset_name") or testset_id,
                            hf_dataset=entry.get("hf_dataset") or data.get("hf_dataset"),
                            hf_subset=entry.get("hf_subset") or data.get("hf_subset"),
                            hf_split=entry.get("hf_split") or data.get("hf_split"),
                            lm_eval_task=entry.get("lm_eval_task") or data.get("lm_eval_task"),
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
                    else:
                        testset_upsert(
                            testset_id=testset_id,
                            name=data.get("testset_name") or testset_id,
                            hf_dataset=data.get("hf_dataset"),
                            hf_subset=data.get("hf_subset"),
                            hf_split=data.get("hf_split"),
                            lm_eval_task=data.get("lm_eval_task"),
                            sample_count=0,
                            is_local=False,
                        )
                if model_path:
                    model = model_get_by_path(model_path)
                    if model is None:
                        # 基座模型首次被评估时注册，便于排行榜显示模型名
                        name = os.path.basename(model_path) or data.get("model_name") or "base"
                        try:
                            arch = None
                            if hasattr(self, "get_model_arch"):
                                hs, nhl = self.get_model_arch(model_path)
                                if hs is not None and nhl is not None:
                                    arch = "hs%d_nhl%d" % (hs, nhl)
                            model = model_register(
                                path=model_path,
                                name=name,
                                source="base",
                                architecture=arch,
                            )
                        except Exception as reg_err:
                            self.logger.warning("[worker] 评估前注册 base 模型失败（继续写入结果）: %s", reg_err)
                            model = model_get_by_path(model_path)
                    if model:
                        model_id = model.id
                evaluation_result_insert(
                    task_id=task_id,
                    testset_id=testset_id,
                    model_id=model_id,
                    accuracy=metrics.get("accuracy"),
                    f1_score=metrics.get("f1_score"),
                    test_cases=metrics.get("test_cases"),
                    context=metrics.get("context") if isinstance(metrics.get("context"), (int, float)) else None,
                    throughput=metrics.get("throughput"),
                    time=metrics.get("time"),
                    metrics=metrics,
                    hf_dataset=data.get("hf_dataset"),
                    hf_subset=data.get("hf_subset"),
                    efficiency_score=metrics.get("efficiency_score"),
                )
        except Exception as e:
            self.logger.warning("[worker] DB evaluation_result_insert 失败: %s", e)

    def _post_task_gpu_cleanup(self, task_id: str) -> None:
        """任务结束后（无论成功/失败/异常）统一释放 Python 侧 GPU 显存与 Ray 实例。
        只做 cache 清理，不终止 Flask 进程或其他正常运行的子进程。"""
        import gc
        try:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            # 若 Ray 被本次任务初始化且现已无活跃 worker，则关闭以释放 CUDA 上下文
            try:
                import ray
                if ray.is_initialized():
                    ray.shutdown()
            except Exception:
                pass
            self.logger.debug("[worker] GPU 显存清理完成 task_id=%s", task_id)
        except Exception as cleanup_err:
            self.logger.warning("[worker] GPU 清理异常（不影响主流程）task_id=%s: %s", task_id, cleanup_err)

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

                merge_dir = os.path.join(self.state.merge_dir, task_id)
                self._db_mark_running(task_id, log_path=merge_dir)

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
                        if not self.config.evolution_merge_entry_exists():
                            self.state.tasks[task_id]["status"] = "error"
                            miss = (
                                "scripts/run_vlm_search_bridge.py 不存在"
                                if getattr(self.config, "MERGEKIT_EVOLUTION_LEGACY_BRIDGE", False)
                                else "evolution/runner.py 缺失（或未设置 MERGEKIT_EVOLUTION_LEGACY_BRIDGE）"
                            )
                            self.state.tasks[task_id]["message"] = miss
                            _result = {"status": "error", "error": miss}
                        else:
                            import subprocess
                            merge_dir = os.path.join(self.state.merge_dir, task_id)
                            os.makedirs(merge_dir, exist_ok=True)
                            progress_path = os.path.join(merge_dir, "progress.json")
                            meta_path = os.path.join(merge_dir, "metadata.json")
                            _evo_meta = {"id": task_id, "type": "merge_evolutionary", "status": "running", **_data}
                            merge_manager._write_metadata(task_id, merge_dir, _evo_meta)
                            
                            task_start_time = time.time()
                            _evo_argv, _evo_cwd = self.config.evolution_merge_popen_args(task_id)
                            proc = subprocess.Popen(
                                _evo_argv,
                                cwd=_evo_cwd,
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
                                        pct, msg = 50, line.strip()[:80]
                                        try:
                                            if os.path.isfile(progress_path):
                                                with open(progress_path, "r", encoding="utf-8") as pf:
                                                    pd = json.load(pf)
                                                pct = int(pd.get("percent", 50))
                                                if "message" in pd and pd["message"]:
                                                    msg = str(pd["message"])[:80]
                                        except Exception:
                                            pass
                                        update_progress(pct, msg)
                                if proc.returncode != 0:
                                    raise RuntimeError("子进程退出码: %s" % proc.returncode)
                                
                                duration = time.time() - task_start_time
                                try:
                                    with open(meta_path, "r", encoding="utf-8") as f:
                                        m = json.load(f)
                                    m["duration_seconds"] = duration
                                    merge_manager._write_metadata(task_id, merge_dir, m)
                                except Exception:
                                    pass

                                # 从 CSV 同步进化步骤到 DB
                                try:
                                    self._sync_evolution_csv_to_db(task_id)
                                except Exception as db_err:
                                    self.logger.warning("[worker] 同步 CSV→DB 失败（不影响任务）: %s", db_err)

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
                                    merge_manager._write_metadata(task_id, merge_dir, m)
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
                    import importlib as _importlib
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
                            _importlib.reload(merge_manager)
                            result = merge_manager.run_merge_task(task_id, _data, update_progress, task_control)
                        if _merge_temp_suffixes:
                            _mm_merge.cleanup_recipe_temp_dirs(task_id, _merge_temp_suffixes)
                    else:
                        _importlib.reload(merge_manager)
                        result = merge_manager.run_merge_task(task_id, data, update_progress, task_control)
                    if result and result.get("status") == "success":
                        self.logger.info("[worker] 标准融合完成 task_id=%s output=%s", task_id, result.get("output_path", ""))
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
                        self._db_update_completion(task_id, "completed", task_type, result)
                        if task_type in ("merge", "recipe_apply"):
                            self._db_register_model(result.get("output_path"), task_id, task_type, result)
                        elif task_type == "merge_evolutionary":
                            out_dir = os.path.join(self.state.merge_dir, task_id, "output")
                            if os.path.isdir(out_dir):
                                self._db_register_model(out_dir, task_id, task_type, result)
                        elif task_type == "eval_only":
                            self._db_insert_eval_result(task_id, data, result)
                    else:
                        self.state.tasks[task_id]["status"] = "error"
                        self.state.tasks[task_id]["message"] = "失败: %s" % result.get("error", "unknown")
                        self._db_update_completion(task_id, "error", task_type, result)
            except Exception as e:
                self.logger.exception("[worker] 任务异常 task_id=%s: %s", task_id, e)
                print("Worker Error:", e)
                with self.state.scheduler_lock:
                    if self.state.running_task_info.get("id") == task_id:
                        self.state.running_task_info["id"] = None
                if self.state.tasks.get(task_id, {}).get("status") not in ["interrupted", "queued", "stopped"]:
                    self.state.tasks[task_id]["status"] = "error"
                    self.state.tasks[task_id]["message"] = "系统内部错误: %s" % str(e)
                    self._db_update_completion(task_id, "error", data.get("type", "merge"), None)
            finally:
                self._post_task_gpu_cleanup(task_id)
                self.state.task_queue.task_done()


class ModelRepoMixin(ModelCompatibilityMixin):
    def model_delete_from_db_by_path(self, path: str) -> bool:
        """删除模型时同步移除 DB 中的 Model 记录；失败仅打日志，不影响主流程。"""
        if not (path or "").strip():
            return False
        app = getattr(self, "app", None)
        if not app:
            return False
        try:
            with app.app_context():
                from app.repositories import model_delete_by_path
                return model_delete_by_path(path)
        except Exception as e:
            self.logger.warning("[model_repo] DB model_delete_by_path 失败: %s", e)
            return False

    def model_get_path_by_id(self, model_id: str) -> str | None:
        """按 DB 的 model_id 取 path，供列表来自 DB 时的删除用。"""
        if not (model_id or "").strip():
            return None
        app = getattr(self, "app", None)
        if not app:
            return None
        try:
            with app.app_context():
                from app.repositories import model_get_by_id
                m = model_get_by_id(model_id)
                return (m.path or "").strip().rstrip("/") or None if m else None
        except Exception:
            return None

    def merge_metadata_by_output_path(self, output_path: str):
        if not output_path or not os.path.isdir(self.state.merge_dir):
            return None
        output_path = os.path.realpath(os.path.abspath(output_path))
        for tid in os.listdir(self.state.merge_dir):
            out_dir = os.path.join(self.state.merge_dir, tid, "output")
            if not os.path.lexists(out_dir):
                continue
            # output 常为指向命名产物目录的符号链接：须用 realpath 与已注册 path 比较
            if os.path.realpath(os.path.abspath(out_dir)) != output_path:
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
        """模型仓库列表：优先 DB，无数据则从文件 + merges 扫描并同步 DB。"""
        app = getattr(self, "app", None)
        if app:
            try:
                with app.app_context():
                    from app.db_read_layer import get_model_repo_list_from_db
                    data = get_model_repo_list_from_db(self.merge_metadata_by_output_path, getattr(self.state, "merge_dir", None))
                    if data is not None and len(data) > 0:
                        return data
            except Exception:
                pass
        data_path = os.path.join(self.state.project_root, "model_repo", "data", "models.json")
        out = []
        if os.path.isfile(data_path):
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                models = data.get("models", data) if isinstance(data.get("models"), dict) else data
                if isinstance(models, dict):
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
            except Exception:
                pass
        # 补充：扫描 merges/ 下已成功但未在列表中的融合目录，保证本地融合产物都能进列表并同步 DB
        if os.path.isdir(self.state.merge_dir):
            existing_paths = {os.path.abspath((m.get("path") or "")).rstrip("/") for m in out if m.get("path")}
            for tid in os.listdir(self.state.merge_dir):
                meta_path = os.path.join(self.state.merge_dir, tid, "metadata.json")
                out_dir = os.path.join(self.state.merge_dir, tid, "output")
                if not os.path.isfile(meta_path) or not os.path.isdir(out_dir):
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("status") != "success":
                        continue
                    path = os.path.abspath(out_dir).rstrip("/")
                    if os.path.islink(out_dir):
                        path = os.path.realpath(out_dir)
                    if path in existing_paths:
                        continue
                    existing_paths.add(path)
                    name = meta.get("custom_name") or os.path.basename(path) or tid
                    m = {
                        "path": path,
                        "name": name,
                        "task_id": tid,
                        "is_base": False,
                        "parent_models": [os.path.basename(p) for p in (meta.get("model_paths") or [])],
                        "recipe": {},
                        "dataset": {},
                    }
                    fusion_info = None
                    fi_path = os.path.join(path, "fusion_info.json")
                    if os.path.isfile(fi_path):
                        try:
                            with open(fi_path, "r", encoding="utf-8") as f:
                                fusion_info = json.load(f)
                        except Exception:
                            fusion_info = None
                    if fusion_info and isinstance(fusion_info, dict):
                        pn = [str(p) for p in (fusion_info.get("parent_names") or []) if p]
                        if pn:
                            m["parent_models"] = pn
                        m["recipe"] = {
                            "weights": fusion_info.get("best_genotype"),
                            "method": "evolutionary",
                            "dtype": fusion_info.get("dtype"),
                            "library": "mergenetic",
                            "pop_size": fusion_info.get("pop_size"),
                            "n_iter": fusion_info.get("n_iter"),
                            "max_samples": fusion_info.get("max_samples"),
                        }
                        m["dataset"] = {
                            "hf_dataset": fusion_info.get("hf_dataset"),
                            "hf_subset": fusion_info.get("hf_subset"),
                            "hf_subsets": fusion_info.get("hf_subsets"),
                            "hf_split": fusion_info.get("hf_split"),
                        }
                    else:
                        merge_meta = self.merge_metadata_by_output_path(path)
                        if merge_meta:
                            m["parent_models"] = [os.path.basename(p) for p in (merge_meta.get("model_paths") or [])]
                            mr = merge_meta.get("recipe")
                            if isinstance(mr, dict):
                                m["recipe"] = {
                                    "weights": mr.get("weights"),
                                    "method": mr.get("method"),
                                    "dtype": mr.get("dtype"),
                                    "library": mr.get("library"),
                                }
                            m["dataset"] = {
                                "hf_dataset": merge_meta.get("hf_dataset"),
                                "hf_subset": merge_meta.get("hf_subset"),
                                "hf_subsets": merge_meta.get("hf_subsets"),
                                "hf_split": merge_meta.get("hf_split"),
                            }
                    out.append(m)
                except Exception:
                    continue
        # 每次打开列表时，将当前文件/merges 中的融合模型 upsert 到 DB，保证及时更新
        app = getattr(self, "app", None)
        try:
            from flask import current_app
            if current_app:
                app = current_app._get_current_object()
        except RuntimeError:
            pass
        if getattr(self, "logger", None):
            if not app:
                self.logger.info("[model_repo_list] 跳过 DB 同步: 无 app 上下文 (列表共 %d 项)", len(out))
            elif not out:
                self.logger.info("[model_repo_list] 跳过 DB 同步: 列表为空 merge_dir=%s", getattr(self.state, "merge_dir", ""))
        if app and out:
            try:
                with app.app_context():
                    from app.repositories import model_register
                    db_uri = (app.config.get("SQLALCHEMY_DATABASE_URI") or "").replace("sqlite:///", "")
                    if getattr(self, "logger", None):
                        self.logger.info("[model_repo_list] 开始同步 %d 个融合模型到 DB: %s", len(out), db_uri or "(default)")
                    synced = 0
                    for m in out:
                        path = (m.get("path") or "").strip()
                        if not path or not os.path.isdir(path):
                            continue
                        path = os.path.abspath(path).rstrip("/")
                        meta = self.merge_metadata_by_output_path(path)
                        name = (m.get("name") or (meta.get("custom_name") if meta else None) or os.path.basename(path))
                        task_id = (m.get("task_id") or (meta.get("id") if meta else None))
                        parent_model_ids = (meta.get("model_paths") or []) if meta else []
                        model_register(path=path, name=name or "merged", source="merged", task_id=task_id, parent_model_ids=parent_model_ids)
                        synced += 1
                    if getattr(self, "logger", None):
                        self.logger.info("[model_repo_list] DB 同步完成: 已写入 %d 条 (Navicat 请连接上述路径)", synced)
            except Exception as e:
                if getattr(self, "logger", None):
                    self.logger.warning("[model_repo_list] DB models 同步失败: %s", e, exc_info=True)
        return out

    def base_models_list(self):
        extra_paths = getattr(self.config, "LOCAL_MODELS_EXTRA_PATHS", None) or []
        bases = [
            getattr(self.config, "LOCAL_MODELS_PATH", None),
            self.state.model_pool_path,
        ] + list(extra_paths)
        seen = set()
        models = []
        for base_path in bases:
            if not base_path or not os.path.isdir(base_path):
                continue
            for m in self.list_models_from_dir(base_path):
                path = (m.get("path") or "").strip()
                if path and path not in seen:
                    seen.add(path)
                    m["is_vlm"] = self.model_is_vlm(path)
                    models.append(m)
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
        """DB-first：先写 DB，再写文件作为缓存。"""
        app = getattr(self, "app", None)
        if app and testsets_dict:
            try:
                with app.app_context():
                    from app.repositories import testset_upsert
                    for tid, entry in testsets_dict.items():
                        if not entry or not isinstance(entry, dict):
                            continue
                        testset_id = entry.get("testset_id") or tid
                        name = entry.get("name") or str(testset_id) or "未命名"
                        testset_upsert(
                            testset_id=str(testset_id),
                            name=name,
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
                self.logger.warning("[save_testsets_dict] DB testset_upsert 失败: %s", e)
        # 文件缓存（失败仅 warn）
        try:
            path = self.testset_repo_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"testsets": testsets_dict}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning("[save_testsets_dict] 文件缓存写入失败: %s", e)

    def _resolve_single_config_sample_count(self, hf_dataset, hf_subset, hf_split, cache_dir):
        """单个 HF config 的样本数（builder 元数据）；失败返回 0,None,None。"""
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
                    builder = load_dataset_builder(
                        hf_dataset, name=subset_name or None, trust_remote_code=True, cache_dir=cache_dir
                    )
                    splits = getattr(builder.info, "splits", None) or {}
                    if splits:
                        preferred = []
                        if hf_split:
                            preferred.append(hf_split.split("[")[0].strip())
                        preferred.extend(["test", "validation", "dev", "val", "train"])
                        for split_name in preferred:
                            if split_name in splits:
                                return (
                                    int(getattr(splits[split_name], "num_examples", 0) or 0),
                                    split_name,
                                    subset_name,
                                )
                        first_key = list(splits.keys())[0]
                        return (
                            int(getattr(splits[first_key], "num_examples", 0) or 0),
                            first_key,
                            subset_name,
                        )
                except Exception:
                    continue
        except Exception as e:
            self.logger.debug("single config sample count failed: %s", e)
        return 0, None, None

    def _resolve_sample_count_via_load_dataset(self, hf_dataset, hf_subset, hf_split, cache_dir):
        """用 load_dataset 实际行数计数（较慢，作 builder 失败时的兜底）。"""
        try:
            from datasets import load_dataset
        except Exception:
            return 0, None, None
        sub = (hf_subset or "").strip() or None
        split_name = (hf_split or "").strip().split("[")[0].strip() or "test"
        try:
            ds = load_dataset(
                hf_dataset,
                name=sub,
                split=split_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            n = len(ds)
            return int(n), split_name, sub
        except Exception:
            try:
                ds = load_dataset(hf_dataset, name=sub, trust_remote_code=True, cache_dir=cache_dir)
                if hasattr(ds, "keys"):
                    for sk in ("test", "validation", "val", "train"):
                        if sk in ds:
                            return int(len(ds[sk])), sk, sub
                    k0 = list(ds.keys())[0]
                    return int(len(ds[k0])), k0, sub
            except Exception as e:
                self.logger.debug("load_dataset count failed: %s", e)
        return 0, None, None

    def resolve_dataset_sample_count(self, hf_dataset, hf_subset, hf_split, cache_dir):
        if not (hf_dataset or "").strip():
            return 0, None, None
        hf_dataset = (hf_dataset or "").strip()
        if hf_dataset.lower() == "unknown":
            return 0, None, None
        hf_subset_key = (hf_subset or "").strip()
        _MMLU_ALIAS = {"high_school_government": "high_school_government_and_politics"}
        ds_lower = hf_dataset.lower()

        # 组级子集（MMLU / CMMMU）：展开后累加各 config 的 num_examples
        expanded = None
        try:
            from app.routes import resolve_hf_subsets

            if "mmlu" in ds_lower and hf_subset_key:
                expanded = resolve_hf_subsets("mmlu", hf_subset_key)
                if expanded:
                    expanded = [_MMLU_ALIAS.get(s, s) for s in expanded]
            elif "cmmmu" in ds_lower and hf_subset_key:
                expanded = resolve_hf_subsets("cmmmu", hf_subset_key)
        except Exception as e:
            self.logger.debug("resolve_hf_subsets import/use failed: %s", e)

        if expanded:
            total = 0
            first_split = None
            used_subset = hf_subset_key
            for sub in expanded:
                n, sp, _ = self._resolve_single_config_sample_count(hf_dataset, sub, hf_split, cache_dir)
                if n <= 0:
                    n2, sp2, _ = self._resolve_sample_count_via_load_dataset(hf_dataset, sub, hf_split, cache_dir)
                    n, sp = n2, sp2 or sp
                total += n
                if first_split is None and sp:
                    first_split = sp
            if total > 0:
                return total, first_split, used_subset

        # 单 config（含别名）
        hf_one = hf_subset_key
        if hf_one and "mmlu" in ds_lower:
            hf_one = _MMLU_ALIAS.get(hf_one, hf_one)
        n, sp, sn = self._resolve_single_config_sample_count(hf_dataset, hf_one, hf_split, cache_dir)
        if n > 0:
            return n, sp, sn or hf_one
        n2, sp2, sn2 = self._resolve_sample_count_via_load_dataset(hf_dataset, hf_one, hf_split, cache_dir)
        if n2 > 0:
            return n2, sp2, sn2 or hf_one
        try:
            from datasets import get_dataset_config_names

            if not hf_one:
                subset_candidates = [None]
                try:
                    configs = get_dataset_config_names(hf_dataset, trust_remote_code=True) or []
                    subset_candidates.extend([c for c in configs if c])
                except Exception:
                    pass
                for subset_name in subset_candidates:
                    n, sp, sn = self._resolve_single_config_sample_count(
                        hf_dataset, subset_name, hf_split, cache_dir
                    )
                    if n > 0:
                        return n, sp, sn
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
            hf_dataset = (entry.get("hf_dataset") or "").strip()
            # 本地条目 hf_dataset 为 Unknown 时，尝试从文件名推断
            if (not hf_dataset or hf_dataset.lower() == "unknown") and entry.get("is_local") and entry.get("local_path"):
                local_path = entry.get("local_path") or ""
                f = os.path.basename(local_path)
                name_part = os.path.splitext(f)[0]
                if "___" in name_part:
                    parts = name_part.split("___")
                    if len(parts) >= 2:
                        hf_dataset = parts[0] + "/" + parts[1].split("__")[0]
                        entry["hf_dataset"] = hf_dataset
                        changed = True
            if not hf_dataset or hf_dataset.lower() == "unknown":
                continue
            # 每次刷新都重新解析条数，确保与 HF 元数据一致（修正错误的 0/100 等）
            n, actual_split, actual_subset = self.resolve_dataset_sample_count(
                hf_dataset,
                entry.get("hf_subset"),
                entry.get("hf_split"),
                cache_dir,
            )
            if n >= 0:
                old_count = int(entry.get("sample_count") or 0)
                # 仅当新解析到有效条数，或原先为 0 时更新，避免网络/加载失败时用 0 覆盖正确值
                if (n > 0 or old_count == 0) and n != old_count:
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
        app = getattr(self, "app", None)
        if app:
            try:
                with app.app_context():
                    from app.repositories import testset_get_by_id
                    row = testset_get_by_id(testset_id)
                    if row is not None:
                        entry = row.to_dict()
                        if refresh:
                            entry = self.refresh_single_testset_count(entry)
                            testsets = self.load_testsets_dict()
                            testsets[testset_id] = entry
                            self.save_testsets_dict(testsets)
                        return entry
            except Exception as e:
                self.logger.debug("[get_testset_by_id] DB 查询失败，回退文件: %s", e)
        testsets = self.load_testsets_dict()
        self._scan_local_testsets(testsets)
        
        entry = testsets.get(testset_id)
        if not entry:
            return None
        if refresh:
            entry = self.refresh_single_testset_count(entry)
            testsets[testset_id] = entry
            self.save_testsets_dict(testsets)
        return entry

    def _schedule_testset_count_refresh(self, entries_snapshot: list):
        """后台补全 sample_count=0 的测试集，不阻塞列表 API。"""
        app = getattr(self, "app", None)
        if not app or not entries_snapshot:
            return

        def run():
            with app.app_context():
                try:
                    from app.repositories import testset_get_by_id, testset_upsert

                    cache_dir = getattr(self.config, "HF_DATASETS_CACHE", None) or os.environ.get(
                        "HF_DATASETS_CACHE"
                    )
                    for d in entries_snapshot:
                        tid = d.get("id") or d.get("testset_id")
                        if not tid:
                            continue
                        row = testset_get_by_id(tid)
                        hf_dataset = (row.hf_dataset if row else d.get("hf_dataset")) or ""
                        hf_subset = row.hf_subset if row else d.get("hf_subset")
                        hf_split = row.hf_split if row else d.get("hf_split")
                        if not (hf_dataset or "").strip():
                            continue
                        n, sp, _ = self.resolve_dataset_sample_count(
                            hf_dataset, hf_subset, hf_split, cache_dir
                        )
                        if n > 0:
                            testset_upsert(
                                testset_id=tid,
                                name=(row.name if row else d.get("name")) or str(tid),
                                hf_dataset=hf_dataset,
                                hf_subset=hf_subset,
                                hf_split=sp or hf_split,
                                lm_eval_task=row.lm_eval_task if row else d.get("lm_eval_task"),
                                benchmark_config=row.benchmark_config if row else d.get("benchmark_config"),
                                version=row.version if row else d.get("version"),
                                sample_count=n,
                                is_local=row.is_local if row else bool(d.get("is_local")),
                                local_path=row.local_path if row else d.get("local_path"),
                                yaml_template_path=row.yaml_template_path if row else d.get("yaml_template_path"),
                                created_by=row.created_by if row else d.get("created_by"),
                                notes=row.notes if row else d.get("notes"),
                                question_type=row.question_type if row else d.get("question_type"),
                                type=row.type if row else d.get("type"),
                                cached_configs=row.cached_configs if row else d.get("cached_configs"),
                                cached_splits=row.cached_splits if row else d.get("cached_splits"),
                            )
                except Exception as e:
                    self.logger.debug("async testset count refresh failed: %s", e)

        threading.Thread(target=run, daemon=True).start()

    def testset_list(self, refresh=True):
        # 以 DB 为主：有 DB 数据时直接返回，否则回退到文件并可选同步到 DB
        app = getattr(self, "app", None)
        if app:
            try:
                with app.app_context():
                    from app.repositories import (
                        testset_list_from_db,
                        testset_upsert,
                        testset_enrich_from_eval_result,
                    )
                    rows = testset_list_from_db()
                    if rows:
                        file_testsets = self.load_testsets_dict() or {}
                        out = []
                        for r in rows:
                            d = r.to_dict()
                            # 补全：DB 中缺少 hf_dataset/lm_eval_task 等时，优先从文件、其次从 EvaluationResult 补全并回写 DB
                            if not (d.get("hf_dataset") or d.get("lm_eval_task")):
                                entry = file_testsets.get(r.id)
                                if entry and isinstance(entry, dict):
                                    for k in ("hf_dataset", "hf_subset", "hf_split", "lm_eval_task", "type", "name", "sample_count"):
                                        if entry.get(k) is not None and d.get(k) in (None, ""):
                                            d[k] = entry.get(k)
                                    testset_upsert(
                                        testset_id=r.id,
                                        name=entry.get("name") or r.name,
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
                                        cached_configs=entry.get("cached_configs"),
                                        cached_splits=entry.get("cached_splits"),
                                    )
                                else:
                                    enrich = testset_enrich_from_eval_result(r.id)
                                    if enrich:
                                        for k, v in enrich.items():
                                            d[k] = v
                                        testset_upsert(
                                            testset_id=r.id,
                                            name=r.name,
                                            hf_dataset=enrich.get("hf_dataset"),
                                            hf_subset=enrich.get("hf_subset"),
                                            hf_split=enrich.get("hf_split"),
                                        )
                            out.append(d)
                        if refresh:
                            zeros = [
                                dict(x)
                                for x in out
                                if int(x.get("sample_count") or 0) == 0
                                and (x.get("hf_dataset") or "").strip()
                            ]
                            if zeros:
                                self._schedule_testset_count_refresh(zeros)
                        return out
            except Exception as e:
                self.logger.debug("[testset_list] DB 查询失败，回退文件: %s", e)
        testsets = self.load_testsets_dict()
        self._scan_local_testsets(testsets)
        # 当 DB 为空时，将当前文件内容同步到 DB（一次性）
        if testsets and app:
            try:
                with app.app_context():
                    from app.repositories import testset_list_from_db
                    if not testset_list_from_db():
                        self.save_testsets_dict(testsets)
            except Exception:
                pass
        lst = list(testsets.values())
        if refresh:
            zeros = [
                dict(e)
                for e in lst
                if int(e.get("sample_count") or 0) == 0 and (e.get("hf_dataset") or "").strip()
            ]
            if zeros and app:
                self._schedule_testset_count_refresh(zeros)
            return lst
        return lst

    def infer_lm_eval_task(self, hf_dataset, hf_subset):
        if not (hf_dataset or "").strip():
            return ""
        key = (hf_dataset or "").strip().lower()
        if key in ("cais/mmlu", "mmlu") and (hf_subset or "").strip():
            subset = (hf_subset or "").strip().replace("-", "_")
            if subset == "biology":
                return "mmlu_college_biology"
            return "mmlu_" + subset
        if key in ("m-a-p/cmmmu", "cmmmu/cmmmu"):
            # 交给 merge_manager 按本机已注册任务动态解析，避免写死到不存在的子任务名
            return "cmmmu"
        return ""

    def _load_leaderboard_from_file(self):
        """从 leaderboard.json 读取（文件回退）。"""
        lb_path = os.path.join(self.state.project_root, "testset_repo", "data", "leaderboard.json")
        if not os.path.isfile(lb_path):
            return {}
        try:
            with open(lb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("leaderboards", {})
        except Exception:
            return {}

    def load_leaderboard(self):
        """排行榜：优先 DB，无数据则回退文件。"""
        app = getattr(self, "app", None)
        if app:
            try:
                with app.app_context():
                    from app.db_read_layer import get_leaderboard_from_db
                    data = get_leaderboard_from_db()
                    if data is not None and len(data) > 0:
                        return data
            except Exception:
                pass
        return self._load_leaderboard_from_file()


class AutomationMixin(BaseService):
    """优胜劣汰：按评估结果打 Keep/Discard 标签，供清理脚本使用。"""

    def compute_top10_tags(self, top_percent=10):
        """
        按测试集聚合评估结果，每测试集内按模型最佳准确率排序，前 top_percent% 打 Keep，其余打 Discard。
        需在 app_context 内调用；无 DB 或无数据时安全返回。
        """
        app = getattr(self, "app", None)
        if not app:
            return {"keep": 0, "discard": 0, "error": "no app context"}
        try:
            with app.app_context():
                from app.repositories import (
                    evaluation_best_per_model_per_testset,
                    model_get_by_id,
                    model_add_tag,
                    model_remove_tag,
                )
                rows = evaluation_best_per_model_per_testset()
                if not rows:
                    return {"keep": 0, "discard": 0}
                # 按 testset_id 分组，每组内按 best_acc 降序，取前 top_percent% 为 Keep
                by_testset = {}
                for tid, mid, acc in rows:
                    by_testset.setdefault(tid, []).append((mid, acc))
                keep_ids = set()
                discard_ids = set()
                for tid, pairs in by_testset.items():
                    pairs.sort(key=lambda x: -x[1])
                    n = len(pairs)
                    k = max(1, int(round(n * top_percent / 100.0)))
                    for i, (mid, _) in enumerate(pairs):
                        if i < k:
                            keep_ids.add(mid)
                        else:
                            discard_ids.add(mid)
                # 参与过评估的模型：先移除 Keep/Discard，再重新打
                all_ids = keep_ids | discard_ids
                for mid in all_ids:
                    m = model_get_by_id(mid)
                    if not m:
                        continue
                    model_remove_tag(m, "Keep")
                    model_remove_tag(m, "Discard")
                for mid in keep_ids:
                    m = model_get_by_id(mid)
                    if m:
                        model_add_tag(m, "Keep", category="lifecycle")
                for mid in discard_ids - keep_ids:
                    m = model_get_by_id(mid)
                    if m:
                        model_add_tag(m, "Discard", category="lifecycle")
                return {"keep": len(keep_ids), "discard": len(discard_ids - keep_ids)}
        except Exception as e:
            if getattr(self, "logger", None):
                self.logger.warning("[automation] compute_top10_tags 失败: %s", e)
            return {"keep": 0, "discard": 0, "error": str(e)}

    def list_models_with_tag(self, tag_name):
        """返回带有指定标签的模型 path 列表，供清理脚本使用。"""
        app = getattr(self, "app", None)
        if not app:
            return []
        try:
            with app.app_context():
                from app.extensions import db
                from app.models import Model, Tag, model_tags
                tag = db.session.query(Tag).filter_by(name=tag_name).first()
                if not tag:
                    return []
                models = db.session.query(Model).join(model_tags, Model.id == model_tags.c.model_id).filter(model_tags.c.tag_id == tag.id).all()
                return [m.path for m in models if m.path and os.path.isdir(m.path)]
        except Exception:
            return []


class RecipeMixin(AutomationMixin, ModelRepoMixin):
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
    def heal_stale_merge_tasks_on_startup(self):
        """启动自愈：修正重启后遗留的 merge/merge_evolutionary running/queued 脏状态。"""
        app = getattr(self, "app", None)
        if not app:
            return
        healed = 0
        try:
            with app.app_context():
                from app.extensions import db
                from app.models import Task
                stale_tasks = (
                    db.session.query(Task)
                    .filter(
                        Task.task_type.in_(["merge", "merge_evolutionary"]),
                        Task.status.in_(["running", "queued"]),
                    )
                    .all()
                )
                now = datetime.utcnow()
                for t in stale_tasks:
                    task_id = t.id
                    new_status = "interrupted"
                    meta_path = os.path.join(self.state.merge_dir, task_id, "metadata.json")
                    if os.path.isfile(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                raw = (json.load(f).get("status") or "").strip()
                            if raw == "success":
                                new_status = "completed"
                            elif raw == "error":
                                new_status = "error"
                        except Exception:
                            new_status = "interrupted"
                    t.status = new_status
                    t.updated_at = now
                    if new_status in ("completed", "error", "failed"):
                        t.finished_at = t.finished_at or now
                    healed += 1
                if healed:
                    db.session.commit()
                    self.logger.info("[startup-heal] 已修复 stale merge 任务状态: %d", healed)
                else:
                    self.logger.info("[startup-heal] 无需修复 stale merge 任务")
        except Exception as e:
            self.logger.warning("[startup-heal] 修复 stale merge 状态失败: %s", e)
    def heal_stale_eval_tasks_on_startup(self):
        """启动自愈：修正重启后遗留的 eval_only running/queued 脏状态。"""
        app = getattr(self, "app", None)
        if not app:
            return
        healed = 0
        try:
            with app.app_context():
                from app.extensions import db
                from app.models import Task
                stale_tasks = (
                    db.session.query(Task)
                    .filter(
                        Task.task_type == "eval_only",
                        Task.status.in_(["running", "queued"]),
                    )
                    .all()
                )
                now = datetime.utcnow()
                for t in stale_tasks:
                    task_id = t.id
                    new_status = "interrupted"
                    meta_path = os.path.join(self.state.merge_dir, task_id, "metadata.json")
                    if os.path.isfile(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                raw = (json.load(f).get("status") or "").strip()
                            if raw == "success":
                                new_status = "completed"
                            elif raw == "error":
                                new_status = "error"
                        except Exception:
                            new_status = "interrupted"
                    t.status = new_status
                    t.updated_at = now
                    if new_status in ("completed", "error", "failed"):
                        t.finished_at = t.finished_at or now
                    healed += 1
                if healed:
                    db.session.commit()
                    self.logger.info("[startup-heal] 已修复 stale eval 任务状态: %d", healed)
                else:
                    self.logger.info("[startup-heal] 无需修复 stale eval 任务")
        except Exception as e:
            self.logger.warning("[startup-heal] 修复 stale eval 状态失败: %s", e)

    def start_worker(self):
        import threading
        threading.Thread(target=self.worker, daemon=True).start()

    def start_task_worker(self):
        self.heal_stale_eval_tasks_on_startup()
        self.heal_stale_merge_tasks_on_startup()
        self.start_worker()
