from dataclasses import dataclass
import os
import glob
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError


@dataclass
class DataRequest:
    hf_dataset: str
    hf_subset: str | None = None
    testset_id: str | None = None  # 可选：用于读/写 TestSet 上 hf_info 缓存


class DatasetInfoService:
    _FILE_CACHE_TTL = 86400 * 7  # 7 天

    def __init__(self, config):
        self.config = config
        self._memo = {}

    def _hf_meta_cache_path(self) -> str:
        root = getattr(self.config, "PROJECT_ROOT", None) or os.getcwd()
        d = os.path.join(root, "cache")
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass
        return os.path.join(d, "hf_info_meta.json")

    def _read_file_meta_cache(self, key: str) -> dict | None:
        path = self._hf_meta_cache_path()
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                store = json.load(f)
            ent = (store or {}).get(key)
            if not ent or not isinstance(ent, dict):
                return None
            if time.time() - float(ent.get("ts", 0)) > self._FILE_CACHE_TTL:
                return None
            return ent.get("data")
        except Exception:
            return None

    def _write_file_meta_cache(self, key: str, data: dict) -> None:
        path = self._hf_meta_cache_path()
        try:
            store = {}
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    store = json.load(f) or {}
            if not isinstance(store, dict):
                store = {}
            store[key] = {"ts": time.time(), "data": data}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(store, f, ensure_ascii=False)
        except Exception:
            pass

    @property
    def cache_candidates(self):
        default_hf_cache = os.path.join(str(Path.home()), ".cache", "huggingface", "datasets")
        return [
            getattr(self.config, "EVAL_HF_DATASETS_CACHE", None),
            getattr(self.config, "HF_DATASETS_CACHE", None),
            os.environ.get("HF_DATASETS_CACHE", None),
            default_hf_cache,
        ]

    def invalidate_hf_info_cache(self, hf_dataset: str, hf_subset: str | None = None) -> None:
        """清除内存与磁盘 hf_info 缓存条目。"""
        hf_dataset = (hf_dataset or "").strip()
        if not hf_dataset:
            return
        sub = (hf_subset or "").strip()
        key = f"{hf_dataset}|{sub}"
        self._memo.pop(key, None)
        path = self._hf_meta_cache_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                store = json.load(f) or {}
            if isinstance(store, dict) and key in store:
                del store[key]
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(store, f, ensure_ascii=False)
        except Exception:
            pass

    def _parse_readme_dataset_info(self, text: str):
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

    def _collect_local_infos(self, hf_dataset: str, cache_root: str):
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
            files.extend(glob.glob(os.path.join(r, norm_name, "**", "dataset_infos.json"), recursive=True))
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
                    yaml_infos = self._parse_readme_dataset_info(text)
                    for cfg, splits_list in (yaml_infos or {}).items():
                        if cfg and cfg not in infos:
                            infos[cfg] = splits_list or []
                except Exception:
                    continue
        hub_root = os.path.join(str(Path.home()), ".cache", "huggingface", "hub")
        if os.path.isdir(hub_root):
            # 优化：只在特定数据集目录下查找，避免跨数据集（同名不同user）混淆
            repo_dir_name = "datasets--" + hf_dataset.replace("/", "--")
            target_dirs = glob.glob(os.path.join(hub_root, repo_dir_name))
            
            hub_files = []
            for d in target_dirs:
                hub_files.extend(glob.glob(os.path.join(d, "**", "dataset_infos.json"), recursive=True))
                hub_files.extend(glob.glob(os.path.join(d, "**", "dataset_info.json"), recursive=True))

            for p in hub_files:
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
                        if cfg and cfg not in infos:
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
                            if cfg and cfg not in infos:
                                infos[cfg] = split_keys
                except Exception:
                    continue
            try:
                readme_paths = glob.glob(os.path.join(hub_root, f"datasets--*--{base_name}", "snapshots", "*", "README.md"))
                for rp in readme_paths:
                    try:
                        with open(rp, "r", encoding="utf-8") as f:
                            text = f.read()
                        yaml_infos = self._parse_readme_dataset_info(text)
                        for cfg, splits_list in (yaml_infos or {}).items():
                            if cfg and cfg not in infos:
                                infos[cfg] = splits_list or []
                    except Exception:
                        continue
            except Exception:
                pass
        return infos

    def _collect_lm_task_infos(self, hf_dataset: str):
        base_name = hf_dataset.split("/")[-1].lower()
        project_root = getattr(self.config, "PROJECT_ROOT", None)
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
                if dataset_path:
                    d_lower = dataset_path.lower()
                    # 支持完整路径匹配（如 cais/mmlu）或 base_name 匹配（如 mmlu）
                    if d_lower != base_name and d_lower != hf_dataset.lower():
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

    def _collect_repo_infos(self, hf_dataset: str):
        data_path = getattr(self.config, "TESTSET_DATA_PATH", None)
        if not data_path or not os.path.isfile(data_path):
            return {}, []
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}, []
        testsets = data.get("testsets") if isinstance(data, dict) else None
        if isinstance(testsets, dict):
            entries = testsets.values()
        elif isinstance(data, dict):
            entries = data.values()
        else:
            entries = []
        infos = {}
        default_splits = []
        hf_dataset_norm = hf_dataset.strip().lower()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_dataset = (entry.get("hf_dataset") or "").strip().lower()
            if entry_dataset != hf_dataset_norm:
                continue
            subset = (entry.get("hf_subset") or "").strip()
            split = (entry.get("hf_split") or "").strip()
            if subset:
                existing = infos.get(subset, [])
                if split and split not in existing:
                    existing.append(split)
                infos[subset] = existing
            else:
                if split and split not in default_splits:
                    default_splits.append(split)
        return infos, default_splits

    def _collect_hub_infos(self, hf_dataset: str):
        endpoint = getattr(self.config, "HF_ENDPOINT", None) or os.environ.get("HF_ENDPOINT", None)
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except Exception:
            return {}
        try:
            api = HfApi(endpoint=endpoint) if endpoint else HfApi()
            files = api.list_repo_files(repo_id=hf_dataset, repo_type="dataset")
        except Exception:
            return {}
        target = ""
        if "dataset_infos.json" in files:
            target = "dataset_infos.json"
        elif "dataset_info.json" in files:
            target = "dataset_info.json"
        if not target:
            return {}
        try:
            cached = hf_hub_download(repo_id=hf_dataset, repo_type="dataset", filename=target, endpoint=endpoint)
        except Exception:
            return {}
        try:
            with open(cached, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            return {}
        infos = {}
        if isinstance(info, dict) and "splits" in info and (info.get("config_name") or "").strip():
            cfg = (info.get("config_name") or "").strip()
            split_info = info.get("splits") or {}
            split_keys = list(split_info.keys()) if isinstance(split_info, dict) else []
            if cfg:
                infos[cfg] = split_keys
            return infos
        for cfg_key, cfg_info in (info or {}).items():
            if not isinstance(cfg_info, dict):
                continue
            cfg = (cfg_info.get("config_name") or cfg_key or "").strip()
            split_info = cfg_info.get("splits") or {}
            split_keys = list(split_info.keys()) if isinstance(split_info, dict) else []
            if cfg:
                infos[cfg] = split_keys
        return infos

    def get_info(self, req: DataRequest) -> dict:
        hf_dataset = (req.hf_dataset or "").strip()
        if not hf_dataset:
            return {"status": "error", "message": "hf_dataset 必填"}
        requested_subset = (req.hf_subset or "").strip()
        cache_dir = getattr(self.config, "EVAL_HF_DATASETS_CACHE", None) or os.environ.get("HF_DATASETS_CACHE")

        key = f"{hf_dataset}|{requested_subset}"
        entry = self._memo.get(key)
        if entry:
            ts, data = entry
            if (data or {}).get("status") == "success" and (ts and (ts + 900) > time.time()):
                return data

        # 磁盘长期缓存（重启后仍可用）
        disk_hit = self._read_file_meta_cache(key)
        if disk_hit and (disk_hit or {}).get("status") == "success":
            self._memo[key] = (time.time(), disk_hit)
            return disk_hit

        # DB 中 TestSet 行缓存的 configs/splits
        testset_id = (getattr(req, "testset_id", None) or "").strip()
        if testset_id:
            try:
                from flask import has_app_context
                if has_app_context():
                    from app.extensions import db
                    from app.models import TestSet
                    row = db.session.get(TestSet, testset_id)
                    if row and row.cached_configs and row.cached_splits:
                        db_hit = {
                            "status": "success",
                            "hf_dataset": hf_dataset,
                            "configs": list(row.cached_configs),
                            "splits": list(row.cached_splits),
                            "source": "db_cache",
                        }
                        self._memo[key] = (time.time(), db_hit)
                        return db_hit
            except Exception:
                pass

        try:
            from datasets import load_dataset_builder, load_dataset  # 兼容旧导入，不强制使用
        except Exception:
            pass

        def _get_configs():
            try:
                from datasets import get_dataset_config_names, DownloadConfig
                dl = DownloadConfig(local_files_only=True)
                return get_dataset_config_names(hf_dataset, trust_remote_code=True, download_config=dl)
            except Exception:
                return []

        def _load_ds():
            try:
                from datasets import DownloadConfig
                dl = DownloadConfig(local_files_only=True)
                ds_local = load_dataset(hf_dataset, trust_remote_code=True, cache_dir=cache_dir, download_config=dl)
                if ds_local and hasattr(ds_local, "keys") and callable(getattr(ds_local, "keys", None)):
                    return list(ds_local.keys())
            except Exception:
                return []
            return []

        def _get_splits_from_builder(first_config):
            try:
                from datasets import DownloadConfig
                dl = DownloadConfig(local_files_only=True)
                b = load_dataset_builder(hf_dataset, first_config or None, trust_remote_code=True, cache_dir=cache_dir, download_config=dl)
                return list(b.info.splits.keys()) if b.info.splits else []
            except Exception:
                return []

        repo_infos, repo_default_splits = self._collect_repo_infos(hf_dataset)

        local_infos = {}
        for c in self.cache_candidates:
            local_infos.update(self._collect_local_infos(hf_dataset, c))
        hub_infos = {}
        lm_task_infos = self._collect_lm_task_infos(hf_dataset)

        merged_infos = {}
        for source in (local_infos, repo_infos, hub_infos, lm_task_infos):
            for cfg, split_list in (source or {}).items():
                if not cfg:
                    continue
                existing = merged_infos.get(cfg, [])
                for s in (split_list or []):
                    if s and s not in existing:
                        existing.append(s)
                merged_infos[cfg] = existing

        configs = []
        splits = []

        configs_set = set([c for c in (configs or []) if c])
        configs_set.update(merged_infos.keys())
        try:
            for c in _get_configs() or []:
                if c:
                    configs_set.add(c)
        except Exception:
            pass
        configs = list(configs_set)
        configs.sort()

        target_config = requested_subset if requested_subset and requested_subset in configs_set else (configs[0] if configs else None)

        if configs:
            splits = merged_infos.get(target_config) or splits

        if not splits and repo_default_splits:
            splits = repo_default_splits

        if not configs:
            configs = ["all"]
        if not splits:
            splits = ["all", "train", "validation", "test"]
        elif "all" not in splits:
            splits = ["all"] + [s for s in splits if s and s != "all"]

        result = {
            "status": "success",
            "hf_dataset": hf_dataset,
            "configs": configs or [],
            "splits": splits,
        }
        try:
            self._memo[key] = (time.time(), result)
            self._write_file_meta_cache(key, result)
            if testset_id:
                from flask import has_app_context
                if has_app_context():
                    from app.repositories import testset_update_hf_cache
                    testset_update_hf_cache(
                        testset_id,
                        cached_configs=list(configs or []),
                        cached_splits=list(splits or []),
                    )
        except Exception:
            pass
        return result
