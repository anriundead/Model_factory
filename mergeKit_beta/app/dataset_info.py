from dataclasses import dataclass
import os
import glob
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError


@dataclass
class DataRequest:
    hf_dataset: str
    hf_subset: str | None = None


class DatasetInfoService:
    def __init__(self, config):
        self.config = config

    @property
    def cache_candidates(self):
        default_hf_cache = os.path.join(str(Path.home()), ".cache", "huggingface", "datasets")
        return [
            getattr(self.config, "EVAL_HF_DATASETS_CACHE", None),
            getattr(self.config, "HF_DATASETS_CACHE", None),
            os.environ.get("HF_DATASETS_CACHE", None),
            default_hf_cache,
        ]

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
            files.extend(glob.glob(os.path.join(r, base_name, "**", "dataset_info.json"), recursive=True))
            files.extend(glob.glob(os.path.join(r, norm_name, "**", "dataset_infos.json"), recursive=True))
            files.extend(glob.glob(os.path.join(r, base_name, "**", "dataset_infos.json"), recursive=True))
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

        datasets_ok = True
        try:
            from datasets import get_dataset_config_names, load_dataset_builder, load_dataset
        except Exception:
            datasets_ok = False

        def _get_configs():
            try:
                return get_dataset_config_names(hf_dataset, trust_remote_code=True)
            except Exception:
                return []

        def _load_ds():
            try:
                ds_local = load_dataset(hf_dataset, trust_remote_code=True, cache_dir=cache_dir)
                if ds_local and hasattr(ds_local, "keys") and callable(getattr(ds_local, "keys", None)):
                    return list(ds_local.keys())
            except Exception:
                return []
            return []

        def _get_splits_from_builder(first_config):
            try:
                b = load_dataset_builder(hf_dataset, first_config or None, trust_remote_code=True, cache_dir=cache_dir)
                return list(b.info.splits.keys()) if b.info.splits else []
            except Exception:
                return []

        repo_infos, repo_default_splits = self._collect_repo_infos(hf_dataset)
        if repo_infos or repo_default_splits:
            configs = list({c for c in repo_infos.keys() if c})
            configs.sort()
            target_config = requested_subset if requested_subset and requested_subset in configs else (configs[0] if configs else None)
            splits = repo_infos.get(target_config) or []
            if not splits and repo_default_splits:
                splits = repo_default_splits
            if not splits:
                splits = ["train", "validation", "test"]
            return {
                "status": "success",
                "hf_dataset": hf_dataset,
                "configs": configs or [],
                "splits": splits,
            }

        local_infos = {}
        for c in self.cache_candidates:
            local_infos.update(self._collect_local_infos(hf_dataset, c))
        hub_infos = {}
        with ThreadPoolExecutor(max_workers=1) as ex:
            try:
                hub_infos = ex.submit(self._collect_hub_infos, hf_dataset).result(timeout=3)
            except TimeoutError:
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

        if not datasets_ok:
            configs = list({c for c in merged_infos.keys() if c})
            configs.sort()
            target_config = requested_subset if requested_subset and requested_subset in configs else (configs[0] if configs else None)
            splits = merged_infos.get(target_config) or []
            if not splits and repo_default_splits:
                splits = repo_default_splits
            if not splits:
                splits = ["train", "validation", "test"]
            return {
                "status": "success",
                "hf_dataset": hf_dataset,
                "configs": configs or [],
                "splits": splits,
            }

        configs = []
        splits = []
        with ThreadPoolExecutor(max_workers=2) as ex:
            try:
                configs = ex.submit(_get_configs).result(timeout=5)
            except TimeoutError:
                configs = []
            try:
                splits = ex.submit(_load_ds).result(timeout=5)
            except TimeoutError:
                splits = []

        configs_set = set([c for c in (configs or []) if c])
        configs_set.update(merged_infos.keys())
        configs = list(configs_set)
        configs.sort()

        target_config = requested_subset if requested_subset and requested_subset in configs_set else (configs[0] if configs else None)

        if configs:
            splits = merged_infos.get(target_config) or splits
            if not splits:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    try:
                        splits = ex.submit(_get_splits_from_builder, target_config).result(timeout=5)
                    except TimeoutError:
                        splits = []
        else:
            if not splits:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    try:
                        splits = ex.submit(_get_splits_from_builder, None).result(timeout=5)
                    except TimeoutError:
                        splits = []

        if not splits and repo_default_splits:
            splits = repo_default_splits

        if not splits:
            splits = ["train", "validation", "test"]

        return {
            "status": "success",
            "hf_dataset": hf_dataset,
            "configs": configs or [],
            "splits": splits,
        }
