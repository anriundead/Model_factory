"""
路径解析：配置文件存前缀（相对 PROJECT_ROOT），代码侧使用相对名或 legacy 绝对路径，运行时拼接为当前环境的绝对路径。
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PATHS_FILE = _PROJECT_ROOT / "config" / "paths.json"


def project_root() -> Path:
    return _PROJECT_ROOT


@lru_cache(maxsize=1)
def load_paths_config() -> dict:
    if not _PATHS_FILE.is_file():
        return {}
    with open(_PATHS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _resolve_config_dir(relative_or_abs: str) -> str:
    s = (relative_or_abs or "").strip()
    if not s:
        return ""
    p = Path(s)
    if p.is_absolute():
        return str(p.resolve())
    return str((_PROJECT_ROOT / p).resolve())


def configured_local_models_path() -> str:
    env = (os.environ.get("LOCAL_MODELS_PATH") or "").strip()
    if env:
        return os.path.abspath(env)
    cfg = load_paths_config()
    return _resolve_config_dir(cfg.get("local_models_dir", ""))


def configured_model_pool_path() -> str:
    env = (os.environ.get("MERGEKIT_MODEL_POOL") or "").strip()
    if env:
        return os.path.abspath(env)
    cfg = load_paths_config()
    rel = cfg.get("model_pool_dir")
    if rel:
        return _resolve_config_dir(rel)
    return os.path.abspath(os.path.join(str(_PROJECT_ROOT), "..", "mergeKit", "models_pool"))


def configured_merge_dir() -> str:
    env = (os.environ.get("MERGEKIT_MERGE_DIR") or "").strip()
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(str(_PROJECT_ROOT), "merges"))


def _norm_prefix(prefix: str) -> str:
    return os.path.abspath((prefix or "").rstrip("/\\"))


def legacy_prefixes(key: str) -> list[str]:
    cfg = load_paths_config()
    raw = cfg.get(key) or []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(_norm_prefix(item.strip()))
    return out


def all_legacy_prefixes() -> list[str]:
    keys = (
        "legacy_local_models_prefixes",
        "legacy_model_pool_prefixes",
        "legacy_merge_prefixes",
    )
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        for prefix in legacy_prefixes(key):
            if prefix not in seen:
                seen.add(prefix)
                out.append(prefix)
    for current in (
        configured_local_models_path(),
        configured_model_pool_path(),
        configured_merge_dir(),
        "/data/Models",
        "/data/models_pool",
    ):
        p = _norm_prefix(current)
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def strip_legacy_prefix(path: str) -> str | None:
    """将 legacy 绝对路径转为相对片段（模型名或 merges 下相对路径）。"""
    if not path or not isinstance(path, str):
        return None
    s = path.strip().replace("\\", "/")
    if not s:
        return None
    abs_path = _norm_prefix(s)
    for prefix in sorted(all_legacy_prefixes(), key=len, reverse=True):
        pref = prefix.replace("\\", "/")
        if abs_path == prefix:
            return ""
        candidate = pref + "/"
        if abs_path.startswith(candidate):
            rel = abs_path[len(prefix) :].lstrip("/\\")
            return rel or None
    if os.path.isabs(s):
        return os.path.basename(abs_path)
    return s.lstrip("/\\")


def join_under_base(base: str, relative_part: str) -> str:
    base = (base or "").rstrip("/\\")
    rel = (relative_part or "").strip().strip("/\\")
    if not base:
        return ""
    if not rel:
        return base
    return os.path.abspath(os.path.join(base, rel.replace("/", os.sep)))


def candidate_bases(extra_bases: Iterable[str] | None = None) -> list[str]:
    bases = [
        configured_local_models_path(),
        configured_model_pool_path(),
        configured_merge_dir(),
    ]
    if extra_bases:
        bases.extend(extra_bases)
    seen: set[str] = set()
    out: list[str] = []
    for b in bases:
        if not b:
            continue
        ab = _norm_prefix(b)
        if ab not in seen:
            seen.add(ab)
            out.append(ab)
    return out


def resolve_to_absolute(name_or_path: str, extra_bases: Iterable[str] | None = None) -> str | None:
    """
    将模型名、相对路径或 legacy 绝对路径解析为当前环境可用的绝对路径（不校验目录完整性）。
    """
    if not name_or_path or not isinstance(name_or_path, str):
        return None
    s = name_or_path.strip()
    if not s:
        return None

    bases = candidate_bases(extra_bases)

    if os.path.isabs(s) and os.path.isdir(s):
        return os.path.abspath(s)

    relative = strip_legacy_prefix(s)
    if relative is None:
        return None

    name = os.path.basename(relative.replace("\\", "/")) if relative else os.path.basename(s)
    search_parts = []
    if relative:
        search_parts.append(relative)
    if name and name not in search_parts:
        search_parts.append(name)

    for base in bases:
        if not base or not os.path.isdir(base):
            continue
        for part in search_parts:
            candidate = join_under_base(base, part)
            if os.path.isdir(candidate):
                return candidate
        if name:
            try:
                for root, dirs, _files in os.walk(base):
                    depth = root[len(base) :].count(os.sep)
                    if depth > 4:
                        dirs[:] = []
                        continue
                    if name in dirs:
                        found = os.path.join(root, name)
                        if os.path.isdir(found):
                            return os.path.abspath(found)
            except OSError:
                pass
    return None
