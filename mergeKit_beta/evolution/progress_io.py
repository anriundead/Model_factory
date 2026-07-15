# -*- coding: utf-8 -*-
"""进化任务 progress.json 读写辅助（无 Flask 依赖，供 Runner 与 Worker 共用）。"""

from __future__ import annotations

import json
import os
import time
from typing import Any

# 失败落盘时尽量保留的字段（与 read_evolution_progress / 前端展示对齐）
_KEEP_ON_ERROR = (
    "current_step",
    "total_expected_steps",
    "current_best",
    "global_best",
    "best_genotype",
    "percent",
    "step",
    "eta_seconds",
    "estimated_completion",
    "gen_durations",
    "total_task_time",
    "total_merge_time",
    "total_eval_time",
    "total_overhead_time",
    "avg_step_time",
    "avg_merge_time",
    "avg_eval_time",
)


def merge_error_payload(existing: dict[str, Any], error_message: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """在已有 progress 字典上合并 error 终态，保留步数/best/percent 等。"""
    out: dict[str, Any] = {}
    for k in _KEEP_ON_ERROR:
        if k in existing:
            out[k] = existing[k]
    msg = (error_message or "").strip() or "进化任务失败"
    out["status"] = "error"
    out["error_detail"] = msg
    out["message"] = msg[:2000]
    out["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if extra:
        out.update(extra)
    return out


def write_progress_error(progress_path: str, error_message: str, extra: dict[str, Any] | None = None) -> None:
    """读取现有 progress.json（若存在），合并 error 字段后原子写入。"""
    existing: dict[str, Any] = {}
    if os.path.isfile(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            pass
    payload = merge_error_payload(existing, error_message, extra)
    tmp = progress_path + ".tmp"
    parent = os.path.dirname(progress_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, progress_path)
