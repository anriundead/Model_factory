#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 拓扑与占用检测（最小依赖版）

设计目标：
- 只解决“任务选择哪些 GPU/卡对可用”的问题，不掺入业务逻辑。
- 通过 nvidia-smi 查询 free memory，做最小可行的可用性判断（避免被常驻进程挤占导致 OOM）。
- 为 NVLink 对内 TP=2 提供“卡对 (0,1)/(2,3)”选择与降级。

注意：
- 这是一个启发式选择器：以“空闲显存阈值”为主，避免复杂的进程枚举/白名单逻辑。
- 阈值与拓扑均可通过环境变量覆盖；调用方负责记录最终选择结果到日志/元数据。
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GpuInfo:
    index: int
    mem_free_mib: int
    mem_total_mib: int


def _run(cmd: list[str], timeout_s: int = 5) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_s)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {cmd!r}: {p.stderr.strip()}")
    return p.stdout


def query_gpus() -> list[GpuInfo]:
    """
    返回每张卡的 (index, free_mib, total_mib)。
    """
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=5,
    )
    gpus: list[GpuInfo] = []
    for line in out.splitlines():
        line = (line or "").strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            if any(re.fullmatch(r"[0-9]+", value) is None for value in parts[:3]):
                raise ValueError("not an ASCII decimal integer")
            idx = int(parts[0])
            free = int(parts[1])
            total = int(parts[2])
        except ValueError:
            continue
        gpus.append(GpuInfo(index=idx, mem_free_mib=free, mem_total_mib=total))
    gpus.sort(key=lambda x: x.index)
    return gpus


def parse_nvlink_pairs(spec: str | None) -> list[tuple[int, int]]:
    """
    spec 示例：
    - "01,23"（默认）
    - "0-1,2-3"
    - "2,3"（会按相邻两两配对：这里会被视为非法，调用方应提供成对格式）
    """
    s = (spec or "").strip()
    if not s:
        s = "01,23"
    pairs: list[tuple[int, int]] = []
    for token in s.split(","):
        t = token.strip()
        if not t:
            continue
        if "-" in t:
            a, b = [x.strip() for x in t.split("-", 1)]
        else:
            # 允许 "01" / "23" 这种紧凑写法
            if len(t) != 2 or not t.isdigit():
                raise ValueError(f"invalid pair token: {t!r}")
            a, b = t[0], t[1]
        pairs.append((int(a), int(b)))
    # 去重但保序
    seen = set()
    out: list[tuple[int, int]] = []
    for a, b in pairs:
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def select_free_pairs(
    *,
    pairs: Iterable[tuple[int, int]],
    min_free_mib: int,
) -> list[tuple[int, int]]:
    """
    从给定 pair 列表中筛选“双方 free_mib 都 >= min_free_mib”的卡对。
    """
    gpus = {g.index: g for g in query_gpus()}
    ok: list[tuple[int, int]] = []
    for a, b in pairs:
        ga = gpus.get(a)
        gb = gpus.get(b)
        if not ga or not gb:
            continue
        if ga.mem_free_mib >= min_free_mib and gb.mem_free_mib >= min_free_mib:
            ok.append((a, b))
    return ok


def env_int(name: str, default: int) -> int:
    v = (os.environ.get(name) or "").strip()
    if not v:
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    v = (os.environ.get(name) or "").strip()
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default
