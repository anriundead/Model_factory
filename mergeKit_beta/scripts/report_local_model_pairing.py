#!/usr/bin/env python3
"""
本地基座/模型池模型的「可融合配对」报告。

规则与后端一致（见 app/services.py:check_merge_compatible）：
- 读取各目录下 config.json，取 hidden_size、num_hidden_layers（含 text_config 等回退）；
- 两模型可配对当且仅当 (hidden_size, num_hidden_layers) 完全相同；
- 另标注 is_vlm（与 model_is_vlm 一致）：同架构下优先「同为 VLM」或「同为非 VLM」配对。

可选：扫描 recipes/*.json 中的 model_paths，将路径解析到当前 LOCAL_MODELS_PATH 后做兼容性说明。

用法（建议在容器内或与运行环境同路径）:
  python3 scripts/report_local_model_pairing.py
  python3 scripts/report_local_model_pairing.py --no-recipes
  python3 scripts/report_local_model_pairing.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 须在 import app.* 之前设置，避免 app/__init__.py 自动 create_app() 与启动 Worker
os.environ["MERGEKIT_CLI_SCRIPT"] = "1"
from itertools import combinations
from pathlib import Path

# 项目根在脚本上级目录的父级
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import logging

from config import Config  # noqa: E402
from app.services import Services  # noqa: E402


class _MiniState:
    """避免 import app.state 触发 app/__init__.py 里的 create_app() 与 Worker。"""

    config = Config
    logger = logging.getLogger("report_local_model_pairing")
    merge_dir = Config.MERGE_DIR
    model_pool_path = Config.MODEL_POOL_PATH
    recipes_dir = Config.RECIPES_DIR

    @property
    def project_root(self):
        return Config.PROJECT_ROOT


def _collect_model_entries(services: Services) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    bases = []
    lp = getattr(services.config, "LOCAL_MODELS_PATH", None)
    if lp:
        bases.append(lp)
    mp = getattr(services.state, "model_pool_path", None)
    if mp and mp not in bases:
        bases.append(mp)
    for extra in getattr(services.config, "LOCAL_MODELS_EXTRA_PATHS", None) or []:
        if extra and extra not in bases:
            bases.append(extra)
    for base in bases:
        if not base or not os.path.isdir(base):
            continue
        for m in services.list_models_from_dir(base):
            p = (m.get("path") or "").strip()
            if not p or p in seen:
                continue
            seen.add(p)
            out.append({"name": m.get("name") or os.path.basename(p), "path": p})
    return out


def _resolve_recipe_path(services: Services, raw: str) -> str | None:
    """配方里常为旧绝对路径，尽量解析到现存目录。"""
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if os.path.isdir(s):
        return os.path.abspath(s)
    r = services.resolve_model_path(s)
    if r and os.path.isdir(r):
        return r
    base = os.path.basename(s.rstrip("/"))
    lp = getattr(services.config, "LOCAL_MODELS_PATH", None)
    if lp:
        cand = os.path.join(lp, base)
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--no-recipes", action="store_true", help="不扫描 recipes")
    args = ap.parse_args()

    Config.setup_environment()
    services = Services(_MiniState())

    entries = _collect_model_entries(services)
    enriched: list[dict] = []
    unreadable: list[dict] = []

    for e in entries:
        p = e["path"]
        hs, nhl = services.get_model_arch(p)
        mt = services.get_model_type(p) or ""
        vlm = services.model_is_vlm(p)
        if hs is None or nhl is None:
            unreadable.append({**e, "model_type": mt or None, "is_vlm": vlm})
            continue
        enriched.append(
            {
                **e,
                "hidden_size": hs,
                "num_hidden_layers": nhl,
                "model_type": mt or None,
                "is_vlm": vlm,
                "arch_key": (hs, nhl),
            }
        )

    # 按架构分组
    from collections import defaultdict

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for m in enriched:
        groups[m["arch_key"]].append(m)

    recipe_report: list[dict] = []
    if not args.no_recipes:
        rd = getattr(services.state, "recipes_dir", None) or Config.RECIPES_DIR
        if os.path.isdir(rd):
            for fn in sorted(os.listdir(rd)):
                if not fn.endswith(".json"):
                    continue
                fp = os.path.join(rd, fn)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                paths_raw = data.get("model_paths") or []
                resolved = []
                for pr in paths_raw:
                    rp = _resolve_recipe_path(services, pr)
                    resolved.append({"raw": pr, "resolved": rp})
                ok_paths = [x["resolved"] for x in resolved if x["resolved"]]
                compat = None
                msg = ""
                if len(ok_paths) >= 2:
                    c_ok, msg, _ = services.check_merge_compatible(ok_paths)
                    compat = c_ok
                elif len(ok_paths) < 2:
                    msg = "无法解析到本地目录的路径不足 2 个"
                    compat = False
                else:
                    compat = True
                recipe_report.append(
                    {
                        "file": fn,
                        "custom_name": data.get("custom_name"),
                        "task_id": data.get("task_id"),
                        "paths": resolved,
                        "check_merge_compatible": compat,
                        "message": msg,
                    }
                )

    if args.json:
        out = {
            "local_models_path": getattr(Config, "LOCAL_MODELS_PATH", None),
            "model_pool_path": getattr(services.state, "model_pool_path", None),
            "scanned_count": len(entries),
            "readable": len(enriched),
            "unreadable": unreadable,
            "groups": {
                f"hs{k[0]}_layers{k[1]}": [
                    {kk: m[kk] for kk in ("name", "path", "model_type", "is_vlm")}
                    for m in sorted(v, key=lambda x: x["name"].lower())
                ]
                for k, v in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1]))
            },
            "pairwise_by_arch": {},
            "recipes": recipe_report,
        }
        # 配对列表（同架构内全组合）
        for k, v in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
            if len(v) < 2:
                continue
            names = sorted(v, key=lambda x: x["name"].lower())
            pairs = []
            for a, b in combinations(names, 2):
                same_kind = a["is_vlm"] == b["is_vlm"]
                pairs.append(
                    {
                        "a": a["name"],
                        "b": b["name"],
                        "same_vlm_class": same_kind,
                    }
                )
            out["pairwise_by_arch"][f"hs{k[0]}_layers{k[1]}"] = pairs
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # 人类可读
    print("=== 本地模型可融合配对报告 ===")
    print("LOCAL_MODELS_PATH:", getattr(Config, "LOCAL_MODELS_PATH", None))
    print("MODEL_POOL_PATH:", getattr(services.state, "model_pool_path", None))
    print("扫描到模型目录数:", len(entries), "| 可解析架构:", len(enriched), "| 无法解析 config:", len(unreadable))
    print()

    if unreadable:
        print("--- 无法读取 hidden_size/num_hidden_layers 的目录 ---")
        for u in unreadable:
            print(f"  {u['name']}\n    path: {u['path']}")
        print()

    print("说明：两模型可执行标准融合（与 POST /api/check_compatibility 一致）当且仅当在同一「架构组」内。")
    print("建议：同组内优先选择 is_vlm 相同的两个模型；VLM 与纯文本模型同层数同宽度也可能 keys 不兼容，需以实际融合报错为准。")
    print()

    for arch_key in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        members = sorted(groups[arch_key], key=lambda x: x["name"].lower())
        hs, nhl = arch_key
        print(f"--- 架构组 hidden_size={hs}, num_hidden_layers={nhl} （共 {len(members)} 个）---")
        for m in members:
            tag = "VLM" if m["is_vlm"] else "LLM"
            print(f"  [{tag}] {m['name']}")
            print(f"       type={m['model_type']!r}")
            print(f"       path={m['path']}")
        if len(members) >= 2:
            print("  可两两配对（无序）:")
            for a, b in combinations(members, 2):
                note = "" if a["is_vlm"] == b["is_vlm"] else "  [注意: VLM/非VLM 混选]"
                print(f"    • {a['name']}  +  {b['name']}{note}")
        else:
            print("  （组内不足 2 个，无配对）")
        print()

    if not args.no_recipes and recipe_report:
        print("=== 配方文件 model_paths（解析后兼容性）===")
        for r in recipe_report:
            print(f"  {r['file']}  task={r.get('task_id')}  name={r.get('custom_name')!r}")
            for pr in r["paths"]:
                print(f"    raw: {pr['raw']}")
                print(f"    ->  {pr['resolved'] or '(未解析到本地目录)'}")
            flag = "OK" if r["check_merge_compatible"] else "NO"
            print(f"    check_merge_compatible: {flag}  {r.get('message') or ''}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
