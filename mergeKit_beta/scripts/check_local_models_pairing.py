#!/usr/bin/env python3
"""
配对检测：对比 API 返回的模型路径与磁盘实际目录是否一致。
在容器内或与 API 同机执行（路径需与 LOCAL_MODELS_PATH / merges 一致）。

用法:
  python3 scripts/check_local_models_pairing.py
  python3 scripts/check_local_models_pairing.py --json   # 仅输出 JSON 摘要
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "check_local_models_pairing/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _path_ok(p: str) -> tuple[bool, str]:
    if not p or not str(p).strip():
        return False, "empty"
    p = os.path.realpath(os.path.abspath(str(p).strip()))
    return os.path.isdir(p), p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("MERGEKIT_API", "http://127.0.0.1:5000"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    out: dict = {"api_models": {}, "model_repo": {}, "disk_orphans": {}}

    try:
        dm = _get(f"{base}/api/models")
    except urllib.error.URLError as e:
        print(f"无法访问 {base}/api/models: {e}", file=sys.stderr)
        return 2

    mm = []
    mo = []
    for m in dm.get("models") or []:
        name = m.get("name") or "?"
        path = (m.get("path") or "").strip()
        ok, rp = _path_ok(path)
        if ok:
            mo.append({"name": name, "path": rp})
        else:
            mm.append({"name": name, "path": path, "resolved": rp})
    out["api_models"] = {
        "base_path": dm.get("base_path"),
        "listed": len(dm.get("models") or []),
        "on_disk": len(mo),
        "missing": mm,
    }

    try:
        dr = _get(f"{base}/api/model_repo/list")
    except urllib.error.URLError as e:
        print(f"无法访问 {base}/api/model_repo/list: {e}", file=sys.stderr)
        return 2

    for label in ("base_models", "merged_models"):
        mo, mm = [], []
        for m in dr.get(label) or []:
            path = (m.get("path") or "").strip()
            name = m.get("name") or m.get("id") or m.get("custom_name") or "?"
            ok, rp = _path_ok(path)
            if ok:
                mo.append({"name": name, "path": rp})
            else:
                mm.append({"name": name, "path": path, "resolved": rp})
        out["model_repo"][label] = {
            "listed": len(dr.get(label) or []),
            "on_disk": len(mo),
            "missing": mm,
        }

    # 基座：磁盘上有目录但未出现在 base_models 名称集合（可选）
    bp = (dm.get("base_path") or "").strip()
    if bp and os.path.isdir(bp):
        try:
            disk_names = sorted(
                x for x in os.listdir(bp) if os.path.isdir(os.path.join(bp, x))
            )
        except OSError:
            disk_names = []
        api_names = {m.get("name") for m in (dr.get("base_models") or []) if m.get("name")}
        orphan = [n for n in disk_names if n not in api_names]
        out["disk_orphans"] = {"base_path": bp, "dirs_not_in_api_names": orphan}

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print("=== GET /api/models ===")
    print("base_path:", out["api_models"]["base_path"])
    print(
        "listed:",
        out["api_models"]["listed"],
        "on_disk:",
        out["api_models"]["on_disk"],
        "missing:",
        len(out["api_models"]["missing"]),
    )
    for row in out["api_models"]["missing"][:80]:
        print("  MISSING", row["name"], "->", row["path"])

    for label in ("base_models", "merged_models"):
        block = out["model_repo"][label]
        print(f"=== GET /api/model_repo/list [{label}] ===")
        print(
            "listed:",
            block["listed"],
            "on_disk:",
            block["on_disk"],
            "missing:",
            len(block["missing"]),
        )
        for row in block["missing"][:80]:
            print("  MISSING", row["name"], "->", row["path"])

    if out["disk_orphans"].get("dirs_not_in_api_names"):
        print("=== 磁盘上存在、名称未出现在 base_models ===")
        for n in out["disk_orphans"]["dirs_not_in_api_names"][:40]:
            print("  ORPHAN_DIR", n)

    return 0 if not (
        out["api_models"]["missing"]
        or out["model_repo"]["base_models"]["missing"]
        or out["model_repo"]["merged_models"]["missing"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
