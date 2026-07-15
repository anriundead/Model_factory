#!/usr/bin/env python3
"""
limit 计划（limit_语义与多卡修复）§4.1～§4.2 自查：无 GPU 亦可运行。
§4.3：命令级烟测由 tests/test_limit_plan_smoke_cmds.py（mock Popen）覆盖；
  真实 accelerate + Hub 仍建议在目标机再跑一条。
"""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    r = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_eval_limit_resolution", "-v"],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        return r.returncode

    r2 = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_limit_plan_smoke_cmds", "-v"],
        cwd=str(ROOT),
    )
    if r2.returncode != 0:
        return r2.returncode

    path = ROOT / "merge_manager.py"
    text = path.read_text(encoding="utf-8")
    if "lim >= 1 and lim < actual_gpus" in text:
        print("FAIL: merge_manager.py 仍含旧条件 lim >= 1 and lim < actual_gpus", file=sys.stderr)
        return 1

    required = (
        "_should_fallback_single_gpu_for_limit",
        "_omit_limit_full",
        "load_dataset 换算 limit 跳过",
    )
    for needle in required:
        if needle not in text:
            print(f"FAIL: merge_manager.py 缺少预期片段: {needle!r}", file=sys.stderr)
            return 1

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rec = log_dir / "limit_plan_verify_last.txt"
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    rec.write_text(
        "%s\n"
        "tests.test_eval_limit_resolution: OK\n"
        "tests.test_limit_plan_smoke_cmds (§4.3 cmd mock): OK\n"
        "merge_manager 静态 grep: OK\n"
        "说明: CMMMU 比例「1.0」本地分支使用 _resolve_eval_dataset_cap，与 test_eval_limit_resolution 一致。\n"
        % stamp,
        encoding="utf-8",
    )
    print("OK: limit 计划 §4.1～§4.2 + §4.3（命令 mock）通过")
    print("记录已写入: %s" % rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
