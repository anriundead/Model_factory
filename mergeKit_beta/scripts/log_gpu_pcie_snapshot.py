#!/usr/bin/env python3
"""Print JSON snapshot of PCIe link sysfs + nvidia-smi -L; optional append NDJSON to -o."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

# 按机器调整，或通过环境变量 MERGEKIT_PCIE_BUSSES 逗号分隔传入
_DEFAULT_BUSSES = (
    "0000:01:00.0,0000:41:00.0,0000:81:00.0,0000:c1:00.0"
)


def _read_text(p: Path) -> str:
    try:
        return p.read_text().strip()
    except OSError:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        help="若指定，将一条 NDJSON 追加写入该文件",
    )
    args = ap.parse_args()

    raw = (os.environ.get("MERGEKIT_PCIE_BUSSES") or _DEFAULT_BUSSES).strip()
    busses = [b.strip() for b in raw.split(",") if b.strip()]

    ts = int(time.time() * 1000)
    buses: dict = {}
    for bus in busses:
        base = Path(f"/sys/bus/pci/devices/{bus}")
        spd = _read_text(base / "current_link_speed")
        wid = _read_text(base / "current_link_width")
        buses[bus] = {
            "link_speed": spd,
            "link_width": wid,
            "broken_sysfs": spd == "Unknown" or wid in ("", "63"),
        }

    smi = ""
    try:
        smi = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        smi = f"<nvidia-smi failed: {e}>"

    smi_errors = re.findall(
        r"Unable to determine the device handle for gpu (0000:[0-9a-f]{2}:[0-9a-f]{2}\.\d)",
        smi,
        re.I,
    )

    data = {
        "buses": buses,
        "nvidia_smi_L_excerpt": smi[:2000],
        "smi_bus_errors": smi_errors,
        "any_sysfs_broken": any(b.get("broken_sysfs") for b in buses.values()),
    }
    print(json.dumps(data, indent=2, ensure_ascii=False))

    if args.output:
        line = {
            "location": "scripts/log_gpu_pcie_snapshot.py",
            "message": "gpu_pcie_nv_snapshot",
            "data": data,
            "timestamp": ts,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
