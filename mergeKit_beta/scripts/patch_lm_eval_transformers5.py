#!/usr/bin/env python3
"""
（已废弃）lm_eval >= 0.4.11 已原生使用 AutoModelForImageTextToText，不再需要此补丁。
保留此脚本仅供回退到旧版 lm_eval 时使用。
"""
import os
import sys

for p in sys.path:
    path = os.path.join(p, "lm_eval", "models", "hf_vlms.py")
    if os.path.isfile(path):
        break
else:
    print("未找到 lm_eval/models/hf_vlms.py，请确认在 mergenetic 环境中运行")
    sys.exit(1)

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

if "AutoModelForImageTextToText" in content:
    print("当前 lm_eval 已原生兼容 transformers 5.x，无需打补丁:", path)
    sys.exit(0)

old = "AUTO_MODEL_CLASS = transformers.AutoModelForVision2Seq"
new = "AUTO_MODEL_CLASS = getattr(transformers, 'AutoModelForVision2Seq', None) or getattr(transformers, 'AutoModelForImageTextToText')"

if old not in content:
    print("未找到预期代码行，请检查 lm_eval 版本。当前文件:", path)
    sys.exit(1)

content = content.replace(old, new)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("已打补丁:", path)
