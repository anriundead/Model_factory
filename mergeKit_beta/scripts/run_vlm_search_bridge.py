#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容入口：历史脚本路径。实现已迁移至 `evolution.runner`。

设置 `MERGEKIT_EVOLUTION_LEGACY_BRIDGE=1` 时 Worker 会优先使用本脚本路径；
默认 Worker 使用 `python -m evolution.runner`。
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evolution.runner import main

if __name__ == "__main__":
    main()
