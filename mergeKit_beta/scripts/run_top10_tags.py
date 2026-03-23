#!/usr/bin/env python3
"""
根据评估结果计算每测试集内 Top 10% 模型并打标签 Keep / Discard。
需在项目根目录执行；依赖 Flask app 与 DB。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    top_percent = 10
    if len(sys.argv) > 1:
        try:
            top_percent = int(sys.argv[1])
        except ValueError:
            pass
    from app import create_app
    from app.services import Services
    from app.state import AppState

    app = create_app()
    with app.app_context():
        state = AppState()
        state.config = app.config
        state.logger = app.logger
        services = Services(state)
        services.app = app
        result = services.compute_top10_tags(top_percent=top_percent)
    if result.get("error"):
        print("失败:", result["error"])
        return 1
    print("Top %d%% 标签已更新: Keep=%d, Discard=%d" % (top_percent, result.get("keep", 0), result.get("discard", 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
