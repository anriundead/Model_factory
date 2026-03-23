#!/usr/bin/env python3
"""
清理标记为 Discard 的模型：仅删除权重文件（.safetensors / .bin），保留 metadata、config、评估记录等。
默认仅做 Dry Run（打印将删除的路径，不实际删除）；加 --execute 才执行删除。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    execute = "--execute" in sys.argv or "-x" in sys.argv
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
        paths = services.list_models_with_tag("Discard")
    if not paths:
        print("没有带 Discard 标签的模型，无需清理。")
        return 0
    print("以下 %d 个模型带有 Discard 标签：" % len(paths))
    for p in paths:
        print("  ", p)
    if not execute:
        print("\n当前为 Dry Run，未删除任何文件。若要执行删除，请加参数: --execute 或 -x")
        return 0
    removed_count = 0
    for model_path in paths:
        if not model_path or not os.path.isdir(model_path):
            continue
        try:
            for f in os.listdir(model_path):
                if f.endswith(".safetensors") or f.endswith(".bin"):
                    fp = os.path.join(model_path, f)
                    os.remove(fp)
                    removed_count += 1
                    print("已删除:", fp)
        except OSError as e:
            print("删除失败 %s: %s" % (model_path, e))
    print("共删除 %d 个权重文件。" % removed_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
