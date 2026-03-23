#!/usr/bin/env python3
"""
一次性脚本：将 testset_repo/data/leaderboard.json 中的历史数据迁移到 DB（EvaluationResult + TestSet）。
用于在「以 DB 为准」之前补齐已有文件数据；重复执行会跳过已存在 task_id 的记录。
需在项目根目录执行，且需有 Flask app 上下文（见下方 __main__）。
"""
import json
import os
import sys

# 项目根目录
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    lb_path = os.path.join(_ROOT, "testset_repo", "data", "leaderboard.json")
    if not os.path.isfile(lb_path):
        print("未找到 leaderboard.json，跳过迁移。路径: %s" % lb_path)
        return 0

    from app import create_app
    from app.extensions import db
    from app.repositories import (
        testset_upsert,
        evaluation_result_insert,
        evaluation_result_get_by_task_id,
        model_list_all,
    )

    app = create_app()
    with app.app_context():
        with open(lb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        leaderboards = data.get("leaderboards") or {}
        if not isinstance(leaderboards, dict):
            print("leaderboards 格式异常，跳过")
            return 0

        models_by_name = {}
        for m in model_list_all():
            name = (m.get("name") or "").strip()
            if name:
                models_by_name.setdefault(name, []).append(m.get("id"))

        inserted = 0
        skipped = 0
        for testset_id, entries in leaderboards.items():
            if not testset_id or not isinstance(entries, list):
                continue
            testset_upsert(
                testset_id=testset_id,
                name=testset_id,
                hf_dataset=None,
                hf_subset=None,
                sample_count=0,
                is_local=False,
            )
            for entry in entries:
                task_id = entry.get("task_id")
                if not task_id:
                    skipped += 1
                    continue
                if evaluation_result_get_by_task_id(task_id) is not None:
                    skipped += 1
                    continue
                model_name = (entry.get("model_name") or "").strip()
                model_id = None
                if model_name and model_name in models_by_name:
                    ids = models_by_name[model_name]
                    model_id = ids[0] if ids else None
                try:
                    evaluation_result_insert(
                        task_id=task_id,
                        testset_id=testset_id,
                        model_id=model_id,
                        accuracy=entry.get("accuracy"),
                        f1_score=entry.get("f1_score"),
                        test_cases=entry.get("test_cases"),
                        context=entry.get("context") if isinstance(entry.get("context"), (int, float)) else None,
                        throughput=entry.get("throughput"),
                        time=entry.get("time"),
                        metrics=entry,
                    )
                    inserted += 1
                except Exception as e:
                    print("插入 task_id=%s 失败: %s" % (task_id, e))
                    skipped += 1

        print("迁移完成: 插入 %d 条，跳过 %d 条。" % (inserted, skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
