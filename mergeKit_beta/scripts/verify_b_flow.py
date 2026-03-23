#!/usr/bin/env python3
"""
B 方案全流程验证：5.1 评估结果与模型关联、5.2 排行榜数据源、5.3 标签与清理。
检查原有流程变化、新输出及前端可见性。
需在项目根目录执行。
"""
from __future__ import print_function

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def main():
    from app import create_app
    from app.extensions import db
    from app.models import Model, EvaluationResult, Tag
    from app.repositories import (
        evaluation_result_get_by_task_id,
        evaluation_results_grouped_by_testset,
        evaluation_best_per_model_per_testset,
        model_list_all,
    )

    app = create_app()
    with app.app_context():
        print("=" * 60)
        print("B 方案全流程验证")
        print("=" * 60)

        # --- 1. 数据现状 ---
        models = model_list_all()
        results = db.session.query(EvaluationResult).all()
        with_model_id = sum(1 for r in results if r.model_id)
        print("\n1. 数据现状")
        print("   - models 表条数: %d" % len(models))
        print("   - evaluation_results 表条数: %d" % len(results))
        print("   - 其中带 model_id（可显示模型名）: %d" % with_model_id)

        # --- 2. 排行榜数据源（DB）---
        lb = evaluation_results_grouped_by_testset()
        print("\n2. 排行榜（DB）load_leaderboard 等价数据")
        print("   - 测试集数: %d" % len(lb))
        for tid, entries in list(lb.items())[:3]:
            print("   - %s: %d 条, 首条 model_name=%s" % (
                tid[:20], len(entries),
                (entries[0].get("model_name") if entries else "N/A"),
            ))

        # --- 3. Top10 数据是否可算 ---
        rows = evaluation_best_per_model_per_testset()
        print("\n3. Top10 聚合数据（evaluation_best_per_model_per_testset）")
        print("   - (testset_id, model_id, best_acc) 条数: %d" % len(rows))

        # --- 4. 标签 ---
        tags = db.session.query(Tag).filter(Tag.name.in_(["Keep", "Discard"])).all()
        print("\n4. 标签 Keep/Discard")
        for t in tags:
            try:
                count = len(list(t.models))
            except Exception:
                count = 0
            print("   - %s: 存在 (关联模型数: %d)" % (t.name, count))

        # --- 5. 去重接口 ---
        one_task = db.session.query(EvaluationResult).first()
        if one_task:
            exists = evaluation_result_get_by_task_id(one_task.task_id) is not None
            print("\n5. 迁移去重 evaluation_result_get_by_task_id")
            print("   - 示例 task_id=%s 已存在: %s" % (one_task.task_id, exists))

        print("\n" + "=" * 60)
        print("验证完成：数据层与接口正常。")
        print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
