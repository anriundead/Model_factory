#!/usr/bin/env python3
"""
DB vs 文件一致性校验脚本。
在 mergeKit_beta 根目录运行：python scripts/check_db_file_consistency.py
需要 Flask app context（自动创建）。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

PROJECT_ROOT = Config.PROJECT_ROOT
TESTSETS_JSON = os.path.join(PROJECT_ROOT, "testset_repo", "data", "testsets.json")
LEADERBOARD_JSON = os.path.join(PROJECT_ROOT, "testset_repo", "data", "leaderboard.json")
MODELS_JSON = os.path.join(PROJECT_ROOT, "model_repo", "data", "models.json")
MERGES_DIR = os.path.join(PROJECT_ROOT, "merges")


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(items, limit=10):
    if not items:
        return "  (无)"
    lines = []
    for i, item in enumerate(items[:limit]):
        lines.append(f"  - {item}")
    if len(items) > limit:
        lines.append(f"  ... 共 {len(items)} 条，仅显示前 {limit} 条")
    return "\n".join(lines)


def check_testsets(app):
    print("\n" + "=" * 60)
    print("1. TestSet: testsets.json vs DB")
    print("=" * 60)
    from app.models import TestSet
    from app.extensions import db

    file_data = _load_json(TESTSETS_JSON)
    if file_data is None:
        print("  [SKIP] testsets.json 不存在")
        return
    file_testsets = file_data.get("testsets", {})
    print(f"  文件中: {len(file_testsets)} 条")

    with app.app_context():
        db_rows = db.session.query(TestSet).all()
        db_map = {r.id: r for r in db_rows}
        print(f"  DB  中: {len(db_map)} 条")

        file_only = [k for k in file_testsets if k not in db_map]
        db_only = [k for k in db_map if k not in file_testsets]
        both = [k for k in file_testsets if k in db_map]

        field_mismatches = []
        for k in both:
            f = file_testsets[k]
            d = db_map[k]
            diffs = []
            for field in ("name", "hf_dataset", "hf_subset", "hf_split", "lm_eval_task", "sample_count"):
                fv = f.get(field)
                dv = getattr(d, field, None)
                if str(fv or "") != str(dv or ""):
                    diffs.append(f"{field}: file={fv!r} db={dv!r}")
            if diffs:
                field_mismatches.append(f"{k}: {'; '.join(diffs)}")

        print(f"\n  仅文件有 ({len(file_only)}):")
        print(_fmt(file_only))
        print(f"\n  仅DB有 ({len(db_only)}):")
        print(_fmt(db_only))
        print(f"\n  字段不一致 ({len(field_mismatches)}):")
        print(_fmt(field_mismatches))


def check_leaderboard(app):
    print("\n" + "=" * 60)
    print("2. EvaluationResult: leaderboard.json vs DB")
    print("=" * 60)
    from app.models import EvaluationResult
    from app.extensions import db

    file_data = _load_json(LEADERBOARD_JSON)
    if file_data is None:
        print("  [SKIP] leaderboard.json 不存在")
        return
    leaderboards = file_data.get("leaderboards", {})
    file_task_ids = set()
    file_count = 0
    for tid, entries in leaderboards.items():
        for e in entries:
            file_count += 1
            if e.get("task_id"):
                file_task_ids.add(e["task_id"])
    print(f"  文件中: {file_count} 条 ({len(leaderboards)} 个测试集)")

    with app.app_context():
        db_rows = db.session.query(EvaluationResult).all()
        db_task_ids = {r.task_id for r in db_rows}
        print(f"  DB  中: {len(db_rows)} 条")

        file_only = file_task_ids - db_task_ids
        db_only = db_task_ids - file_task_ids
        print(f"\n  仅文件有 (by task_id, {len(file_only)}):")
        print(_fmt(sorted(file_only)))
        print(f"\n  仅DB有 (by task_id, {len(db_only)}):")
        print(_fmt(sorted(db_only)))


def check_models(app):
    print("\n" + "=" * 60)
    print("3. Model: models.json vs DB")
    print("=" * 60)
    from app.models import Model
    from app.extensions import db

    file_data = _load_json(MODELS_JSON)
    if file_data is None:
        print("  [SKIP] models.json 不存在")
        return
    file_models = file_data.get("models", {})
    file_paths = {v.get("path") for v in file_models.values() if v.get("path")}
    print(f"  文件中: {len(file_models)} 条 ({len(file_paths)} 个唯一路径)")

    with app.app_context():
        db_rows = db.session.query(Model).all()
        db_paths = {r.path for r in db_rows}
        print(f"  DB  中: {len(db_rows)} 条")

        file_only = file_paths - db_paths
        db_only = db_paths - file_paths
        print(f"\n  仅文件有 ({len(file_only)}):")
        print(_fmt(sorted(file_only)))
        print(f"\n  仅DB有 ({len(db_only)}):")
        print(_fmt(sorted(db_only)))


def check_tasks(app):
    print("\n" + "=" * 60)
    print("4. Task: merges/*/metadata.json vs DB")
    print("=" * 60)
    from app.models import Task
    from app.extensions import db

    if not os.path.isdir(MERGES_DIR):
        print("  [SKIP] merges/ 目录不存在")
        return

    file_tasks = {}
    for name in os.listdir(MERGES_DIR):
        meta_path = os.path.join(MERGES_DIR, name, "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                file_tasks[name] = meta
            except Exception:
                file_tasks[name] = {"_parse_error": True}
    print(f"  文件中: {len(file_tasks)} 个任务目录有 metadata.json")

    with app.app_context():
        db_rows = db.session.query(Task).all()
        db_map = {r.id: r for r in db_rows}
        print(f"  DB  中: {len(db_map)} 条")

        file_only = [k for k in file_tasks if k not in db_map]
        db_only = [k for k in db_map if k not in file_tasks]

        status_mismatches = []
        for k in file_tasks:
            if k in db_map:
                f_status = file_tasks[k].get("status", "")
                d_status = db_map[k].status or ""
                f_norm = {"success": "completed", "error": "failed"}.get(f_status, f_status)
                d_norm = d_status
                if f_norm != d_norm and f_status and d_status:
                    status_mismatches.append(f"{k}: file={f_status!r} db={d_status!r}")

        print(f"\n  仅文件有 ({len(file_only)}):")
        print(_fmt(file_only))
        print(f"\n  仅DB有 ({len(db_only)}):")
        print(_fmt(db_only))
        print(f"\n  状态不一致 ({len(status_mismatches)}):")
        print(_fmt(status_mismatches))


def main():
    from app import create_app
    app = create_app()

    print("=" * 60)
    print("DB vs 文件一致性校验报告")
    print("=" * 60)

    check_testsets(app)
    check_leaderboard(app)
    check_models(app)
    check_tasks(app)

    print("\n" + "=" * 60)
    print("校验完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
