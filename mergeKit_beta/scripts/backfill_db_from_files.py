#!/usr/bin/env python3
"""
从文件回填 DB：TestSet（testsets.json）、Task（merges/*/metadata.json）。
确保不重复：TestSet 按 id upsert，Task 按 id 存在则更新、不存在则插入。
用法：在 mergeKit_beta 根目录运行
  python scripts/backfill_db_from_files.py
  python scripts/backfill_db_from_files.py --compare   # 回填后自动跑一致性对比
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

PROJECT_ROOT = Config.PROJECT_ROOT
TESTSETS_JSON = os.path.join(PROJECT_ROOT, "testset_repo", "data", "testsets.json")
MERGES_DIR = os.path.join(PROJECT_ROOT, "merges")


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def backfill_testsets(app):
    """testsets.json -> TestSet 表。以文件为准 upsert，补齐 DB 缺失记录并修正字段不一致。"""
    data = _load_json(TESTSETS_JSON)
    if not data:
        return 0, 0
    testsets = data.get("testsets", {})
    if not testsets:
        return 0, 0

    from app.repositories import testset_upsert
    created = updated = 0
    with app.app_context():
        from app.models import TestSet
        from app.extensions import db
        for tid, entry in testsets.items():
            if not entry or not isinstance(entry, dict):
                continue
            testset_id = entry.get("testset_id") or tid
            name = entry.get("name") or str(testset_id) or "未命名"
            row = db.session.get(TestSet, testset_id)
            testset_upsert(
                testset_id=str(testset_id),
                name=name,
                hf_dataset=entry.get("hf_dataset"),
                hf_subset=entry.get("hf_subset"),
                hf_split=entry.get("hf_split"),
                lm_eval_task=entry.get("lm_eval_task"),
                benchmark_config=entry.get("benchmark_config"),
                version=entry.get("version"),
                sample_count=int(entry.get("sample_count") or 0),
                is_local=bool(entry.get("is_local")),
                local_path=entry.get("local_path"),
                yaml_template_path=entry.get("yaml_template_path"),
                created_by=entry.get("created_by"),
                notes=entry.get("notes"),
                question_type=entry.get("question_type"),
                type=entry.get("type"),
            )
            if row is None:
                created += 1
            else:
                updated += 1
    return created, updated


def backfill_tasks(app):
    """merges/*/metadata.json -> Task 表。仅插入缺失任务，已存在则用文件更新。"""
    if not os.path.isdir(MERGES_DIR):
        return 0, 0

    from app.repositories import task_backfill_from_metadata
    created = updated = 0
    with app.app_context():
        from app.models import Task
        from app.extensions import db
        for name in os.listdir(MERGES_DIR):
            meta_path = os.path.join(MERGES_DIR, name, "metadata.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue
            if meta.get("_parse_error"):
                continue
            task_id = meta.get("id") or name
            task = db.session.get(Task, task_id)
            task_backfill_from_metadata(task_id, meta)
            if task is None:
                created += 1
            else:
                updated += 1
    return created, updated


def main():
    parser = argparse.ArgumentParser(description="从文件回填 TestSet / Task 到 DB")
    parser.add_argument("--compare", action="store_true", help="回填后运行一致性校验脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将回填的数量，不写入")
    args = parser.parse_args()

    from app import create_app
    app = create_app()

    print("=" * 60)
    print("DB 回填：TestSet + Task")
    print("=" * 60)

    if args.dry_run:
        with app.app_context():
            from app.models import TestSet, Task
            from app.extensions import db
            n_ts_db = db.session.query(TestSet).count()
            n_task_db = db.session.query(Task).count()
        file_ts = _load_json(TESTSETS_JSON)
        n_ts_file = len(file_ts.get("testsets", {})) if file_ts else 0
        n_task_dirs = 0
        if os.path.isdir(MERGES_DIR):
            for name in os.listdir(MERGES_DIR):
                if os.path.isfile(os.path.join(MERGES_DIR, name, "metadata.json")):
                    n_task_dirs += 1
        print(f"  TestSet: 文件 {n_ts_file} 条, DB 现有 {n_ts_db} 条")
        print(f"  Task:   文件(merges) {n_task_dirs} 个, DB 现有 {n_task_db} 条")
        return

    # TestSet
    c_ts, u_ts = backfill_testsets(app)
    print(f"\n  TestSet: 新增 {c_ts} 条, 更新 {u_ts} 条")

    # Task
    c_t, u_t = backfill_tasks(app)
    print(f"  Task:   新增 {c_t} 条, 更新 {u_t} 条")

    print("\n回填完成。")

    if args.compare:
        print("\n" + "=" * 60)
        print("运行一致性对比")
        print("=" * 60)
        import subprocess
        r = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "check_db_file_consistency.py")],
            cwd=PROJECT_ROOT,
        )
        if r.returncode != 0:
            sys.exit(r.returncode)
        print("\n对比结束")


if __name__ == "__main__":
    main()
