#!/usr/bin/env python3
"""
回填 TestSet.sample_count（支持 MMLU 组级子集累加）。

用法（在 mergeKit_beta 根目录）:
  python scripts/backfill_testset_sample_counts.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def main():
    from app import create_app
    from app.repositories import testset_list_from_db, testset_upsert
    from app.state import AppState
    from app.services import Services
    from config import Config

    app = create_app()
    state = AppState()
    services = Services(state)
    services.app = app
    cache_dir = getattr(Config, "HF_DATASETS_CACHE", None) or os.environ.get("HF_DATASETS_CACHE")

    with app.app_context():
        rows = testset_list_from_db()
        updated = 0
        for row in rows:
            n, sp, _ = services.resolve_dataset_sample_count(
                row.hf_dataset, row.hf_subset, row.hf_split, cache_dir
            )
            if n <= 0:
                print(f"skip (0) {row.id} {row.hf_dataset} subset={row.hf_subset}")
                continue
            testset_upsert(
                testset_id=row.id,
                name=row.name,
                hf_dataset=row.hf_dataset,
                hf_subset=row.hf_subset,
                hf_split=sp or row.hf_split,
                lm_eval_task=row.lm_eval_task,
                benchmark_config=row.benchmark_config,
                version=row.version,
                sample_count=n,
                is_local=row.is_local,
                local_path=row.local_path,
                yaml_template_path=row.yaml_template_path,
                created_by=row.created_by,
                notes=row.notes,
                question_type=row.question_type,
                type=row.type,
                cached_configs=getattr(row, "cached_configs", None),
                cached_splits=getattr(row, "cached_splits", None),
            )
            print(f"ok {row.id} sample_count={n} split={sp or row.hf_split}")
            updated += 1
        print(f"done, updated={updated} / total={len(rows)}")


if __name__ == "__main__":
    main()
