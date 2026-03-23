#!/usr/bin/env python3
"""Verify CMMMU eval task resolution and run a minimal eval flow test."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "eval_task_mapping.json")) as f:
        m = json.load(f)
    v = m.get("m-a-p/CMMMU")
    print("eval_task_mapping m-a-p/CMMMU ->", v)
    if v != "cmmmu_art":
        print("FAIL: expected cmmmu_art")
        return 1

    import merge_manager as mm
    assert "m-a-p/cmmmu" in mm.HF_DATASET_TO_LM_EVAL_TASK
    print("HF_DATASET_TO_LM_EVAL_TASK has m-a-p/cmmmu ->", mm.HF_DATASET_TO_LM_EVAL_TASK.get("m-a-p/cmmmu"))

    extra = mm._load_eval_task_mapping()
    task_name = "cmmmu"
    hf_dataset = "m-a-p/CMMMU"
    cand = (extra.get(hf_dataset) or extra.get(hf_dataset.lower())) if extra and hf_dataset else None
    if cand and str(cand).strip() and str(cand).strip().lower() != "cmmmu":
        task_name = str(cand).strip()
    print("Resolved task_name for m-a-p/CMMMU + lm_eval_task=cmmmu:", task_name)
    if task_name != "cmmmu_art":
        print("FAIL: expected cmmmu_art, got", task_name)
        return 1
    print("OK: CMMMU resolution verified.")
    print("Full E2E: start app, create/select testset with hf_dataset=m-a-p/CMMMU, submit POST /api/evaluate;")
    print("  if lm_eval uses a different CMMMU subtask name, set it in config/eval_task_mapping.json.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
