#!/usr/bin/env python3
"""
向 DB 注册计划中的扩展测试集（若已存在同 id 则更新元数据，不强制改 sample_count）。

用法:
  python scripts/seed_expanded_testsets.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

# id, name, hf_dataset, hf_subset, hf_split, lm_eval_task, notes
SEED = [
    ("bench-ai2-arc-easy", "ARC-Easy", "allenai/ai2_arc", "ARC-Easy", "test", "arc_easy", "科学推理 MCQ"),
    ("bench-ai2-arc-challenge", "ARC-Challenge", "allenai/ai2_arc", "ARC-Challenge", "test", "arc_challenge", "科学推理 MCQ"),
    ("bench-winogrande", "WinoGrande", "allenai/winogrande", "winogrande_xl", "validation", "winogrande", "常识指代"),
    ("bench-boolq", "BoolQ", "google/boolq", "boolq", "validation", "boolq", "布尔问答"),
    ("bench-medqa-usmle", "MedQA-USMLE-4opt", "GBaker/MedQA-USMLE-4-options", "all", "test", "", "医学 MCQ，需 eval 映射"),
    ("bench-pubmedqa", "PubMedQA", "qiaojin/PubMedQA", "pqa_labeled", "train", "pubmedqa", "医学文献 QA"),
    ("bench-medmcqa", "MedMCQA", "openlifescienceai/medmcqa", "english", "validation", "medmcqa", "医学 MCQ"),
    ("bench-cmmlu", "CMMLU", "haonan-li/cmmlu", "agronomy", "test", "cmmlu", "中文 MMLU 单学科示例"),
    ("bench-ceval", "C-Eval", "ceval/ceval-exam", "computer_network", "test", "ceval-valid", "中文考试，子集需按学科选"),
    ("bench-humaneval", "HumanEval", "openai/openai_humaneval", "openai_humaneval", "test", "humaneval", "代码生成"),
    ("bench-mbpp", "MBPP", "google-research-datasets/mbpp", "full", "test", "mbpp", "代码生成"),
    ("bench-mmbench", "MMBench-dev", "lmms-lab/MMBench", "default", "test", "", "VLM，需 run_vlm_eval 适配"),
    ("bench-mme", "MME", "lmms-lab/MME", "default", "test", "", "VLM，需 run_vlm_eval 适配"),
]


def main():
    from app import create_app
    from app.repositories import testset_get_by_id, testset_upsert

    app = create_app()
    with app.app_context():
        for tid, name, hfd, sub, spl, lmt, notes in SEED:
            row = testset_get_by_id(tid)
            sc = int(row.sample_count) if row else 0
            testset_upsert(
                testset_id=tid,
                name=name,
                hf_dataset=hfd,
                hf_subset=sub,
                hf_split=spl,
                lm_eval_task=lmt or None,
                sample_count=sc,
                type="benchmark",
                notes=notes,
            )
            print("upsert", tid, name)
        print("seed done")


if __name__ == "__main__":
    main()
