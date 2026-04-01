#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import time
import torch
import sys
from pathlib import Path

# Add current directory to sys.path to ensure we can import run_vlm_search
sys.path.append(str(Path(__file__).resolve().parent))

from run_vlm_search import llm_text_eval_mmlu, _load_mmlu_samples_one_split, load_prompt_yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--hf-dataset", default="cais/mmlu")
    p.add_argument("--hf-subset", nargs="+", default=["college_medicine"])
    p.add_argument("--hf-subset-group", default="")
    p.add_argument("--hf-split", required=True)
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--output-file", required=True)
    p.add_argument("--prompt-yaml", default="")
    return p.parse_args()

def main():
    args = parse_args()
    logger.info("Starting final evaluation for %s on %s/%s", args.model_path, args.hf_dataset, args.hf_split)
    
    t_start = time.time()

    _cd = (os.environ.get("HF_DATASETS_CACHE") or "").strip()
    cache_dir = _cd if _cd else None
    prompt_cfg = {}
    if args.prompt_yaml:
        prompt_cfg = load_prompt_yaml(args.prompt_yaml)

    # Load samples
    samples = _load_mmlu_samples_one_split(
        args.hf_dataset,
        args.hf_subset,
        args.hf_split,
        args.max_samples,
        args.seed,
        cache_dir,
        args.hf_subset_group
    )
    
    logger.info("Loaded %d samples", len(samples))

    # Eval
    acc = llm_text_eval_mmlu(
        args.model_path,
        samples=samples,
        prompt_cfg=prompt_cfg,
        device=args.device,
        torch_dtype=getattr(torch, args.dtype),
        max_new_tokens=64,
        batch_size=4
    )
    
    t_end = time.time()
    duration = t_end - t_start
    
    logger.info("Final Eval: acc=%.4f duration=%.4f", acc, duration)
    
    with open(args.output_file, "w") as f:
        json.dump({
            "final_test_acc": acc,
            "split": args.hf_split,
            "duration": duration
        }, f, indent=2)

if __name__ == "__main__":
    main()
