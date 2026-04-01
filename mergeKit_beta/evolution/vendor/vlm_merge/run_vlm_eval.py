#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM 评测入口：三种模式
1) 仅评测完整 VLM：--vlm-path <dir> --hf-subsets ... --hf-split val
2) 自定义文本塔 + VLM 视觉/processor：--llm-dir <dir> --vlm-path <vlm>
3) 融合后评测：--base-model A --model-paths B --genotype x y --vlm-path VLM --hf-subsets ... --hf-split val
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
import yaml
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_prompt_cfg(prompt_yaml: str) -> dict:
    p = Path(prompt_yaml)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_vlm_on_cmmmu(
    vlm_path: str,
    hf_subsets: list[str],
    hf_split: str,
    device: str,
    torch_dtype: torch.dtype,
    prompt_yaml: str,
    max_samples: int | None,
) -> dict:
    """对完整 VLM 在多个 CMMMU 子集上评测，返回各子集 acc 和总体 acc。"""
    try:
        from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
    from datasets import load_dataset
    from transformers import AutoProcessor

    from eval.eval_utils import evaluate_answer
    from vlm_fitness import build_cmmmu_prompt, parse_choice

    prompt_cfg = _load_prompt_cfg(prompt_yaml)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    vlm.eval()
    processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)

    results = {}
    all_entries = []
    for subset in hf_subsets:
        ds = load_dataset("m-a-p/CMMMU", subset, split=hf_split, trust_remote_code=True)
        n = len(ds)
        if max_samples and n > max_samples:
            ds = ds.select(range(max_samples))
        entries = []
        for i in range(len(ds)):
            ex = ds[i]
            prompt = build_cmmmu_prompt(ex, prompt_cfg)
            image = ex.get("image")
            if image is None:
                continue
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(device, dtype=torch_dtype) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out_ids = vlm.generate(**inputs, max_new_tokens=64)
            out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            pred = parse_choice(out_text)
            gold = ex.get("answer", "")
            if isinstance(gold, int):
                gold = chr(65 + min(max(0, gold), 3))
            entries.append({
                "type": "选择",
                "answer": gold,
                "predicted_answer": pred,
                "index2ans": {"A": ex.get("option1", ""), "B": ex.get("option2", ""), "C": ex.get("option3", ""), "D": ex.get("option4", "")},
            })
            all_entries.extend(entries)
        if entries:
            r = evaluate_answer(entries)
            results[subset] = r.get("acc", 0.0)
            logger.info("[eval] subset=%s acc=%.4f (%s/%s)", subset, results[subset], r.get("correct_num", 0), r.get("entries_num", 0))
    if all_entries:
        overall = evaluate_answer(all_entries)
        results["overall"] = overall.get("acc", 0.0)
    else:
        results["overall"] = 0.0
    del vlm, processor
    torch.cuda.empty_cache()
    return results


def _run_vlm_with_custom_llm(llm_dir: str, vlm_path: str, hf_subsets: list[str], hf_split: str, device: str, torch_dtype: torch.dtype, prompt_yaml: str, max_samples: int | None) -> dict:
    """用自定义 LLM 替换 VLM 的 language_model 后评测。"""
    try:
        from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
    from transformers import AutoModelForCausalLM, AutoProcessor
    from datasets import load_dataset

    from vlm_fitness import build_cmmmu_prompt, parse_choice
    from eval.eval_utils import evaluate_answer

    prompt_cfg = _load_prompt_cfg(prompt_yaml)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    merged_lm = AutoModelForCausalLM.from_pretrained(
        llm_dir,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    vlm.language_model.load_state_dict(merged_lm.state_dict(), strict=False)
    del merged_lm
    torch.cuda.empty_cache()
    vlm.eval()
    processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)

    results = {}
    all_entries = []
    for subset in hf_subsets:
        ds = load_dataset("m-a-p/CMMMU", subset, split=hf_split, trust_remote_code=True)
        n = len(ds)
        if max_samples and n > max_samples:
            ds = ds.select(range(max_samples))
        entries = []
        for i in range(len(ds)):
            ex = ds[i]
            prompt = build_cmmmu_prompt(ex, prompt_cfg)
            image = ex.get("image")
            if image is None:
                continue
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(device, dtype=torch_dtype) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out_ids = vlm.generate(**inputs, max_new_tokens=64)
            out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            pred = parse_choice(out_text)
            gold = ex.get("answer", "")
            if isinstance(gold, int):
                gold = chr(65 + min(max(0, gold), 3))
            entries.append({
                "type": "选择",
                "answer": gold,
                "predicted_answer": pred,
                "index2ans": {"A": ex.get("option1", ""), "B": ex.get("option2", ""), "C": ex.get("option3", ""), "D": ex.get("option4", "")},
            })
        all_entries.extend(entries)
        if entries:
            r = evaluate_answer(entries)
            results[subset] = r.get("acc", 0.0)
            logger.info("[eval] subset=%s acc=%.4f", subset, results[subset])
    if all_entries:
        results["overall"] = evaluate_answer(all_entries).get("acc", 0.0)
    else:
        results["overall"] = 0.0
    del vlm, processor
    torch.cuda.empty_cache()
    return results


def _merge_then_eval(
    base_model: str,
    model_paths: list[str],
    genotype: list[float],
    vlm_path: str,
    hf_subsets: list[str],
    hf_split: str,
    device: str,
    dtype: str,
    prompt_yaml: str,
    max_samples: int | None,
) -> dict:
    """先按 genotype 融合 base + model_paths，再替换 VLM 的 language_model 做评测。"""
    from mergenetic.merging.ties_dare_merger import TiesDareMerger
    import tempfile
    import uuid

    run_id = "eval_" + uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory(prefix="run_vlm_eval_") as tmp:
        yaml_dir = Path(tmp) / "configs"
        merged_dir = Path(tmp) / "merged"
        yaml_dir.mkdir(parents=True, exist_ok=True)
        merged_dir.mkdir(parents=True, exist_ok=True)
        merger = TiesDareMerger(
            run_id=run_id,
            path_to_base_model=base_model,
            model_paths=model_paths,
            path_to_store_yaml=str(yaml_dir),
            path_to_store_merged_model=str(merged_dir),
            dtype=dtype,
        )
        x = list(genotype)
        cfg_path = merger.create_individual_configuration(x)
        out_path = merger.merge_model_from_configuration(cfg_path)
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype, torch.bfloat16)
        return _run_vlm_with_custom_llm(
            str(out_path),
            vlm_path,
            hf_subsets,
            hf_split,
            device,
            torch_dtype,
            prompt_yaml,
            max_samples,
        )


def parse_args():
    p = argparse.ArgumentParser(description="VLM 评测（完整 VLM / 自定义 LLM / 融合后评测）")
    p.add_argument("--vlm-path", type=str, default="", help="VLM 路径（提供视觉塔+tokenizer）")
    p.add_argument("--llm-dir", type=str, default="", help="已有文本塔路径（与 vlm-path 组合评测）")
    p.add_argument("--base-model", type=str, default="", help="融合 base 模型（融合后评测）")
    p.add_argument("--model-paths", type=str, nargs="*", default=[], help="融合的其它模型路径")
    p.add_argument("--genotype", type=float, nargs="+", default=[], help="融合权重 [w1,d1,w2,d2,...]")
    p.add_argument("--hf-subsets", type=str, nargs="+", default=["health_and_medicine"], help="CMMMU 子集")
    p.add_argument("--hf-split", type=str, default="val")
    p.add_argument("--prompt-yaml", type=str, default="eval/prompt.yaml")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    if args.llm_dir and args.vlm_path:
        logger.info("[run_vlm_eval] 模式: 自定义文本塔 + VLM")
        results = _run_vlm_with_custom_llm(
            args.llm_dir,
            args.vlm_path,
            args.hf_subsets,
            args.hf_split,
            args.device,
            torch_dtype,
            args.prompt_yaml,
            args.max_samples,
        )
    elif args.base_model and args.model_paths and args.genotype and args.vlm_path:
        logger.info("[run_vlm_eval] 模式: 融合后评测")
        results = _merge_then_eval(
            args.base_model,
            args.model_paths,
            args.genotype,
            args.vlm_path,
            args.hf_subsets,
            args.hf_split,
            args.device,
            args.dtype,
            args.prompt_yaml,
            args.max_samples,
        )
    elif args.vlm_path:
        logger.info("[run_vlm_eval] 模式: 完整 VLM 评测")
        results = _run_vlm_on_cmmmu(
            args.vlm_path,
            args.hf_subsets,
            args.hf_split,
            args.device,
            torch_dtype,
            args.prompt_yaml,
            args.max_samples,
        )
    else:
        raise SystemExit("请指定 --vlm-path，或 --llm-dir + --vlm-path，或 --base-model + --model-paths + --genotype + --vlm-path")

    logger.info("[run_vlm_eval] overall acc=%.4f", results.get("overall", 0.0))
    print("overall acc:", results.get("overall", 0.0))


if __name__ == "__main__":
    main()
