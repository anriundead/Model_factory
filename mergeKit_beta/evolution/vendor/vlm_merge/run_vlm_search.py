#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM/LLM 进化融合搜索：TIES-DARE + CMA-ES，支持 text 模式（HF 数据集如 MMLU）。
内置于 mergeKit_beta `evolution/vendor/vlm_merge`；由 `evolution.runner` 子进程调用。数据集在传入 Problem 前转为 list[dict]，避免 Ray 内 dataclass 错误。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml
import ray
from ray.util.multiprocessing.pool import Pool
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import StarmapParallelization
from transformers import AutoModelForCausalLM, AutoTokenizer

from mergenetic.merging.ties_dare_merger import TiesDareMerger
from mergenetic.optimization.merging_problem import BaseMergingProblem
from mergenetic.searcher.searcher import Searcher

# 本目录为算法根（prompt 相对路径、本地数据候选路径均相对此目录）
PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def ensure_text_architecture(model_paths: list[str]) -> None:
    """
    仅对 Qwen2/Qwen2.5-VL 相关模型修改 config.json 的 model_type/architectures，
    避免误将 Llama 等改为 qwen2。Llama 模型直接跳过。
    """
    for path in model_paths:
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_type = (cfg.get("model_type") or "").lower()
            if model_type not in ("qwen2", "qwen2_vl", "qwen2_5_vl"):
                logger.info("[arch_fix] 跳过非 Qwen2 系模型 %s（model_type=%s）", path, model_type)
                continue
            # 仅对 Qwen2/VL 统一为 Qwen2ForCausalLM 以便兼容
            cfg["model_type"] = "qwen2"
            cfg["architectures"] = ["Qwen2ForCausalLM"]
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            logger.info("[arch_fix] patching architectures for %s -> Qwen2ForCausalLM", path)
        except Exception as e:
            logger.warning("[arch_fix] skip %s: %s", path, e)


def _dataset_to_list_of_dicts(dataset) -> list[dict[str, Any]]:
    """
    将 HuggingFace Dataset 转为纯 Python list[dict]，避免在 Ray 中触发
    dataclasses.fields() 对非 dataclass 的调用（HF 行类型）。
    """
    out: list[dict[str, Any]] = []
    try:
        if hasattr(dataset, "to_pandas"):
            df = dataset.to_pandas()
            out = df.to_dict("records")
            out = [dict(r) for r in out]
            return out
    except Exception:
        pass
    try:
        cols = getattr(dataset, "column_names", None) or []
        n = len(dataset)
        for i in range(n):
            row = dataset[i]
            if hasattr(row, "keys"):
                out.append(dict(row))
            elif isinstance(row, (list, tuple)) and cols:
                out.append(dict(zip(cols, row)))
            else:
                out.append(dict(zip(cols, [row[k] for k in cols])))
    except Exception:
        pass
    if not out and hasattr(dataset, "__iter__"):
        try:
            for row in dataset:
                if isinstance(row, dict):
                    out.append(dict(row))
                else:
                    out.append(dict(zip(getattr(dataset, "column_names", []), row)))
        except Exception:
            pass
    # 确保每项为纯 dict，且可序列化
    result = []
    for r in out:
        clean = {}
        for k, v in r.items():
            if hasattr(v, "tolist"):
                clean[k] = v.tolist()
            elif isinstance(v, (list, tuple)) and v and hasattr(v[0], "tolist"):
                clean[k] = [x.tolist() if hasattr(x, "tolist") else x for x in v]
            else:
                clean[k] = v
        result.append(clean)
    return result


def _load_mmlu_from_local(base_path: str | Path, subset: str, split: str) -> list[dict[str, Any]] | None:
    """
    从本地目录加载 MMLU 风格数据。目录结构可为：
    - {base_path}/{subset}/{split}.parquet 或 .json
    - {base_path}/{subset}.parquet 或 {subset}.json
    - {base_path} 下任意 .parquet/.json（按 subset 名匹配）
    返回 list[dict] 或 None（未找到时）。
    """
    base = Path(base_path).resolve()
    if not base.is_dir():
        return None
    # 尝试 {base}/{subset}/{split}.parquet 等
    for ext in ("parquet", "json"):
        p = base / subset / f"{split}.{ext}"
        if p.exists():
            try:
                from datasets import load_dataset
                ds = load_dataset(ext, data_files={split: str(p)}, split=split)
                return _dataset_to_list_of_dicts(ds)
            except Exception as e:
                logger.warning("[local] 加载 %s 失败: %s", p, e)
    for ext in ("parquet", "json"):
        p = base / f"{subset}.{ext}"
        if p.exists():
            try:
                from datasets import load_dataset
                ds = load_dataset(ext, data_files=str(p), split=split if split != "validation" else "test")
                return _dataset_to_list_of_dicts(ds)
            except Exception as e:
                logger.warning("[local] 加载 %s 失败: %s", p, e)
    # 单文件：{base}/xxx.parquet 或 xxx.json
    for ext in ("parquet", "json"):
        for f in base.glob(f"*{subset}*.{ext}"):
            try:
                from datasets import load_dataset
                ds = load_dataset("parquet" if f.suffix == ".parquet" else "json", data_files=str(f))
                split_key = split if split in ("train", "test", "validation") else "test"
                if hasattr(ds, split_key):
                    ds = getattr(ds, split_key)
                else:
                    ds = ds["train"] if "train" in ds else list(ds.values())[0]
                return _dataset_to_list_of_dicts(ds)
            except Exception as e:
                logger.warning("[local] 加载 %s 失败: %s", f, e)
    return None


def _load_mmlu_samples_one_split(
    hf_dataset_id: str,
    hf_subset_list: list[str],
    hf_split: str,
    max_samples: int | None,
    seed: int,
    cache_dir: str | None,
    hf_subset_group: str,
) -> list[dict[str, Any]]:
    """
    加载 MMLU 风格数据的一个 split（用于进化或最终 test 评测）。
    优先本地路径，否则 HuggingFace；返回 list[dict]。
    """
    from datasets import load_dataset

    if not hf_dataset_id:
        hf_dataset_id = "cais/mmlu"
    workspaces = PROJECT_ROOT.parent.parent.parent if len(PROJECT_ROOT.parents) >= 3 else PROJECT_ROOT.parent
    local_base_for_group = None
    for candidate in [
        Path(hf_dataset_id),
        (PROJECT_ROOT / hf_dataset_id),
        (PROJECT_ROOT.parent / hf_dataset_id),
        (workspaces / hf_dataset_id),
        (workspaces.parent / hf_dataset_id),
    ]:
        try:
            c = candidate.resolve()
            if c.exists() and c.is_dir():
                local_base_for_group = str(c)
                break
        except Exception:
            continue

    samples: list[dict[str, Any]] = []
    if local_base_for_group and hf_subset_group:
        domain_part = _load_mmlu_from_local(local_base_for_group, hf_subset_group, hf_split)
        if domain_part:
            samples = domain_part
    if not samples:
        for sub in hf_subset_list:
            part: list[dict[str, Any]] = []
            local_base = local_base_for_group
            if local_base:
                part = _load_mmlu_from_local(local_base, sub, hf_split) or []
            if not part:
                try:
                    ds = load_dataset(
                        hf_dataset_id,
                        sub,
                        split=hf_split,
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                    )
                    part = _dataset_to_list_of_dicts(ds)
                except (TypeError, Exception) as e:
                    err_str = str(e).lower()
                    if "dataclass" in err_str or "fields" in err_str:
                        try:
                            with tempfile.TemporaryDirectory(prefix="mmlu_cache_") as tmp_cache:
                                ds = load_dataset(
                                    hf_dataset_id,
                                    sub,
                                    split=hf_split,
                                    trust_remote_code=True,
                                    cache_dir=tmp_cache,
                                )
                                part = _dataset_to_list_of_dicts(ds)
                        except Exception:
                            if hf_split != "test":
                                try:
                                    with tempfile.TemporaryDirectory(prefix="mmlu_cache_") as tmp_cache:
                                        ds = load_dataset(hf_dataset_id, sub, split="test", trust_remote_code=True, cache_dir=tmp_cache)
                                        part = _dataset_to_list_of_dicts(ds)
                                except Exception:
                                    pass
                            if not part:
                                raise e
                    elif hf_split != "test":
                        try:
                            ds = load_dataset(hf_dataset_id, sub, split="test", trust_remote_code=True, cache_dir=cache_dir)
                            part = _dataset_to_list_of_dicts(ds)
                        except Exception:
                            raise e
                    else:
                        raise
            if part:
                samples.extend(part)
    if max_samples and len(samples) > max_samples:
        import random
        random.seed(seed)
        samples = random.sample(samples, max_samples)
    return samples


def load_prompt_yaml(prompt_yaml: str) -> dict:
    path = Path(prompt_yaml)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists() and not path.is_absolute():
        alt = PROJECT_ROOT / "eval" / (path.name or "prompt.yaml")
        if alt.exists():
            path = alt
    
    # Check for non-YAML files (like .py or .md from auto-discovery)
    if path.suffix.lower() not in [".yaml", ".yml"]:
        logger.warning(f"File {path} is not a YAML file. Using default configuration.")
        # Return default MMLU-style template
        return {
            "multi_choice_example_format": ["{}\n{}\nAnswer:"]
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to parse YAML from {path}: {e}. Using default configuration.")
        return {
            "multi_choice_example_format": ["{}\n{}\nAnswer:"]
        }


def build_mmlu_prompt(question: str, choices: list, prompt_cfg: dict) -> str:
    """MMLU: question, choices (list of 4), answer 0-3 -> prompt string."""
    template = (prompt_cfg.get("multi_choice_example_format") or [""])[0]
    opts = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    return template.format(question, opts).strip() + " "


def parse_mmlu_answer(response: str) -> str:
    response = (response or "").strip().upper()
    if not response:
        return ""
    letter_match = re.search(r"(?<![A-Z])([ABCD])(?![A-Z])", response)
    if letter_match:
        return letter_match.group(1)
    digit_match = re.search(r"(?<!\d)([0-3])(?!\d)", response)
    if digit_match:
        return chr(65 + int(digit_match.group(1)))
    digit_match = re.search(r"(?<!\d)([1-4])(?!\d)", response)
    if digit_match:
        return chr(64 + int(digit_match.group(1)))
    if response[0] in "ABCD":
        return response[0]
    return ""


def load_llm(model_path: str, device: str, torch_dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model = model.to(device)
    model.eval()
    return tokenizer, model


def _generate_responses(
    prompts: list[str],
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    responses = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        if device == "auto":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], out_ids)]
        responses.extend(
            tokenizer.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        )
    return responses


def llm_text_eval_mmlu(
    merged_llm_dir: str,
    *,
    samples: list[dict],
    prompt_cfg: dict,
    device: str,
    torch_dtype: torch.dtype,
    max_new_tokens: int = 64,
    batch_size: int = 4,
) -> float:
    """
    在纯 list[dict] 样本上做 MMLU 风格评测；samples 必须为 list[dict]，不得为 HF Dataset。
    """
    if not samples:
        return 0.0
    try:
        if not isinstance(samples, list):
            samples = list(samples)
        for i, s in enumerate(samples):
            if not isinstance(s, dict):
                samples[i] = dict(s) if hasattr(s, "keys") else {}
    except Exception as e:
        logger.warning("[eval] samples 转换异常: %s", e)
        return 0.0

    tokenizer, model = load_llm(merged_llm_dir, device, torch_dtype)
    prompts = []
    gold_letters = []
    for s in samples:
        q = s.get("question", "")
        choices = s.get("choices", [])
        if isinstance(choices, (list, tuple)):
            choices = [str(c) for c in choices]
        else:
            choices = []
        ans_idx = s.get("answer", 0)
        if isinstance(ans_idx, (str, float)):
            try:
                ans_idx = int(float(ans_idx))
            except (ValueError, TypeError):
                ans_idx = 0
        gold_letters.append(chr(65 + min(max(0, ans_idx), 3)))
        prompts.append(build_mmlu_prompt(q, choices, prompt_cfg))

    responses = _generate_responses(
        prompts, tokenizer, model, device=device, max_new_tokens=max_new_tokens, batch_size=batch_size
    )
    correct = 0
    # 临时调试：打印前若干条样本的 gold/pred/raw，便于定位 acc=0 问题
    debug_limit = int(os.environ.get("MERGEKIT_DEBUG_PRED_LIMIT", "8") or 8)
    for idx, (resp, gold) in enumerate(zip(responses, gold_letters)):
        pred = parse_mmlu_answer(resp)
        if idx < debug_limit:
            logger.info(
                "[debug-pred] idx=%s gold=%s pred=%s raw=%r",
                idx,
                gold,
                pred,
                (resp or "")[:220],
            )
        if pred == gold:
            correct += 1
    acc = correct / len(samples) if samples else 0.0

    try:
        del model, tokenizer
        torch.cuda.empty_cache()
    except Exception:
        pass
    return float(acc)


class TextLLMMergingProblem(BaseMergingProblem):
    """进化融合 + 文本评测（MMLU 等），评测数据使用 list[dict] 避免 Ray 内 dataclass 错误。"""

    def __init__(
        self,
        merger: TiesDareMerger,
        llm_eval_fn: Callable[..., float],
        llm_eval_kwargs: dict,
        n_var: int,
        device: str,
        progress_file: str | None = None,
        xl: float = 0.0,
        xu: float = 1.0,
        global_best_path: Path | None = None,
    ) -> None:
        super().__init__(
            merger=merger,
            n_var=n_var,
            n_obj=1,
            xl=xl,
            xu=xu,
            discrete=False,
            device=device,
            load_in_4bit=False,
            use_lm_eval=False,
        )
        self.llm_eval_fn = llm_eval_fn
        self.llm_eval_kwargs = llm_eval_kwargs
        self.progress_file = progress_file
        self.base_yaml_path = Path(self.merger.path_to_store_yaml)
        self.base_out_path = Path(self.merger.path_to_store_merged_model)
        self.best_f: float | None = None
        self.best_x: list | None = None
        self.global_best_path = global_best_path

    def metrics_4_genotype(self, model, tokenizer=None):
        raise NotImplementedError("LLM 评分在 _evaluate 中直接调用")

    def test(self, genotype):
        out = {}
        return self._evaluate(genotype, out)

    def get_data(self) -> pd.DataFrame:
        return self.results_df

    def _evaluate(self, x, out, *args, **kwargs):
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                force=True,
            )
        logger.info(
            "[eval-print] step=%s pid=%s cuda_visible=%s genotype=%s",
            self.step + 1,
            os.getpid(),
            os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
            list(np.array(x).flatten()),
        )
        try:
            t0 = time.time()
            uid = uuid.uuid4().hex
            base_yaml = self.base_yaml_path
            base_out = self.base_out_path
            self.merger.path_to_store_yaml = base_yaml.parent / f"{base_yaml.stem}_{uid}.yaml"
            self.merger.path_to_store_merged_model = base_out / uid

            merged_cfg = self.merger.create_individual_configuration(x)
            
            merged_dir = self.merger.merge_model_from_configuration(merged_cfg)
            merged_dir = str(merged_dir)

            acc = float(self.llm_eval_fn(merged_dir, **self.llm_eval_kwargs))
            f = [-acc]
            out["F"] = f

            self.step += 1
            log_entry = dict(
                zip(self.phenotype_feature_list, np.array(x).flatten()),
                **{self.objective_list[0]: f[0]},
                step=self.step,
            )
            row_df = pd.DataFrame([log_entry]).reindex(columns=self.results_df.columns)
            self.results_df = pd.concat([self.results_df, row_df], ignore_index=True)

            if self.best_f is None or f[0] < self.best_f:
                self.best_f = f[0]
                self.best_x = list(np.array(x).flatten())
            best_f = self.best_f if self.best_f is not None else f[0]
            best_g = self.best_x if self.best_x is not None else list(np.array(x).flatten())

            # 进度文件
            if self.progress_file:
                try:
                    progress = {
                        "step": self.step,
                        "current_best": -float(best_f),
                        "global_best": -float(best_f),
                        "best_genotype": best_g,
                        "message": "",
                    }
                    Path(self.progress_file).parent.mkdir(parents=True, exist_ok=True)
                    with open(self.progress_file, "w", encoding="utf-8") as pf:
                        json.dump(progress, pf, ensure_ascii=False, indent=2)
                except Exception as pe:
                    logger.warning("[progress] write failed: %s", pe)

            logger.info("[eval] step=%s acc=%.4f best_acc=%.4f genotype=%s", self.step, acc, -best_f, best_g)
            
            try:
                shutil.rmtree(merged_dir, ignore_errors=True)
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return acc
        except Exception as e:
            logger.exception("[eval] failed genotype=%s", x)
            raise e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM/LLM 进化融合搜索（text 或 vlm 模式）")
    p.add_argument("--run-id", type=str, default="vlm_search")
    p.add_argument("--model-paths", type=str, nargs="+", required=True, help="待融合模型路径（第一个为 base）")
    p.add_argument("--vlm-path", type=str, default="", help="VLM 路径（提供视觉塔+tokenizer，vlm 模式必填）")
    p.add_argument("--eval-mode", type=str, default="text", choices=["text", "vlm"])
    p.add_argument("--hf-dataset", type=str, default="cais/mmlu", help="评测数据集，如 cais/mmlu 或 m-a-p/CMMMU")
    p.add_argument("--hf-subset", type=str, nargs="+", default=["college_medicine"], help="数据集子集，可多个，将合并后评测")
    p.add_argument("--hf-subset-group", type=str, default="", help="领域 id（如 stem），本地路径下可整合同领域子集到 base/<group>/<split>.parquet")
    p.add_argument("--hf-split", type=str, default="test", help="进化阶段使用的 split，如 validation（训练/验证）")
    p.add_argument("--hf-split-final", type=str, default="", help="最终准确率评测使用的 split，如 test；若设则保存最优模型后额外跑一次评测并写 final_acc_file")
    p.add_argument("--final-acc-file", type=str, default="", help="写入最终 test 集 acc 的 JSON 路径，需与 --hf-split-final 配合")
    p.add_argument("--prompt-yaml", type=str, default="", help="Prompt 模板，如 eval/prompt.yaml 或 prompt.yaml")
    p.add_argument("--pop-size", type=int, default=20, help="CMA-ES 种群大小")
    p.add_argument("--n-iter", type=int, default=15, help="迭代轮数")
    p.add_argument("--max-samples", type=int, default=64, help="每次评测最大样本数")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--final-vlm-output", type=str, default="", help="最终输出路径，如 ./best_vlm")
    p.add_argument("--progress-file", type=str, default="")
    p.add_argument("--ray-num-gpus", type=int, default=2, help="Ray GPU 并行数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", type=str, default="vlm_search_results/vlm_search")
    return p.parse_args()


def main():
    args = parse_args()
    use_vlm = bool(getattr(args, "vlm_path", "")) and (
        getattr(args, "eval_mode", "text") == "vlm" or "CMMMU" in (getattr(args, "hf_dataset", "") or "")
    )
    prompt_yaml = getattr(args, "prompt_yaml", "") or ("eval/prompt.yaml" if use_vlm else "eval/prompt_mmlu.yaml")

    logger.info(
        "[main] run_id=%s pop=%s n_iter=%s dtype=%s device=%s ray_gpus=%s",
        args.run_id,
        args.pop_size,
        args.n_iter,
        args.dtype,
        args.device,
        args.ray_num_gpus,
    )
    hf_subset_list = args.hf_subset if isinstance(args.hf_subset, (list, tuple)) else [args.hf_subset]
    logger.info(
        "[main] model_paths=%s vlm_path=%s eval_data=... hf_dataset=%s subsets=%s split=%s",
        args.model_paths,
        getattr(args, "vlm_path", ""),
        args.hf_dataset,
        hf_subset_list,
        args.hf_split,
    )

    ensure_text_architecture(args.model_paths)

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    if use_vlm:
        from vlm_fitness import vlm_cmmmu_fitness
        eval_fn = vlm_cmmmu_fitness
        eval_kwargs = {
            "vlm_path": args.vlm_path,
            "device": args.device,
            "torch_dtype": torch_dtype,
            "hf_dataset": args.hf_dataset,
            "hf_subset": hf_subset_list[0] if hf_subset_list else "health_and_medicine",
            "hf_subsets": hf_subset_list,
            "hf_split": args.hf_split,
            "max_samples": args.max_samples,
            "prompt_yaml": prompt_yaml,
            "project_root": PROJECT_ROOT,
        }
        logger.info("[main] VLM 模式: 使用 vlm_cmmmu_fitness, vlm_path=%s, 子集数=%s", args.vlm_path, len(hf_subset_list))
    else:
        # 进化阶段使用 hf_split（训练/验证建议 val，最终 acc 使用 hf_split_final=test）
        hf_split = args.hf_split if args.hf_split not in ("val", "validation") else "test"
        if args.hf_split in ("val", "validation") and (args.hf_dataset or "").lower().strip() in ("cais/mmlu", "mmlu", "m-a-p/mmlu"):
            logger.info("[main] MMLU 进化用 split=test（请求 val 时）以避免 datasets 兼容问题")
        hf_dataset_id = (args.hf_dataset or "cais/mmlu").strip()
        cache_dir = os.environ.get("HF_DATASETS_CACHE") or None
        if cache_dir:
            logger.info("[main] 使用 HF 缓存目录: %s", cache_dir)
        hf_subset_group = getattr(args, "hf_subset_group", "") or ""
        samples = _load_mmlu_samples_one_split(
            hf_dataset_id,
            hf_subset_list,
            hf_split,
            args.max_samples,
            args.seed,
            cache_dir,
            hf_subset_group,
        )
        logger.info("[main] text 模式: loaded %s samples (plain list[dict]), split=%s, 子集数=%s", len(samples), hf_split, len(hf_subset_list))
        prompt_cfg = load_prompt_yaml(prompt_yaml)
        eval_fn = llm_text_eval_mmlu
        eval_kwargs = {
            "samples": samples,
            "prompt_cfg": prompt_cfg,
            "device": args.device,
            "torch_dtype": torch_dtype,
            "max_new_tokens": 64,
            "batch_size": 4,
        }

    results_root = Path(args.results_dir)
    yaml_dir = results_root / "configs"
    merged_dir = results_root / "merged_models"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    merger = TiesDareMerger(
        run_id=args.run_id,
        path_to_base_model=args.model_paths[0],
        model_paths=args.model_paths[1:] if len(args.model_paths) > 1 else [args.model_paths[0]],
        path_to_store_yaml=str(yaml_dir),
        path_to_store_merged_model=str(merged_dir),
        dtype=args.dtype,
    )
    n_var = 2 * max(1, len(args.model_paths) - 1)
    if n_var < 2:
        n_var = 2

    problem = TextLLMMergingProblem(
        merger=merger,
        llm_eval_fn=eval_fn,
        llm_eval_kwargs=eval_kwargs,
        n_var=n_var,
        device=args.device,
        progress_file=args.progress_file or None,
        xl=0.0,
        xu=1.0,
        global_best_path=results_root / "global_best.json",
    )

    algorithm = CMAES(pop_size=args.pop_size)
    algorithm.sampling = None
    algorithm.evaluator = Evaluator()
    logger.info("[debug] CMAES popsize set to %s, options=%s", args.pop_size, getattr(algorithm, "options", {}))

    use_ray = args.ray_num_gpus > 1
    if use_ray and not ray.is_initialized():
        ray.init(num_gpus=args.ray_num_gpus, ignore_reinit_error=True)

    ray_pool = None
    if use_ray:
        num_workers = max(1, args.ray_num_gpus)
        ray_pool = Pool(processes=num_workers, ray_remote_args={"num_gpus": 1})
        problem.elementwise_runner = StarmapParallelization(ray_pool.starmap)
    else:
        logger.info("[main] ray_num_gpus<=1，串行评估")

    searcher = Searcher(
        problem=problem,
        algorithm=algorithm,
        results_path=str(results_root),
        n_iter=args.n_iter,
        run_id=args.run_id,
        seed=args.seed,
        verbose=True,
    )
    searcher.search()

    if ray_pool is not None:
        ray_pool.close()
        ray_pool.join()
        try:
            ray.shutdown()
        except Exception:
            pass

    # 保存最优到 final_vlm_output
    try:
        best_x = None
        global_best_file = results_root / "global_best.json"
        if global_best_file.exists():
            try:
                data = json.loads(global_best_file.read_text(encoding="utf-8"))
                best_x = data.get("best_genotype")
            except Exception:
                pass
        if best_x is None and searcher.result_X is not None:
            if hasattr(searcher.result_X, "__len__") and len(searcher.result_X) > 0:
                idx = int(np.argmin(searcher.result_F)) if searcher.result_F is not None else 0
                best_x = np.array(searcher.result_X[idx]).flatten().tolist()
            else:
                best_x = np.array(searcher.result_X).flatten().tolist()
        if best_x is not None and args.final_vlm_output:
            best_x = np.asarray(best_x, dtype=np.float32)
            best_cfg = merger.create_individual_configuration(best_x)
            final_merged = merger.merge_model_from_configuration(best_cfg)
            out_dir = Path(args.final_vlm_output)
            out_dir.mkdir(parents=True, exist_ok=True)
            if Path(final_merged).resolve() != out_dir.resolve() and out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            if Path(final_merged).resolve() != out_dir.resolve():
                shutil.move(str(final_merged), str(out_dir))
            logger.info("[main] best model saved to %s", out_dir)
    except Exception as e:
        logger.warning("[main] post-process best model failed: %s", e)

    logger.info("搜索完成，结果目录 %s", results_root)


if __name__ == "__main__":
    main()
