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
import socket
import shutil
import subprocess
import sys
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
from pymoo.optimize import minimize
from pymoo.termination import get_termination

try:
    # pymoo 版本差异：termination 模块路径可能不同，按需在运行期报错提示
    from pymoo.termination.max_eval import MaximumFunctionCallTermination  # type: ignore
except Exception:  # pragma: no cover
    MaximumFunctionCallTermination = None  # type: ignore

# 本目录为算法根（prompt 相对路径、本地数据候选路径均相对此目录）
PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


_shadow_dirs: list[str] = []


def _cleanup_shadow_dirs():
    for d in _shadow_dirs:
        shutil.rmtree(d, ignore_errors=True)
    _shadow_dirs.clear()


import atexit as _atexit
_atexit.register(_cleanup_shadow_dirs)


def ensure_text_architecture(model_paths: list[str]) -> list[str]:
    """
    对 Qwen2-VL / Qwen2.5-VL 模型，创建 shadow 目录（symlink + 修补后的 config.json +
    仅文本 tensor 的 index），让 mergekit 按 Qwen2ForCausalLM 处理文本塔权重。
    不修改原模型目录（兼容 :ro 只读挂载）。

    Returns: 与 model_paths 等长的路径列表（未修补的保持原路径，修补的返回 shadow 路径）。
    """
    VISUAL_PREFIXES = ("visual", "model.visual", "model.vision_tower", "model.mm_projector")

    result_paths = []
    for path in model_paths:
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            result_paths.append(path)
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_type = (cfg.get("model_type") or "").lower()

            if model_type not in ("qwen2", "qwen2_vl", "qwen2_5_vl"):
                logger.info("[arch_fix] 跳过非 Qwen2 系模型 %s（model_type=%s）", path, model_type)
                result_paths.append(path)
                continue

            current_arch = (cfg.get("architectures") or [None])[0]
            if current_arch == "Qwen2ForCausalLM":
                logger.info("[arch_fix] %s 已是 Qwen2ForCausalLM，无需修补", path)
                result_paths.append(path)
                continue

            model_name = Path(path).name
            shadow_dir = Path(tempfile.mkdtemp(prefix=f"arch_fix_{model_name}_"))
            _shadow_dirs.append(str(shadow_dir))

            for item in Path(path).iterdir():
                if item.name in ("config.json", "model.safetensors.index.json"):
                    continue
                target = shadow_dir / item.name
                target.symlink_to(item, target_is_directory=item.is_dir())

            cfg["model_type"] = "qwen2"
            cfg["architectures"] = ["Qwen2ForCausalLM"]
            rope = cfg.get("rope_scaling")
            if isinstance(rope, dict) and rope.get("type") == "mrope":
                del cfg["rope_scaling"]
            for vl_key in [k for k in cfg if any(
                w in k.lower() for w in ("vision", "visual", "image", "video", "mrope", "mm_")
            )]:
                del cfg[vl_key]

            with open(shadow_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)

            idx_path = Path(path) / "model.safetensors.index.json"
            if idx_path.exists():
                idx = json.loads(idx_path.read_text())
                text_wm = {
                    k: v for k, v in idx.get("weight_map", {}).items()
                    if not any(k.startswith(p) for p in VISUAL_PREFIXES)
                }
                new_idx = {"metadata": idx.get("metadata", {}), "weight_map": text_wm}
                with open(shadow_dir / "model.safetensors.index.json", "w") as f:
                    json.dump(new_idx, f, indent=2)
                logger.info("[arch_fix] shadow index: %d/%d tensors (text only)",
                            len(text_wm), len(idx.get("weight_map", {})))

            logger.info("[arch_fix] shadow %s -> Qwen2ForCausalLM (src: %s)", shadow_dir, path)
            result_paths.append(str(shadow_dir))
        except Exception as e:
            logger.warning("[arch_fix] skip %s: %s", path, e)
            result_paths.append(path)

    return result_paths


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

    # 离线/本地优先策略：
    # - 默认允许在线（由 HF_* 环境变量决定），但可用 MERGEKIT_DATASETS_LOCAL_ONLY 强制只读本地缓存
    # - 强制模式下：不再尝试向 Hub 拉取；缓存缺失则直接抛错，避免长时间卡在下载/锁
    local_only = (os.environ.get("MERGEKIT_DATASETS_LOCAL_ONLY") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    # datasets 支持 local_files_only（新版本）；若不支持则靠 HF_*_OFFLINE 环境变量兜底
    ds_kwargs_common: dict[str, Any] = {
        "trust_remote_code": True,
        "cache_dir": cache_dir,
    }
    if local_only:
        # 仅复用已有缓存，不强制 redownload。注意：不要传 local_files_only=True——
        # datasets 会把该标志写进 cache key（如 *-local_files_only=True），导致无法命中
        # 历史在线阶段已下载的同一子集缓存；离线/禁网由 compose 注入的 HF_*_OFFLINE 兜底。
        ds_kwargs_common["download_mode"] = "reuse_dataset_if_exists"

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
                    ds = load_dataset(hf_dataset_id, sub, split=hf_split, **ds_kwargs_common)
                    part = _dataset_to_list_of_dicts(ds)
                except (TypeError, Exception) as e:
                    err_str = str(e).lower()
                    if "dataclass" in err_str or "fields" in err_str:
                        try:
                            with tempfile.TemporaryDirectory(prefix="mmlu_cache_") as tmp_cache:
                                tmp_kwargs = dict(ds_kwargs_common)
                                tmp_kwargs["cache_dir"] = tmp_cache
                                ds = load_dataset(hf_dataset_id, sub, split=hf_split, **tmp_kwargs)
                                part = _dataset_to_list_of_dicts(ds)
                        except Exception:
                            if hf_split != "test":
                                try:
                                    with tempfile.TemporaryDirectory(prefix="mmlu_cache_") as tmp_cache:
                                        tmp_kwargs = dict(ds_kwargs_common)
                                        tmp_kwargs["cache_dir"] = tmp_cache
                                        ds = load_dataset(hf_dataset_id, sub, split="test", **tmp_kwargs)
                                        part = _dataset_to_list_of_dicts(ds)
                                except Exception:
                                    pass
                            if not part:
                                raise e
                    elif hf_split != "test":
                        try:
                            ds = load_dataset(hf_dataset_id, sub, split="test", **ds_kwargs_common)
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


def _prepare_vllm_torch_dist_env_for_local_tp() -> None:
    """
    vLLM 在 tensor_parallel_size>1 时会起多进程并用 PyTorch c10d（TCPStore）做 rank 协调。

    在 Docker + 多 Ray worker 并行评测时，常见问题：
    - MASTER_ADDR 默认为容器 hostname，解析到 bridge IP（如 172.x），子进程监听/连接路径不一致导致超时；
    - 多个 worker 继承同一 MASTER_PORT（如 29500），TCPStore 端口冲突。

    每次构造 LLM 前为本进程绑定回环地址 + 系统分配的独占端口，避免跨 actor 抢端口与错误网卡。
    若首次 bind 失败，先清除继承的 MASTER_ADDR/MASTER_PORT 再重试一次。
    """
    bind = (os.environ.get("MERGEKIT_VLLM_MASTER_ADDR") or "127.0.0.1").strip() or "127.0.0.1"

    def _try_bind() -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((bind, 0))
            port = int(s.getsockname()[1])
            s.close()
            os.environ["MASTER_ADDR"] = bind
            os.environ["MASTER_PORT"] = str(port)
            # 强制 Gloo 走 loopback，降低子进程把 MASTER 解析成 Docker bridge IP（如 172.x）的概率
            if bind == "127.0.0.1":
                os.environ["GLOO_SOCKET_IFNAME"] = "lo"
            logger.info("[vllm-dist] MASTER_ADDR=%s MASTER_PORT=%s pid=%s", bind, port, os.getpid())
            return True
        except OSError as e:
            logger.warning("[vllm-dist] 无法在 %s 绑定独占端口: %s", bind, e)
            return False

    if _try_bind():
        return
    cleared = []
    for k in ("MASTER_ADDR", "MASTER_PORT"):
        if k in os.environ:
            cleared.append(k)
            del os.environ[k]
    if cleared:
        logger.info("[vllm-dist] 已清除继承的 %s，重试 bind", cleared)
    if not _try_bind():
        logger.warning("[vllm-dist] 重试仍失败，vLLM 将沿用当前进程环境（可能仍有 TCPStore 风险）")


def _want_vllm_tp_subprocess(tensor_parallel_size: int) -> bool:
    """TP>1 时默认子进程隔离；MERGEKIT_VLLM_TP_SUBPROCESS=0 关闭（回滚进程内 vLLM，有同进程第二轮 c10d 风险）。"""
    if max(1, int(tensor_parallel_size)) <= 1:
        return False
    raw = (os.environ.get("MERGEKIT_VLLM_TP_SUBPROCESS") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _llm_text_eval_mmlu_vllm_in_process(
    merged_llm_dir: str,
    *,
    samples: list[dict],
    prompt_cfg: dict,
    tensor_parallel_size: int,
    max_new_tokens: int = 64,
    batch_size: int = 4,
) -> float:
    """当前进程内跑 vLLM TP 评测（供子进程 worker 或 SUBPROCESS=0 使用）。"""
    if not samples:
        return 0.0
    try:
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer  # type: ignore
        if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
            Qwen2Tokenizer.all_special_tokens_extended = property(lambda self: list(getattr(self, "all_special_tokens", [])))  # type: ignore
    except Exception:
        pass
    try:
        from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast  # type: ignore
        if not hasattr(Qwen2TokenizerFast, "all_special_tokens_extended"):
            Qwen2TokenizerFast.all_special_tokens_extended = property(lambda self: list(getattr(self, "all_special_tokens", [])))  # type: ignore
    except Exception:
        pass

    from vllm import LLM, SamplingParams

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
        gold_letters.append(chr(65 + min(max(0, int(ans_idx)), 3)))
        prompts.append(build_mmlu_prompt(q, choices, prompt_cfg))

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    llm = None
    try:
        if max(1, int(tensor_parallel_size)) > 1:
            _prepare_vllm_torch_dist_env_for_local_tp()
        llm = LLM(
            model=merged_llm_dir,
            tensor_parallel_size=max(1, int(tensor_parallel_size)),
            dtype="bfloat16",
            trust_remote_code=True,
        )

        correct = 0
        debug_limit = int(os.environ.get("MERGEKIT_DEBUG_PRED_LIMIT", "8") or 8)
        for start in range(0, len(prompts), max(1, int(batch_size))):
            batch = prompts[start : start + batch_size]
            outs = llm.generate(batch, sampling)
            for i, o in enumerate(outs):
                text = ""
                try:
                    text = (o.outputs[0].text or "").strip()
                except Exception:
                    text = ""
                idx = start + i
                gold = gold_letters[idx] if idx < len(gold_letters) else ""
                pred = parse_mmlu_answer(text)
                if idx < debug_limit:
                    logger.info("[debug-pred] idx=%s gold=%s pred=%s raw=%r", idx, gold, pred, text[:220])
                if pred == gold:
                    correct += 1

        return float(correct / len(samples)) if samples else 0.0
    finally:
        try:
            if llm is not None:
                eng = getattr(llm, "llm_engine", None)
                if eng is not None:
                    mex = getattr(eng, "model_executor", None)
                    if mex is not None and hasattr(mex, "shutdown"):
                        try:
                            mex.shutdown()
                        except Exception:
                            pass
                del llm
        except Exception:
            pass
        try:
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


def _json_sanitize_for_subprocess(x: Any) -> Any:
    """将 MMLU 样本 / prompt_cfg 中的 numpy、pandas 标量转为 json 可序列化类型（供子进程 job 文件）。"""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _json_sanitize_for_subprocess(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sanitize_for_subprocess(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_sanitize_for_subprocess(x.tolist())
    if isinstance(x, np.generic):
        return _json_sanitize_for_subprocess(x.item())
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if pd.api.types.is_scalar(x) and not isinstance(x, str):
        try:
            return _json_sanitize_for_subprocess(getattr(x, "item", lambda: x)())
        except Exception:
            pass
    return x


def _llm_text_eval_mmlu_vllm_subprocess(
    merged_llm_dir: str,
    *,
    samples: list[dict],
    prompt_cfg: dict,
    tensor_parallel_size: int,
    max_new_tokens: int = 64,
    batch_size: int = 4,
) -> float:
    """TP>1 时在独立子进程中跑 vLLM，避免 Ray actor 内多轮 LLM() 触发 c10d/TCPStore 问题。"""
    job = _json_sanitize_for_subprocess(
        {
            "merged_llm_dir": merged_llm_dir,
            "samples": samples,
            "prompt_cfg": prompt_cfg,
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_new_tokens": int(max_new_tokens),
            "batch_size": int(batch_size),
        }
    )
    fd, path = tempfile.mkstemp(prefix="vllm_tp_job_", suffix=".json")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(job, fp, ensure_ascii=False)
        timeout_s = int(float(os.environ.get("MERGEKIT_VLLM_SUBPROCESS_TIMEOUT_S") or "1200"))
        env = os.environ.copy()
        for k in list(env.keys()):
            if k.startswith("MASTER_"):
                env.pop(k, None)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--vllm-tp-eval-worker", path]
        logger.info("[vllm-tp-subproc] timeout=%ss cmd=%s", timeout_s, cmd)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            raise RuntimeError("vllm_tp_worker exit=%s stderr=%s stdout=%s" % (proc.returncode, err[-1500:], out[-500:]))
        acc = None
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and '"acc"' in line:
                acc = float(json.loads(line)["acc"])
                break
        if acc is None:
            raise RuntimeError("vllm_tp_worker no acc in stdout: %s" % (out[-800:],))
        return acc
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def llm_text_eval_mmlu_vllm(
    merged_llm_dir: str,
    *,
    samples: list[dict],
    prompt_cfg: dict,
    tensor_parallel_size: int,
    max_new_tokens: int = 64,
    batch_size: int = 4,
) -> float:
    """
    使用 vLLM 做 text 模式 MMLU 推理评测。

    TP>1 时默认在子进程中执行（MERGEKIT_VLLM_TP_SUBPROCESS=0 时改为进程内，有已知同进程第二轮风险）。
    """
    if not samples:
        return 0.0
    if _want_vllm_tp_subprocess(tensor_parallel_size):
        return _llm_text_eval_mmlu_vllm_subprocess(
            merged_llm_dir,
            samples=samples,
            prompt_cfg=prompt_cfg,
            tensor_parallel_size=tensor_parallel_size,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
    return _llm_text_eval_mmlu_vllm_in_process(
        merged_llm_dir,
        samples=samples,
        prompt_cfg=prompt_cfg,
        tensor_parallel_size=tensor_parallel_size,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )


def _vllm_tp_eval_worker_main(job_path: str) -> None:
    """子进程入口：读 job JSON，stdout 打印一行 {\"acc\": float}。"""
    with open(job_path, encoding="utf-8") as fp:
        job = json.load(fp)
    acc = _llm_text_eval_mmlu_vllm_in_process(
        job["merged_llm_dir"],
        samples=job["samples"],
        prompt_cfg=job["prompt_cfg"],
        tensor_parallel_size=int(job.get("tensor_parallel_size", 1)),
        max_new_tokens=int(job.get("max_new_tokens", 64)),
        batch_size=int(job.get("batch_size", 4)),
    )
    print(json.dumps({"acc": acc}), flush=True)


class MaxEvalsReached(RuntimeError):
    """达到 max_evals 后主动终止搜索（由 main 捕获并进入收尾流程）。"""


def _runner_owns_task_progress() -> bool:
    """为真时 evolution.runner 负责 progress.json；本进程不得覆盖，仅写 progress_mergenetic_debug.json。"""
    v = (os.environ.get("MERGEKIT_RUNNER_OWNS_PROGRESS") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _write_task_progress_json(progress_file: str | None, data: dict) -> None:
    """写入 --progress-file 或同目录下的 debug 侧车文件（runner 独占模式）。"""
    if not progress_file:
        return
    parent = Path(progress_file).parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as pe:
        logger.warning("[progress] mkdir failed: %s", pe)
        return
    target = parent / "progress_mergenetic_debug.json" if _runner_owns_task_progress() else Path(progress_file)
    try:
        with open(target, "w", encoding="utf-8") as pf:
            json.dump(data, pf, ensure_ascii=False, indent=2)
    except Exception as pe:
        logger.warning("[progress] write %s failed: %s", target, pe)


def _bump_eval_counter(path: Path) -> int:
    """跨进程安全递增计数器（同一共享文件系统）。"""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        with os.fdopen(fd, "r+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                raw = f.read().strip()
                cur = int(json.loads(raw).get("eval_count", 0)) if raw else 0
            except Exception:
                cur = 0
            cur += 1
            f.seek(0)
            f.truncate()
            f.write(json.dumps({"eval_count": cur}, ensure_ascii=False))
            f.flush()
            return cur
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


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
        max_evals: int = 0,
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
        self.max_evals = max(0, int(max_evals or 0))
        self._eval_counter_path = (
            (self.global_best_path.parent if self.global_best_path else Path.cwd()) / "eval_counter.json"
        )

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
        # 进入融合/评测前先写一次进度（runner 模式下仅写 progress_mergenetic_debug.json）
        if self.progress_file:
            _write_task_progress_json(
                self.progress_file,
                {
                    "step": self.step,
                    "current_best": None if self.best_f is None else -float(self.best_f),
                    "global_best": None if self.best_f is None else -float(self.best_f),
                    "best_genotype": self.best_x,
                    "message": "evaluating...",
                },
            )
        try:
            t0 = time.time()
            if self.max_evals > 0:
                n = _bump_eval_counter(self._eval_counter_path)
                if n > self.max_evals:
                    logger.info("[main] max_evals=%s reached (n=%s), stop search", self.max_evals, n)
                    if self.progress_file:
                        _write_task_progress_json(
                            self.progress_file,
                            {
                                "status": "running",
                                "message": f"max_evals reached: {self.max_evals}",
                                "step": self.step,
                            },
                        )
                    raise MaxEvalsReached(f"max_evals reached: {self.max_evals}")
            uid = uuid.uuid4().hex
            base_yaml = self.base_yaml_path
            base_out = self.base_out_path
            self.merger.path_to_store_yaml = base_yaml.parent / f"{base_yaml.stem}_{uid}.yaml"
            self.merger.path_to_store_merged_model = base_out / uid

            t_merge_start = time.time()
            merged_cfg = self.merger.create_individual_configuration(x)

            merged_dir = self.merger.merge_model_from_configuration(merged_cfg)
            merged_dir = str(merged_dir)
            t_merge_end = time.time()

            t_eval_start = time.time()
            acc = float(self.llm_eval_fn(merged_dir, **self.llm_eval_kwargs))
            t_eval_end = time.time()
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

            # 进度文件（runner 模式下仅写 debug 侧车，避免覆盖 progress.json）
            if self.progress_file:
                _write_task_progress_json(
                    self.progress_file,
                    {
                        "step": self.step,
                        "current_best": -float(best_f),
                        "global_best": -float(best_f),
                        "best_genotype": best_g,
                        "message": "",
                    },
                )

            t_cleanup_start = time.time()
            logger.info(
                "[eval] step=%s acc=%.4f best_acc=%.4f merge=%.1fs eval=%.1fs total=%.1fs genotype=%s",
                self.step,
                acc,
                -best_f,
                (t_merge_end - t_merge_start),
                (t_eval_end - t_eval_start),
                (time.time() - t0),
                best_g,
            )
            
            try:
                shutil.rmtree(merged_dir, ignore_errors=True)
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            _ = (time.time() - t_cleanup_start)  # 留给日志扩展；避免未来引入 cleanup 字段时改动过大
            return acc
        except Exception as e:
            logger.exception("[eval] failed genotype=%s", x)
            # 失败时写入 debug（runner 模式）；API 仍由 runner 根据子进程输出/退出码写 progress.json
            if self.progress_file:
                msg = str(e)
                _write_task_progress_json(
                    self.progress_file,
                    {
                        "step": self.step,
                        "error": msg[:2000],
                        "message": f"eval_failed: {msg[:240]}",
                    },
                )
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
    p.add_argument("--max-evals", type=int, default=0, help="评测次数硬上限（0=不限制，仍由 n_iter 语义控制）")
    p.add_argument("--max-samples", type=int, default=64, help="每次评测最大样本数")
    p.add_argument("--batch-size", type=int, default=4, help="评测 batch size（显存不足时可在 runner 降低）")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--final-vlm-output", type=str, default="", help="最终输出路径，如 ./best_vlm")
    p.add_argument("--progress-file", type=str, default="")
    p.add_argument("--ray-num-gpus", type=int, default=2, help="Ray GPU 并行数")
    p.add_argument("--tp-size", type=int, default=1, help="推理张量并行度（TP）。text 模式下用于 vLLM tensor_parallel_size；默认 1。")
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

    args.model_paths = ensure_text_architecture(args.model_paths)

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
        # text 模式：若启用 TP>1，则优先使用 vLLM 以获得 tensor_parallel_size 能力
        # 运维兜底：可用环境变量强制关闭 vLLM（回退 transformers）
        vllm_enable = (os.environ.get("MERGEKIT_VLLM_ENABLE") or "1").strip().lower() not in ("0", "false", "no", "off")
        tp_size = int(getattr(args, "tp_size", 1) or 1)
        if tp_size > 1 and vllm_enable:
            try:
                import vllm  # noqa: F401
                def _eval_with_fallback(merged_llm_dir, **kw):
                    try:
                        return llm_text_eval_mmlu_vllm(  # type: ignore
                            merged_llm_dir,
                            samples=kw["samples"],
                            prompt_cfg=kw["prompt_cfg"],
                            tensor_parallel_size=tp_size,
                            max_new_tokens=kw.get("max_new_tokens", 64),
                            batch_size=kw.get("batch_size", 4),
                        )
                    except Exception as e:
                        msg = str(e)
                        # 仅对“可预期的编译链/inductor/triton”问题做降级，避免吞掉真实业务错误
                        keys = (
                            "Failed to find C compiler",
                            "BackendCompilerFailed",
                            "torch._dynamo",
                            "inductor",
                            "triton",
                            # Docker+Ray 同进程多轮 vLLM TP 时 c10d/Gloo 偶发连向 bridge IP；降级单进程 transformers 以保证任务可完成
                            "TCPStore",
                            "c10d::TCPStore",
                            "DistNetworkError",
                            "timed out after 600000ms",
                            "socket has timed out",
                        )
                        if any(k in msg for k in keys):
                            logger.warning("[vllm-fallback] vLLM/TP=%s 失败，降级 transformers: %s", tp_size, msg[:240])
                            return llm_text_eval_mmlu(merged_llm_dir, **kw)
                        raise

                eval_fn = _eval_with_fallback
                logger.info("[main] text 模式启用 vLLM + TP=%s（CUDA_VISIBLE_DEVICES=%s）", tp_size, os.environ.get("CUDA_VISIBLE_DEVICES", ""))
            except Exception as e:
                eval_fn = llm_text_eval_mmlu
                logger.warning("[main] vLLM 不可用，回退 transformers 推理（TP 失效）: %s", e)
        else:
            eval_fn = llm_text_eval_mmlu
            if tp_size > 1 and not vllm_enable:
                logger.info("[main] MERGEKIT_VLLM_ENABLE=0，强制使用 transformers（忽略 TP=%s）", tp_size)
        eval_kwargs = {
            "samples": samples,
            "prompt_cfg": prompt_cfg,
            "device": args.device,
            "torch_dtype": torch_dtype,
            "max_new_tokens": 64,
            "batch_size": max(1, int(getattr(args, "batch_size", 4) or 4)),
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
        max_evals=int(getattr(args, "max_evals", 0) or 0),
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
        ray_kw: dict = {"num_gpus": args.ray_num_gpus, "ignore_reinit_error": True}
        node_ip = (os.environ.get("MERGEKIT_RAY_NODE_IP_ADDRESS") or "").strip()
        single_loop = (os.environ.get("MERGEKIT_RAY_SINGLE_NODE_LOOPBACK") or "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if node_ip:
            ray_kw["_node_ip_address"] = node_ip
            logger.info("[main] ray.init _node_ip_address=%s（来自 MERGEKIT_RAY_NODE_IP_ADDRESS）", node_ip)
        elif single_loop:
            ray_kw["_node_ip_address"] = "127.0.0.1"
            logger.info("[main] ray.init _node_ip_address=127.0.0.1（MERGEKIT_RAY_SINGLE_NODE_LOOPBACK=1，仅单机 Docker）")
        ray.init(**ray_kw)

    ray_pool = None
    # TP=2 模式：用 2 卡一个 worker（避免 Ray 把同一 trial 调度到被挤占的单卡上）
    tp_size = int(getattr(args, "tp_size", 1) or 1)
    if use_ray:
        if tp_size > 1:
            if args.ray_num_gpus % tp_size != 0:
                logger.warning(
                    "[main] ray_num_gpus(%s) 不能整除 tp_size(%s)，将回退为 tp_size=1",
                    args.ray_num_gpus,
                    tp_size,
                )
                tp_size = 1
        if tp_size > 1:
            num_workers = max(1, int(args.ray_num_gpus // tp_size))
            # 兜底：多 Ray worker 同时各启一套 vLLM TP>1 时，若仍遇 c10d/TCPStore 异常可设 MERGEKIT_VLLM_TP_SERIALIZE=1 串行 eval
            if (os.environ.get("MERGEKIT_VLLM_TP_SERIALIZE") or "").strip().lower() in (
                "1", "true", "yes", "on",
            ):
                num_workers = 1
                logger.info(
                    "[main] MERGEKIT_VLLM_TP_SERIALIZE=1：TP>1 时仅 1 个 Ray worker（串行 eval，降低多 vLLM 并行风险）"
                )
            rargs: dict = {"num_gpus": tp_size}
            if (os.environ.get("MERGEKIT_RAY_POOL_RUNTIME_ENV") or "").strip().lower() in ("1", "true", "yes", "on"):
                rargs["runtime_env"] = {"env_vars": {"GLOO_SOCKET_IFNAME": "lo"}}
                logger.info("[main] Ray Pool runtime_env GLOO_SOCKET_IFNAME=lo（MERGEKIT_RAY_POOL_RUNTIME_ENV=1）")
            ray_pool = Pool(processes=num_workers, ray_remote_args=rargs)
            logger.info("[main] Ray 并行：workers=%s, num_gpus/worker=%s (TP)", num_workers, tp_size)
        else:
            num_workers = max(1, args.ray_num_gpus)
            rargs1: dict = {"num_gpus": 1}
            if (os.environ.get("MERGEKIT_RAY_POOL_RUNTIME_ENV") or "").strip().lower() in ("1", "true", "yes", "on"):
                rargs1["runtime_env"] = {"env_vars": {"GLOO_SOCKET_IFNAME": "lo"}}
            ray_pool = Pool(processes=num_workers, ray_remote_args=rargs1)
            logger.info("[main] Ray 并行：workers=%s, num_gpus/worker=1", num_workers)
        logger.info(
            "[main] ray_tp_summary ray_num_gpus=%s tp_size=%s ray_workers=%s MERGEKIT_VLLM_TP_SERIALIZE=%r CUDA_VISIBLE_DEVICES=%s",
            args.ray_num_gpus,
            tp_size,
            num_workers,
            os.environ.get("MERGEKIT_VLLM_TP_SERIALIZE", ""),
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        )
        problem.elementwise_runner = StarmapParallelization(ray_pool.starmap)
    else:
        logger.info("[main] ray_num_gpus<=1，串行评估")

    result_X = None
    result_F = None
    try:
        termination = ("n_iter", args.n_iter)
        if int(getattr(args, "max_evals", 0) or 0) > 0:
            # 使用 get_termination("n_eval", ...) 以确保与 pymoo 计数语义一致
            termination = get_termination("n_eval", int(args.max_evals))
            logger.info("[main] max_evals=%s：启用 termination=n_eval", int(args.max_evals))
        try:
            result = minimize(problem, algorithm, termination, seed=args.seed, verbose=True)
        except MaxEvalsReached as e:
            logger.info("[main] early stop: %s", e)
            result = None
        # 与 mergenetic Searcher 兼容：离散问题会做缩放；当前问题通常为连续
        if result is not None:
            result_X = result.X / 10 if getattr(problem, "discrete", False) else result.X
            result_F = result.F
            logger.info(
                "[main] Best solution found: %s. Best function value: %s",
                getattr(result, "X", None),
                getattr(result, "F", None),
            )

        # 与 Searcher 行为对齐：将 problem.results_df 写到 results_root
        if hasattr(problem, "results_df"):
            if isinstance(problem.results_df, pd.DataFrame):
                search_path = results_root / Path(f"{args.run_id}.csv")
                problem.results_df.to_csv(search_path)
            elif isinstance(problem.results_df, dict):
                for key, value in problem.results_df.items():
                    search_path = results_root / Path(f"{args.run_id}_{key}.csv")
                    value.to_csv(search_path)
    finally:
        # search() 抛错时也必须释放 Ray worker / GPU，否则容器内易残留显存与僵尸子进程
        if ray_pool is not None:
            try:
                ray_pool.close()
                ray_pool.join()
            except Exception as e:
                logger.warning("[main] ray pool teardown: %s", e)
        try:
            if ray.is_initialized():
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
        if best_x is None and result_X is not None:
            if hasattr(result_X, "__len__") and len(result_X) > 0:
                idx = int(np.argmin(result_F)) if result_F is not None else 0
                best_x = np.array(result_X[idx]).flatten().tolist()
            else:
                best_x = np.array(result_X).flatten().tolist()
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
    if len(sys.argv) >= 3 and sys.argv[1] == "--vllm-tp-eval-worker":
        try:
            _vllm_tp_eval_worker_main(sys.argv[2])
        except Exception as e:
            logger.exception("[vllm-tp-eval-worker] failed: %s", e)
            print(json.dumps({"error": str(e)[:2000]}), flush=True)
            sys.exit(1)
        sys.exit(0)
    main()
