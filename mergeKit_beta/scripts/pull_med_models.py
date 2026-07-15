#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉取医疗相关候选模型（HF / HF mirror），并在下载前做严格校验：

- 必须是 full weights（至少包含 .safetensors 或 pytorch_model*.bin）
- 禁止 GGUF / LoRA-only / 量化权重（AWQ/GPTQ等）进入清单
- config.json 解析出的 (hidden_size, num_hidden_layers, is_vlm) 必须能匹配你本地已有架构组
- 下载落盘到 /home/a/Model_factory_data/models/<local_dirname>

用法：
  python3 scripts/pull_med_models.py --dry-run
  python3 scripts/pull_med_models.py

说明：
- 默认使用 HF 镜像端点（与容器一致）：HF_ENDPOINT=https://hf-mirror.com
- 如模型为 gated/需要登录，会自动跳过并输出原因（不会半拉子落盘）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except Exception as e:
    raise SystemExit(
        "缺少 huggingface_hub：请在宿主机 python 环境安装 `pip install huggingface_hub`。错误：%s" % (e,)
    )


MODELS_ROOT = Path("/home/a/Model_factory_data/models")
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").strip() or "https://hf-mirror.com"
HF_HOME = os.environ.get("HF_HOME", "/home/a/Workspace/.hf_home")
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", str(Path(HF_HOME) / "hub"))


@dataclass(frozen=True)
class ArchKey:
    hidden_size: int
    num_hidden_layers: int
    is_vlm: bool


@dataclass(frozen=True)
class Candidate:
    repo_id: str
    local_dirname: str
    # expected target arch group in your local pool (must match)
    expected: ArchKey
    notes: str


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_arch_key_from_config(cfg: dict) -> ArchKey:
    hs = cfg.get("hidden_size")
    nhl = cfg.get("num_hidden_layers")
    mt = (cfg.get("model_type") or "").strip().lower()
    for subk in ("text_config", "decoder_config", "language_config"):
        sub = cfg.get(subk)
        if isinstance(sub, dict):
            hs = hs or sub.get("hidden_size")
            nhl = nhl or sub.get("num_hidden_layers") or sub.get("n_layer") or sub.get("n_layers") or sub.get("depth")
            mt = mt or (sub.get("model_type") or "").strip().lower()
    is_vlm = bool(
        cfg.get("vision_config") is not None
        or any(cfg.get(k) is not None for k in ("image_token_id", "vision_start_token_id", "vision_token_id"))
        or ("_vl" in mt)
    )
    if hs is None or nhl is None:
        raise ValueError("config.json 缺少 hidden_size/num_hidden_layers（或回退字段）")
    return ArchKey(int(hs), int(nhl), bool(is_vlm))


def _looks_like_quantized_or_adapter(files: Iterable[str]) -> tuple[bool, str]:
    """
    Return (reject, reason).
    """
    fs = [f.lower() for f in files]
    joined = " ".join(fs)
    # hard reject formats
    if any(f.endswith(".gguf") for f in fs):
        return True, "检测到 GGUF 文件"
    if any("gptq" in f or "awq" in f or "bitsandbytes" in f or "bnb" in f for f in fs):
        return True, "检测到量化相关文件名（gptq/awq/bnb）"
    if "adapter_model.safetensors" in fs or "adapter_model.bin" in fs:
        # could still have full weights; keep checking below
        pass
    # full weights check
    has_full = any(f.endswith(".safetensors") for f in fs) or any(re.match(r"pytorch_model.*\.bin$", f) for f in fs)
    if not has_full:
        return True, "未检测到 full weights（.safetensors 或 pytorch_model*.bin）"
    # adapter-only heuristic: only adapter weights present
    only_adapter = has_full and all(
        (
            (f.endswith(".safetensors") and "adapter" in f)
            or f.endswith((".json", ".md", ".txt", ".model", ".py", ".gitattributes", ".lock"))
            or f.startswith("tokenizer")
            or f in ("config.json", "generation_config.json", "special_tokens_map.json", "tokenizer_config.json")
            or f.endswith((".tiktoken", ".sentencepiece", ".spm"))
        )
        for f in fs
    )
    if only_adapter:
        return True, "疑似 LoRA/Adapter-only（仅检测到 adapter 权重）"
    # also reject obvious quantization_config in config.json via file name scan
    if "quantization" in joined:
        return True, "检测到 quantization 相关文件/目录"
    return False, ""


def _get_local_existing_arch_groups(models_root: Path) -> dict[ArchKey, list[str]]:
    groups: dict[ArchKey, list[str]] = {}
    if not models_root.is_dir():
        return groups
    for cfg_path in models_root.glob("*/config.json"):
        name = cfg_path.parent.name
        try:
            cfg = _read_json(cfg_path)
            key = _infer_arch_key_from_config(cfg)
        except Exception:
            continue
        groups.setdefault(key, []).append(name)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只检查，不下载")
    args = ap.parse_args()

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    # Your local existing arch groups (as ground truth)
    existing = _get_local_existing_arch_groups(MODELS_ROOT)
    if not existing:
        print("ERROR: 未在 %s 下发现任何本地模型 config.json，无法对齐架构组。" % MODELS_ROOT, file=sys.stderr)
        return 2

    # Candidates chosen to match your existing arch groups (no duplicates)
    # Qwen2.5 7B LLM: hidden_size=3584 layers=28 is_vlm=False
    # Llama3 8B LLM: hidden_size=4096 layers=32 is_vlm=False
    # Qwen3 8B LLM: hidden_size=4096 layers=36 is_vlm=False
    # Qwen2.5-VL 7B VLM: hidden_size=3584 layers=28 is_vlm=True
    candidates: list[Candidate] = [
        Candidate(
            repo_id="FreedomIntelligence/HuatuoGPT-o1-7B",
            local_dirname="HuatuoGPT-o1-7B",
            expected=ArchKey(3584, 28, False),
            notes="医疗推理向（Qwen2.5 7B 线），补齐你现有 Qwen2.5-7B 父代多样性。",
        ),
        Candidate(
            repo_id="OpenMeditron/Meditron3-8B",
            local_dirname="Meditron3-8B",
            expected=ArchKey(4096, 32, False),
            notes="临床医疗向（Llama3 8B 线），与现有 Llama3 医疗父代互补。",
        ),
        Candidate(
            repo_id="Intelligent-Internet/II-Medical-8B",
            local_dirname="II-Medical-8B",
            expected=ArchKey(4096, 36, False),
            notes="医疗推理向（Qwen3 8B 线），与你现有 Qwen3-8B-Hippocratesv1 可互配。",
        ),
        # 医疗 VLM（Qwen2.5-VL 7B 线）：只纳入 full weights（非 LoRA/非量化），并确保与本地 Qwen2.5-VL-7B 组匹配
        Candidate(
            repo_id="lingshu-medical-mllm/Lingshu-7B",
            local_dirname="Lingshu-7B",
            expected=ArchKey(3584, 28, True),
            notes="医疗多模态（基于 Qwen2.5-VL 7B 架构）full weights 候选；用于扩充医疗 VLM 父代池。",
        ),
        Candidate(
            repo_id="UCSC-VLAA/MedVLThinker-7B-RL_m23k-RL_PMC",
            local_dirname="MedVLThinker-7B-RL_m23k-RL_PMC",
            expected=ArchKey(3584, 28, True),
            notes="医疗多模态 VLM（Qwen2.5-VL 7B 架构）候选；具体是否 full weights 由文件检查决定。",
        ),
        Candidate(
            repo_id="eliem/Qwen2.5-VL-7B-radiology",
            local_dirname="Qwen2.5-VL-7B-radiology",
            expected=ArchKey(3584, 28, True),
            notes="放射/影像方向 Qwen2.5-VL-7B 微调候选；仅当仓库提供 full weights 且非量化时才会拉取。",
        ),
    ]

    api = HfApi(endpoint=HF_ENDPOINT)
    ok: list[tuple[Candidate, ArchKey, list[str]]] = []
    skipped: list[tuple[Candidate, str]] = []

    for c in candidates:
        target_dir = MODELS_ROOT / c.local_dirname
        if target_dir.exists():
            skipped.append((c, "本地已存在目录（避免重复）: %s" % target_dir))
            continue

        # pre-check: expected arch group must exist locally (ensures merge compatibility with your pool)
        if c.expected not in existing:
            skipped.append((c, "本地不存在目标架构组 %s（先补齐基座组再拉取）" % (c.expected,)))
            continue

        try:
            info = api.model_info(c.repo_id)
            files = [s.rfilename for s in (info.siblings or [])]
        except Exception as e:
            skipped.append((c, "无法访问/可能 gated/网络问题: %s" % (e,)))
            continue

        reject, reason = _looks_like_quantized_or_adapter(files)
        if reject:
            skipped.append((c, "不满足 full weights/非量化要求: %s" % reason))
            continue

        # fetch config.json only
        try:
            cfg_path = Path(
                hf_hub_download(
                    repo_id=c.repo_id,
                    filename="config.json",
                    endpoint=HF_ENDPOINT,
                )
            )
            cfg = _read_json(cfg_path)
            actual_key = _infer_arch_key_from_config(cfg)
        except Exception as e:
            skipped.append((c, "读取/解析 config.json 失败: %s" % (e,)))
            continue

        if actual_key != c.expected:
            skipped.append((c, "架构不匹配：expected=%s actual=%s" % (c.expected, actual_key)))
            continue

        ok.append((c, actual_key, files))

    print("=== 本地架构组（用于匹配） ===")
    for k, names in sorted(existing.items(), key=lambda x: (x[0].is_vlm, x[0].hidden_size, x[0].num_hidden_layers)):
        print("- %s count=%d examples=%s" % (k, len(names), ", ".join(sorted(names)[:3])))

    print("\n=== 通过检查、可拉取清单（不重复，且可与你本地模型融合） ===")
    if not ok:
        print("(空)")
    for c, key, _files in ok:
        print("- %s -> %s  # %s" % (c.repo_id, MODELS_ROOT / c.local_dirname, c.notes))

    print("\n=== 跳过清单（含原因） ===")
    if not skipped:
        print("(空)")
    for c, why in skipped:
        print("- %s  SKIP: %s" % (c.repo_id, why))

    if args.dry_run:
        return 0

    if not ok:
        print("\n无可拉取项，结束。")
        return 0

    # Download
    for c, _key, _files in ok:
        dst = MODELS_ROOT / c.local_dirname
        print("\n=== 下载: %s -> %s ===" % (c.repo_id, dst))
        try:
            snapshot_download(
                repo_id=c.repo_id,
                local_dir=str(dst),
                local_dir_use_symlinks=False,
                endpoint=HF_ENDPOINT,
                resume_download=True,
                cache_dir=HF_HUB_CACHE,
            )
        except Exception as e:
            print("ERROR: 下载失败 %s: %s" % (c.repo_id, e), file=sys.stderr)
            continue

        # post-check: must have config.json + some weights
        cfg = dst / "config.json"
        if not cfg.is_file():
            print("ERROR: 下载后缺少 config.json（视为失败）: %s" % dst, file=sys.stderr)
            continue
        weight_ok = any(p.suffix == ".safetensors" for p in dst.glob("**/*.safetensors")) or any(
            p.name.startswith("pytorch_model") and p.suffix == ".bin" for p in dst.glob("**/*.bin")
        )
        if not weight_ok:
            print("ERROR: 下载后未检测到权重文件（视为失败）: %s" % dst, file=sys.stderr)
            continue
        print("OK: 下载与基本校验通过: %s" % dst)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
