#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
from collections import OrderedDict

from transformers import AutoTokenizer
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from qwen2_5_text_causallm import Qwen2_5_VLTextForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlm",
        type=str,
        default="FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL",
        help="Huatuo 多模态模型 name_or_path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./HuatuoGPT-Vision-7B-LLM",
        help="保存抽取出的纯文本 LLM 目录",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='加载用的 device，例如 "cuda" 或 "cuda:0" 或 "cpu"',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"[INFO] Loading Huatuo VLM from {args.vlm}")
    vlm: Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True,
    )

    # 1. 拿到 VLM 的完整 state_dict（包含 language_model + visual + lm_head）
    print("[INFO] Collecting full state_dict from VLM ...")
    full_sd = vlm.state_dict()
    print(f"[INFO] Total params in VLM state_dict: {len(full_sd)}")

    # 2. 过滤：只保留 language_model + lm_head（并且把前缀改成 model.*）
    print("[INFO] Filtering LLM-only parameters (language_model + lm_head) ...")
    llm_sd = OrderedDict()
    for k, v in full_sd.items():
        # 文本塔：model.language_model.xxx -> model.xxx
        if k.startswith("model.language_model."):
            new_k = k.replace("model.language_model.", "model.")
            llm_sd[new_k] = v
        # 输出头：lm_head.xxx 原样保留
        elif k.startswith("lm_head."):
            llm_sd[k] = v
        # 视觉塔：忽略
        else:
            continue

    print(f"[INFO] LLM param count after filtering & renaming: {len(llm_sd)}")

    # 3. 用 language_model 的 config 构建 CausalLM wrapper
    print("[INFO] Building pure LLM from language_model.config ...")
    lm_config = vlm.language_model.config  # Qwen2_5_VLTextConfig
    llm = Qwen2_5_VLTextForCausalLM(lm_config)

    # 4. 把过滤后的权重 load 进去
    print("[INFO] Loading filtered weights into Qwen2_5_VLTextForCausalLM ...")
    missing, unexpected = llm.load_state_dict(llm_sd, strict=False)

    if missing:
        print("[WARN] Missing keys when loading into LLM:")
        for k in missing:
            print("   -", k)
    if unexpected:
        print("[WARN] Unexpected keys when loading into LLM:")
        for k in unexpected:
            print("   -", k)

    sd_after = llm.state_dict()
    assert "model.embed_tokens.weight" in sd_after, "[ERROR] 抽取后的 LLM 里没有 model.embed_tokens.weight！"
    assert "model.norm.weight" in sd_after, "[ERROR] 抽取后的 LLM 里没有 model.norm.weight！"
    assert "lm_head.weight" in sd_after, "[ERROR] 抽取后的 LLM 里没有 lm_head.weight！"

    # 5. 保存 LLM（权重 + config）
    print(f"[INFO] Saving pure LLM to {args.output_dir}")
    llm.save_pretrained(args.output_dir)

    # 6. 顺便把 tokenizer 也存一份
    print("[INFO] Saving tokenizer (from VLM) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.vlm, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
