#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def scan_safetensors(model_dir: Path, max_abs_threshold: float) -> dict:
    shard_files = sorted(model_dir.glob("model-*-of-*.safetensors"))
    results = {
        "checked_tensors": 0,
        "bad_tensors": 0,
        "examples": [],
        "shards": [],
    }
    for shard in shard_files:
        shard_bad = 0
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key).float()
                results["checked_tensors"] += 1
                has_nan = torch.isnan(t).any().item()
                has_inf = torch.isinf(t).any().item()
                max_abs = float(t.abs().max())
                if has_nan or has_inf or max_abs > max_abs_threshold:
                    results["bad_tensors"] += 1
                    shard_bad += 1
                    if len(results["examples"]) < 8:
                        results["examples"].append(
                            {
                                "shard": shard.name,
                                "tensor": key,
                                "has_nan": bool(has_nan),
                                "has_inf": bool(has_inf),
                                "max_abs": max_abs,
                            }
                        )
        results["shards"].append({"name": shard.name, "bad_tensors": shard_bad})
    return results


def smoke_generate(model_dir: Path, device: str, max_new_tokens: int) -> dict:
    prompt = "Question: 2+2=?\nChoices:\nA.1\nB.2\nC.3\nD.4\nAnswer:"
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    ins = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**ins).logits[0, -1].float()
        all_nan = bool(torch.isnan(logits).all().item())
        top_vals, top_ids = torch.topk(logits, 5)
        out = model.generate(
            **ins,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    decoded = tok.decode(out[0][ins["input_ids"].shape[1] :], skip_special_tokens=True)
    return {
        "all_nan_logits": all_nan,
        "top5_ids": [int(i) for i in top_ids.tolist()],
        "top5_vals": [float(v) for v in top_vals.tolist()],
        "sample_output": decoded[:200],
    }


def main():
    parser = argparse.ArgumentParser(description="Check model file and output health.")
    parser.add_argument("--model-dir", required=True, help="Model folder path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max-abs-threshold", type=float, default=1000.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--json-out", default="", help="Optional JSON report path")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    scan = scan_safetensors(model_dir, args.max_abs_threshold)
    smoke = smoke_generate(model_dir, args.device, args.max_new_tokens)

    healthy = (
        scan["bad_tensors"] == 0
        and not smoke["all_nan_logits"]
        and "!!!!" not in smoke["sample_output"]
    )
    report = {
        "model_dir": str(model_dir),
        "healthy": healthy,
        "scan": scan,
        "smoke": smoke,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        out_path = Path(args.json_out).resolve()
        os.makedirs(out_path.parent, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {out_path}")

    if not healthy:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
