# vlm_fitness.py
"""
VLM 融合搜索的 fitness：用 VLM（视觉塔 + 替换后的文本塔）在 CMMMU 等数据集上评测。
merged_llm_dir 为 TIES-DARE 融合后的 LLM；用其权重替换 vlm_path 的 language_model 后做多模态推理。
"""

from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)

# 默认 VLM（提供视觉塔 + tokenizer）
DEFAULT_VLM = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"


def _dataset_to_list_of_dicts(dataset) -> list[dict[str, Any]]:
    """将 HF Dataset 转为 list[dict]，避免 Ray 内 dataclass 错误。"""
    out: list[dict[str, Any]] = []
    try:
        if hasattr(dataset, "to_pandas"):
            df = dataset.to_pandas()
            return df.to_dict("records")
    except Exception:
        pass
    try:
        cols = getattr(dataset, "column_names", None) or []
        for i in range(len(dataset)):
            row = dataset[i]
            out.append(dict(row) if hasattr(row, "keys") else dict(zip(cols, row)))
    except Exception:
        pass
    return out


def load_prompt_cfg(prompt_yaml: str, project_root: Path) -> dict:
    p = Path(prompt_yaml)
    if not p.is_absolute():
        p = project_root / p
    if not p.exists():
        alt = project_root / "eval" / (p.name or "prompt.yaml")
        if alt.exists():
            p = alt
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cmmmu_options_list(example: dict) -> list[str]:
    """CMMMU 行内选项为 option1..option4，而非 choices 列表。"""
    opts = example.get("options", example.get("choices", []))
    if isinstance(opts, (list, tuple)) and opts:
        return [str(x) for x in opts if x is not None and str(x).strip()]
    out: list[str] = []
    for i in range(1, 5):
        v = example.get(f"option{i}")
        if v is not None and str(v).strip():
            out.append(str(v).strip())
    return out


def _cmmmu_first_image(example: dict):
    """CMMMU 图像字段为 image_1..image_5（或 image），不是 image。"""
    for k in ("image", "image_1", "image_2", "image_3", "image_4", "image_5"):
        im = example.get(k)
        if im is not None:
            return im
    return None


def _coerce_hf_image_to_pil(im: Any) -> Any:
    """将 datasets Image 行（常为 dict: bytes/path）或 PIL，转为 Qwen2.5-VL processor 可接受的 PIL。"""
    if im is None:
        return None
    try:
        from PIL import Image as PILImage
    except ImportError:
        return im
    if isinstance(im, PILImage.Image):
        return im.convert("RGB") if im.mode != "RGB" else im
    if isinstance(im, dict):
        b = im.get("bytes")
        if b:
            return PILImage.open(io.BytesIO(b)).convert("RGB")
        p = im.get("path")
        if p and isinstance(p, str) and os.path.isfile(p):
            return PILImage.open(p).convert("RGB")
        return None
    if isinstance(im, (bytes, bytearray, memoryview)):
        return PILImage.open(io.BytesIO(bytes(im))).convert("RGB")
    return im


def _normalize_cmmmu_gold(answer) -> str:
    """将标签规范为单个大写字母 A-D。"""
    if answer is None:
        return ""
    s = str(answer).strip().upper()
    if not s or s == "?":
        return ""
    if s in ("A", "B", "C", "D"):
        return s
    if s.isdigit():
        i = int(s)
        if 0 <= i <= 3:
            return chr(65 + i)
    if len(s) >= 1 and s[0] in "ABCD":
        return s[0]
    return ""


def build_cmmmu_prompt(example: dict, prompt_cfg: dict) -> str:
    """根据 prompt.yaml 构建 CMMMU 题目 prompt。"""
    question = example.get("question", "")
    options = _cmmmu_options_list(example)
    options_block = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
    template = (prompt_cfg.get("multi_choice_example_format") or [""])[0]
    if "{}" in template:
        return template.format(question, options_block).strip() + " "
    return f"问题：{question}\n选项：\n{options_block}\n正确答案： "


def parse_choice(output_text: str) -> str:
    """
    从模型输出中提取 CMMMU 多选题的 A-D 答案。

    Qwen2.5-VL 的 `generate()` 返回通常包含完整 prompt，导致首行是“问题/选项”，
    不能直接取 `first_line[0]`。这里改为优先截取“正确答案”之后的片段，再在片段里提取首个 A-D。
    """
    s = (output_text or "").strip()
    if not s:
        return ""

    # CMMMU prompt 固定包含：正确答案：
    pos = s.rfind("正确答案")
    tail = s[pos + len("正确答案") :] if pos != -1 else s
    tail = tail.lstrip(" \t\r\n:：")

    m = re.search(r"([ABCD])", tail)
    if m:
        return m.group(1)

    # fallback：从后往前找以 A-D 开头的行
    for ln in reversed([x.strip() for x in s.splitlines() if x.strip()]):
        if ln and ln[0].upper() in "ABCD":
            return ln[0].upper()
    return ""


def vlm_cmmmu_fitness(
    merged_llm_dir: str,
    vlm_path: str = DEFAULT_VLM,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    hf_dataset: str = "m-a-p/CMMMU",
    hf_subset: str = "health_and_medicine",
    hf_subsets: list | None = None,
    hf_split: str = "val",
    max_samples: int | None = 64,
    prompt_yaml: str = "eval/prompt.yaml",
    project_root: Path | None = None,
) -> float:
    """
    用 VLM（vision + 替换后的 merged LLM）在 CMMMU 上评测。
    merged_llm_dir: 当前 trial 融合后的 LLM 路径。
    hf_subsets: 若提供则加载多个子集并合并；否则仅用 hf_subset。
    返回 accuracy。
    """
    try:
        from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
    root = project_root or Path(__file__).resolve().parent
    prompt_cfg = load_prompt_cfg(prompt_yaml, root)

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    vlm.eval()

    merged_lm = AutoModelForCausalLM.from_pretrained(
        merged_llm_dir,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    lm_target = getattr(vlm, "language_model", None)
    if lm_target is None and hasattr(vlm, "model") and hasattr(vlm.model, "language_model"):
        lm_target = vlm.model.language_model
    if lm_target is None:
        raise RuntimeError("VLM 上找不到 language_model（transformers 结构变更？）")
    lm_target.load_state_dict(merged_lm.state_dict(), strict=False)
    del merged_lm
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)

    # Qwen2.5-VL 需要在 text 里包含图像占位 token（input_ids 中的 image_token_id），
    # 否则会报：Image features and image tokens do not match（tokens=0, features=...）
    image_token_str = None
    try:
        image_token_id = getattr(vlm.config, "image_token_id", None)
        tok = getattr(processor, "tokenizer", None)
        if image_token_id is not None and tok is not None and hasattr(tok, "decode"):
            image_token_str = tok.decode([int(image_token_id)]).strip()
    except Exception:
        image_token_str = None
    if not image_token_str:
        image_token_str = "<|image_pad|>"

    # CMMMU 的 test 分割答案常为 '?'（占位），无法计算有意义的 fitness；进化阶段应用 validation
    # 注意 Hub ID 为 CMMMU（小写 cmmmu，非 cmmu），旧写法 "cmmu" in ds 永假
    load_split = hf_split
    ds_l = (hf_dataset or "").lower()
    if ("cmmmu" in ds_l or "cmmu" in ds_l) and load_split in ("test", "testing"):
        load_split = "validation"
        logger.info(
            "[VLM fitness] CMMMU 在 split=test 下无公开标签，已改用 split=validation 计算准确率"
        )

    subsets_to_load = hf_subsets if hf_subsets else [hf_subset]
    cache_dir = os.environ.get("HF_DATASETS_CACHE") or None
    samples = []
    last_used_split: str | None = None
    for sub in subsets_to_load:
        last_err = None
        loaded = None
        used_sp = load_split
        for sp in (load_split, "validation", "val"):
            try:
                loaded = load_dataset(hf_dataset, sub, split=sp, trust_remote_code=True, cache_dir=cache_dir)
                used_sp = sp
                break
            except Exception as e:
                last_err = e
        if loaded is None:
            logger.error("[VLM fitness] 无法加载子集 %s: %s", sub, last_err)
            continue
        if used_sp != load_split:
            logger.info("[VLM fitness] 子集 %s 实际使用 split=%s", sub, used_sp)
        last_used_split = used_sp
        samples.extend(_dataset_to_list_of_dicts(loaded))
    if max_samples and len(samples) > max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, max_samples)

    correct = 0
    evaluated = 0
    for ex in samples:
        prompt = build_cmmmu_prompt(ex, prompt_cfg)
        if image_token_str and image_token_str not in prompt:
            prompt = f"{image_token_str}\n{prompt}"
        image = _coerce_hf_image_to_pil(_cmmmu_first_image(ex))
        if image is None:
            continue
        gold = _normalize_cmmmu_gold(ex.get("answer"))
        if not gold:
            continue
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        # 勿将 input_ids / attention_mask 等转为 bfloat16，embedding 需要 Long 索引
        moved: dict[str, Any] = {}
        for k, v in inputs.items():
            if not hasattr(v, "to"):
                moved[k] = v
            elif isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                moved[k] = v.to(device, dtype=torch_dtype)
            elif isinstance(v, torch.Tensor):
                moved[k] = v.to(device)
            else:
                moved[k] = v
        inputs = moved
        with torch.no_grad():
            out_ids = vlm.generate(**inputs, max_new_tokens=64)
        out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        pred = parse_choice(out_text)
        evaluated += 1
        if pred and pred == gold:
            correct += 1

    acc = correct / evaluated if evaluated > 0 else 0.0
    logger.info(
        "[VLM fitness] merged=%s acc=%.4f (%s/%s) split_used=%s split_requested=%s",
        merged_llm_dir,
        acc,
        correct,
        evaluated,
        last_used_split or load_split,
        hf_split,
    )
    try:
        del vlm, processor
        torch.cuda.empty_cache()
    except Exception:
        pass
    return float(acc)
