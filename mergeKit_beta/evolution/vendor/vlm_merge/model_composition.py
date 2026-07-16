"""Strictly combine a merged language model with a complete VLM."""


def language_model_of(vlm):
    target = getattr(vlm, "language_model", None)
    if target is None and hasattr(vlm, "model"):
        target = getattr(vlm.model, "language_model", None)
    if target is None:
        raise ValueError("architecture_mismatch: VLM language model is missing")
    return target


def replace_language_model_weights(vlm, merged_lm) -> None:
    target = language_model_of(vlm)
    source = merged_lm.state_dict()
    expected = target.state_dict()
    if source.keys() != expected.keys():
        raise ValueError("architecture_mismatch: language tensor names differ")
    for name in source:
        if source[name].shape != expected[name].shape:
            raise ValueError("architecture_mismatch: %s shape differs" % name)
    target.load_state_dict(source, strict=True)


def materialize_full_vlm(merged_llm_dir, vlm_base_path, output_dir, dtype) -> None:
    """Save a standalone VLM after replacing only its compatible language tower."""
    import gc
    import os
    import shutil

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    vlm = None
    merged_lm = None
    processor = None
    try:
        vlm = AutoModelForImageTextToText.from_pretrained(
            vlm_base_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        merged_lm = AutoModelForCausalLM.from_pretrained(
            merged_llm_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        replace_language_model_weights(vlm, merged_lm)
        if os.path.lexists(output_dir):
            if os.path.islink(output_dir) or os.path.isfile(output_dir):
                os.unlink(output_dir)
            else:
                shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        vlm.save_pretrained(output_dir)
        processor = AutoProcessor.from_pretrained(vlm_base_path, trust_remote_code=True)
        processor.save_pretrained(output_dir)
    finally:
        del processor, merged_lm, vlm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
