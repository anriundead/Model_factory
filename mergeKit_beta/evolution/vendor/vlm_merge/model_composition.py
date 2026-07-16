"""Strictly combine a merged language model with a complete VLM."""


def load_merged_language_model_on_cpu(model_path, torch_dtype):
    """Keep the temporary source tower off GPU while copying its weights."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )


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
    if source.keys() == expected.keys():
        normalized = source
    else:
        normalized = {
            name.removeprefix("model."): tensor
            for name, tensor in source.items()
            if name.startswith("model.")
        }
        if (
            len(normalized) != len(source) - ("lm_head.weight" in source)
            or normalized.keys() != expected.keys()
        ):
            raise ValueError("architecture_mismatch: language tensor names differ")
    for name, tensor in normalized.items():
        if tensor.shape != expected[name].shape:
            raise ValueError("architecture_mismatch: %s shape differs" % name)
    target.load_state_dict(normalized, strict=True)
    if normalized is source:
        return

    source_head = source.get("lm_head.weight")
    target_head = getattr(vlm, "lm_head", None)
    target_head_weight = getattr(target_head, "weight", None)
    if source_head is None or target_head_weight is None:
        raise ValueError("architecture_mismatch: VLM lm_head is missing")
    if source_head.shape != target_head_weight.shape:
        raise ValueError("architecture_mismatch: lm_head.weight shape differs")
    target_head_weight.detach().copy_(source_head)


def materialize_full_vlm(merged_llm_dir, vlm_base_path, output_dir, dtype) -> None:
    """Save a standalone VLM after replacing only its compatible language tower."""
    import gc
    import os
    import shutil
    import tempfile

    output_dir = os.path.abspath(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError("output_dir already exists: %s" % output_dir)
    parent_dir = os.path.dirname(output_dir)
    os.makedirs(parent_dir, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=".%s." % os.path.basename(output_dir), dir=parent_dir)

    vlm = None
    merged_lm = None
    processor = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
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
        vlm.save_pretrained(staging_dir)
        processor = AutoProcessor.from_pretrained(vlm_base_path, trust_remote_code=True)
        processor.save_pretrained(staging_dir)
        os.rename(staging_dir, output_dir)
        staging_dir = None
    finally:
        del processor, merged_lm, vlm
        gc.collect()
        if staging_dir:
            shutil.rmtree(staging_dir, ignore_errors=True)
