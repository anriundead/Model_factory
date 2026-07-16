"""Structural inspection for local Hugging Face model directories."""

from dataclasses import dataclass
import glob
import hashlib
import json
import os
from typing import Sequence


_IMAGE_TOKEN_NAMES = (
    "image_token_id",
    "vision_start_token_id",
    "vision_token_id",
    "video_token_id",
)
_LANGUAGE_SIGNATURE_NAMES = ("hidden_size", "num_hidden_layers", "vocab_size")
_VISUAL_WEIGHT_PREFIXES = (
    "visual.",
    "model.visual.",
    "vision_tower.",
    "model.vision_tower.",
    "vision_model.",
)
_VLM_MODEL_TYPES = {
    "aria",
    "chameleon",
    "cogvlm",
    "cogvlm2",
    "deepseek_vl",
    "deepseek_vl_v2",
    "gemma3",
    "gemma3n",
    "idefics",
    "idefics2",
    "idefics3",
    "internvl",
    "llava",
    "llava_next",
    "llava_next_video",
    "llava_onevision",
    "mllama",
    "molmo",
    "paligemma",
    "phi3_v",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
}
_VLM_ARCHITECTURE_PREFIXES = (
    "AriaForConditionalGeneration",
    "ChameleonForConditionalGeneration",
    "CogVLM",
    "DeepseekVLV2",
    "DeepseekVL",
    "Gemma3ForConditionalGeneration",
    "Idefics",
    "InternVL",
    "Llava",
    "MllamaForConditionalGeneration",
    "MolmoForCausalLM",
    "PaliGemmaForConditionalGeneration",
    "Phi3VForCausalLM",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
)
_WEIGHT_FILE_PATTERNS = (
    "*.safetensors",
    "pytorch_model*.bin",
    "adapter_model*.bin",
    "*.gguf",
)


@dataclass(frozen=True)
class ModelInspection:
    path: str
    model_type: str
    architectures: Sequence[str]
    is_vlm: bool
    is_complete_vlm: bool
    processor_class: str | None
    image_token_ids: dict[str, int]
    visual_weight_count: int
    language_weight_count: int
    language_signature: tuple[int | None, int | None, int | None]
    config_sha256: str


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON object required: %s" % path)
    return value


def _read_weight_keys(model_path: str) -> set[str]:
    index_paths = sorted(glob.glob(os.path.join(model_path, "*.safetensors.index.json")))
    if index_paths:
        keys = set()
        for index_path in index_paths:
            weight_map = _read_json(index_path).get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError("weight_map is required: %s" % index_path)
            keys.update(str(key) for key in weight_map)
        return keys

    keys = set()
    for shard_path in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        from safetensors import safe_open

        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def _weight_file_paths(model_path: str) -> list[str]:
    referenced = set()
    for index_path in sorted(glob.glob(os.path.join(model_path, "*.safetensors.index.json"))):
        weight_map = _read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("weight_map is required: %s" % index_path)
        referenced.update(str(value) for value in weight_map.values())
    if not referenced:
        for pattern in _WEIGHT_FILE_PATTERNS:
            referenced.update(os.path.basename(path) for path in glob.glob(os.path.join(model_path, pattern)))

    paths = []
    for relative in sorted(referenced):
        if os.path.isabs(relative) or relative.replace("\\", "/").split("/").count(".."):
            raise ValueError("unsafe weight file path: %s" % relative)
        path = os.path.realpath(os.path.join(model_path, relative))
        if os.path.commonpath((model_path, path)) != model_path or os.path.islink(os.path.join(model_path, relative)):
            raise ValueError("unsafe weight file path: %s" % relative)
        if not os.path.isfile(path):
            raise ValueError("weight file is missing: %s" % relative)
        paths.append(path)
    if not paths:
        raise ValueError("model contains no supported weight files: %s" % model_path)
    return paths


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_weight_fingerprint(path: str) -> dict:
    """Hash model weight bytes without slowing ordinary structural inspection."""
    model_path = os.path.realpath(os.path.abspath(path))
    if not os.path.isdir(model_path):
        raise ValueError("model path is not a directory: %s" % path)

    entries = []
    combined = hashlib.sha256()
    total_bytes = 0
    for weight_path in _weight_file_paths(model_path):
        relative = os.path.relpath(weight_path, model_path).replace(os.sep, "/")
        size_bytes = os.path.getsize(weight_path)
        sha256 = _file_sha256(weight_path)
        entries.append({"path": relative, "size_bytes": size_bytes, "sha256": sha256})
        total_bytes += size_bytes
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size_bytes).encode("ascii"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(sha256))

    return {
        "source_path": model_path,
        "weights_sha256": combined.hexdigest(),
        "weight_bytes": total_bytes,
        "weight_files": entries,
    }


def _read_processor_class(model_path: str) -> str | None:
    image_processor_type = None
    for name in ("processor_config.json", "preprocessor_config.json"):
        path = os.path.join(model_path, name)
        if not os.path.isfile(path):
            continue
        config = _read_json(path)
        value = config.get("processor_class")
        if isinstance(value, str) and value:
            return value
        value = config.get("image_processor_type")
        if image_processor_type is None and isinstance(value, str) and value:
            image_processor_type = value
    return image_processor_type


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected integer model config value") from exc


def _has_vlm_config_signal(config: dict, model_type: str, architectures: Sequence[str], image_token_ids: dict[str, int]) -> bool:
    return bool(
        config.get("vision_config") is not None
        or config.get("visual_config") is not None
        or image_token_ids
        or model_type in _VLM_MODEL_TYPES
        or any(architecture.startswith(_VLM_ARCHITECTURE_PREFIXES) for architecture in architectures)
    )


def inspect_model(path: str) -> ModelInspection:
    model_path = os.path.realpath(os.path.abspath(path))
    if not os.path.isdir(model_path):
        raise ValueError("model path is not a directory: %s" % path)

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "rb") as handle:
        config_bytes = handle.read()
    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid config.json: %s" % model_path) from exc
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object: %s" % model_path)

    architectures = tuple(str(value) for value in config.get("architectures") or ())
    model_type = str(config.get("model_type") or "").lower()
    image_token_ids = {
        name: _as_int(config[name])
        for name in _IMAGE_TOKEN_NAMES
        if config.get(name) is not None
    }
    weight_keys = _read_weight_keys(model_path)
    visual_weight_count = sum(key.startswith(_VISUAL_WEIGHT_PREFIXES) for key in weight_keys)
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    language_signature = tuple(_as_int(text_config.get(name)) for name in _LANGUAGE_SIGNATURE_NAMES)
    is_vlm = _has_vlm_config_signal(config, model_type, architectures, image_token_ids)
    processor_class = _read_processor_class(model_path)

    return ModelInspection(
        path=model_path,
        model_type=model_type,
        architectures=architectures,
        is_vlm=is_vlm,
        is_complete_vlm=bool(is_vlm and processor_class and image_token_ids and visual_weight_count),
        processor_class=processor_class,
        image_token_ids=image_token_ids,
        visual_weight_count=visual_weight_count,
        language_weight_count=len(weight_keys) - visual_weight_count,
        language_signature=language_signature,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )


def _inspect_complete(path: str | None) -> ModelInspection | None:
    if not path:
        return None
    try:
        inspection = inspect_model(path)
    except (OSError, ValueError):
        return None
    return inspection if inspection.is_complete_vlm else None


def resolve_vlm_base(recipe: dict, override_path: str | None = None) -> ModelInspection:
    if override_path is not None:
        inspection = _inspect_complete(override_path)
        if inspection is None:
            raise ValueError("vlm_base_missing: admin override is not a complete VLM")
        return inspection

    recorded = recipe.get("vlm_base") if isinstance(recipe.get("vlm_base"), dict) else {}
    recorded_path = recorded.get("source_path")
    try:
        recorded_inspection = inspect_model(recorded_path) if recorded_path else None
    except (OSError, ValueError):
        recorded_inspection = None
    expected_fingerprint = recorded.get("config_sha256")
    expected_weights = recorded.get("weights_sha256")
    if (expected_fingerprint or expected_weights) and recorded_inspection is None:
        raise ValueError("source_fingerprint_mismatch: recorded VLM source is unavailable")
    if expected_fingerprint and recorded_inspection is not None:
        if recorded_inspection.config_sha256 != expected_fingerprint:
            raise ValueError("source_fingerprint_mismatch: recorded VLM config differs from source")
    if expected_weights and recorded_inspection is not None:
        try:
            actual_weights = model_weight_fingerprint(recorded_inspection.path)["weights_sha256"]
        except (OSError, ValueError) as exc:
            raise ValueError("source_fingerprint_mismatch: recorded VLM weights are unavailable") from exc
        if actual_weights != expected_weights:
            raise ValueError("source_fingerprint_mismatch: recorded VLM weights differ from source")
    if recorded_inspection is not None and recorded_inspection.is_complete_vlm:
        return recorded_inspection

    candidates = [recipe.get("vlm_path")]
    candidates.extend(recipe.get("model_paths") or [])
    for candidate in candidates:
        inspection = _inspect_complete(candidate)
        if inspection is not None:
            return inspection
    raise ValueError("vlm_base_missing: no complete VLM parent was found")


def assert_language_compatible(model_paths: list[str], vlm_inspection: ModelInspection) -> None:
    expected = vlm_inspection.language_signature
    if None in expected:
        raise ValueError("architecture_mismatch: VLM language signature is incomplete")
    for path in model_paths:
        actual = inspect_model(path).language_signature
        if actual != expected:
            raise ValueError("architecture_mismatch: %s has %s, expected %s" % (path, actual, expected))
