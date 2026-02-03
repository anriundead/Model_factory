"""模型仓库存储：读写 model_repo/data/models.json"""
import json
import os
import uuid
from pathlib import Path

def _data_path():
    return Path(__file__).resolve().parent / "data" / "models.json"

def load_models():
    p = _data_path()
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("models", data) if isinstance(data.get("models"), dict) else {}

def save_models(models):
    p = _data_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"models": models}, f, ensure_ascii=False, indent=2)
