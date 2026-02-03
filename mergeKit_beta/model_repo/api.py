"""模型仓库 API：列表、路径查询、注册融合模型"""
import os
import time
import uuid
from .storage import load_models, save_models

def list_models():
    return list(load_models().values())

def get_path(model_id):
    models = load_models()
    m = models.get(model_id)
    if not m:
        return None
    return m.get("path")

def register_merged_model(output_dir, name, recipe):
    models = load_models()
    model_id = str(uuid.uuid4())
    models[model_id] = {
        "model_id": model_id,
        "name": name,
        "path": os.path.abspath(output_dir) if isinstance(output_dir, str) else str(output_dir),
        "architecture": "Qwen2ForCausalLM",
        "model_type": "LLM",
        "is_merged": True,
        "recipe": recipe or {},
        "param_count": 0,
        "created_at": time.time(),
        "created_by": "system",
        "eval_history": [],
        "tags": [],
    }
    save_models(models)
    return model_id

import os
