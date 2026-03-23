"""
从 DB 读取并组装为与「文件数据」相同结构的只读层。

供 services 在「优先 DB、文件回退」时调用；需在 Flask app_context 内执行。
"""
import os

from app.repositories import (
    task_list_for_fusion_history,
    task_list_for_eval_history,
    evaluation_results_grouped_by_testset,
    model_list_all,
)


def _task_row_to_fusion_history_item(row):
    """将 Task 行转为 get_all_history 返回的单项格式。"""
    config = row.get("config") or {}
    task_type = row.get("task_type") or "merge"
    fusion_method = "Evolutionary (Ties-Dare)" if task_type == "merge_evolutionary" else "Recipe Config" if task_type == "recipe_apply" else "Standard"
    return {
        "id": row.get("id"),
        "type": task_type,
        "custom_name": config.get("custom_name") or config.get("display_name") or "",
        "created_at": row.get("created_at"),
        "status": row.get("status"),
        "fusion_method": fusion_method,
        "config": config,
    }


def _task_row_to_eval_history_item(row):
    """将 Task 行转为 get_all_eval_history 返回的单项格式。"""
    config = row.get("config") or {}
    return {
        "id": row.get("id"),
        "task_id": row.get("id"),
        "type": "eval_only",
        "model_name": config.get("model_name") or config.get("custom_name") or "Unknown Model",
        "custom_name": config.get("model_name") or config.get("custom_name"),
        "created_at": row.get("created_at"),
        "status": row.get("status"),
        "config": config,
    }


def get_fusion_history_from_db():
    """融合历史（与 get_all_history 同结构）。无数据或异常返回 None。"""
    try:
        rows = task_list_for_fusion_history()
        if not rows:
            return None
        return [_task_row_to_fusion_history_item(r) for r in rows]
    except Exception:
        return None


def get_eval_history_from_db():
    """评估历史（与 get_all_eval_history 同结构）。无数据或异常返回 None。"""
    try:
        rows = task_list_for_eval_history()
        if not rows:
            return None
        return [_task_row_to_eval_history_item(r) for r in rows]
    except Exception:
        return None


def get_leaderboard_from_db():
    """排行榜（与 load_leaderboard 同结构：dict[testset_id, list]）。无数据或异常返回 None。"""
    try:
        by_testset = evaluation_results_grouped_by_testset()
        if not by_testset:
            return None
        return by_testset
    except Exception:
        return None


def get_model_repo_list_from_db(merge_metadata_by_path_fn):
    """
    模型列表（与 model_repo_list 同结构）。
    merge_metadata_by_path_fn(path) 返回 metadata 或 None，用于补全 parent_models/recipe/dataset。
    无数据或异常返回 None。
    """
    try:
        rows = model_list_all()
        if not rows:
            return None
        out = []
        for m in rows:
            path = (m.get("path") or "").strip().rstrip("/")
            name = m.get("name") or ""
            meta = merge_metadata_by_path_fn(path) if path and merge_metadata_by_path_fn else None
            parent_models = []
            recipe = {}
            dataset = {}
            if meta:
                parent_models = [os.path.basename(p) for p in (meta.get("model_paths") or [])]
                recipe = {
                    "weights": (meta.get("recipe") or {}).get("weights"),
                    "method": (meta.get("recipe") or {}).get("method"),
                    "dtype": (meta.get("recipe") or {}).get("dtype"),
                    "library": (meta.get("recipe") or {}).get("library"),
                }
                dataset = {
                    "hf_dataset": meta.get("hf_dataset"),
                    "hf_subset": meta.get("hf_subset"),
                    "hf_subsets": meta.get("hf_subsets"),
                    "hf_split": meta.get("hf_split"),
                }
            out.append({
                "path": path,
                "name": name,
                "model_id": m.get("id"),
                "is_base": False,
                "parent_models": parent_models,
                "recipe": recipe,
                "dataset": dataset,
                "tag_names": m.get("tag_names") or [],
            })
        return out
    except Exception:
        return None
