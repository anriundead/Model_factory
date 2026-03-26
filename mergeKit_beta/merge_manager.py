"""
融合任务执行模块 - 使用 mergenetic + mergekit 执行模型融合，并维护 metadata.json
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import re
import yaml
import math
import shlex

from config import Config
from core.process_manager import ProcessManager

# 评估用（与 mergeKit_alpha 一致）
STANDARD_BENCHMARKS = getattr(Config, "STANDARD_BENCHMARKS", ["hellaswag", "arc_easy", "boolq", "winogrande"])

# 路径与配置
MODEL_POOL_PATH = Config.MODEL_POOL_PATH
MERGE_DIR = Config.MERGE_DIR
LOGS_DIR = Config.LOGS_DIR
RECIPES_DIR = getattr(Config, "RECIPES_DIR", None) or os.path.join(Config.PROJECT_ROOT, "recipes")
MERGENETIC_PYTHON = Config.MERGENETIC_PYTHON

os.makedirs(MERGE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 统一 metadata 写入 + 可选 DB 同步
# ---------------------------------------------------------------------------

def _sync_metadata_to_db(task_id, data):
    """将 metadata 关键字段同步到 DB Task 表（需 Flask app context）。"""
    try:
        from flask import current_app
        if not current_app:
            return
    except (ImportError, RuntimeError):
        return
    try:
        from app.extensions import db
        from app.models import Task
        task = db.session.get(Task, task_id)
        if task is None:
            return
        status_map = {"success": "completed", "error": "failed"}
        raw_status = data.get("status")
        mapped = status_map.get(raw_status, raw_status)
        if mapped:
            task.status = mapped
        for attr in ("custom_name", "error", "duration_seconds", "model_path"):
            if attr in data:
                setattr(task, attr, data[attr])
        cfg = task.config or {}
        for k in ("models", "weights", "method", "dtype", "hf_dataset",
                   "hf_subset", "hf_split", "dataset", "output_path",
                   "metrics", "base_name", "recipe_id", "testset_id"):
            if k in data:
                cfg[k] = data[k]
        task.config = cfg
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(task, "config")
        from datetime import datetime as _dt
        task.updated_at = _dt.utcnow()
        if mapped == "completed" and not task.finished_at:
            task.finished_at = _dt.utcnow()
        db.session.commit()
    except Exception as exc:
        _logger.debug("[_sync_metadata_to_db] %s", exc)


def _write_metadata(task_id, output_path, data, sync_db=True):
    """统一写 metadata.json，可选同步 DB。"""
    meta_path = os.path.join(output_path, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if sync_db:
        _sync_metadata_to_db(task_id, data)


def _popen_group_kwargs():
    return ProcessManager.create_process_group_kwargs()


def _get_conda_activate_cmd(cmd_list):
    """
    Wraps a command list in a shell command that activates the mergenetic environment.
    """
    # Try to find conda.sh based on MERGENETIC_PYTHON
    conda_sh = None
    if MERGENETIC_PYTHON and "envs" in MERGENETIC_PYTHON:
        # e.g. /home/a/miniconda3/envs/mergenetic/bin/python
        parts = MERGENETIC_PYTHON.split("/envs/")
        if len(parts) > 0:
            base = parts[0]
            candidate = os.path.join(base, "etc", "profile.d", "conda.sh")
            if os.path.isfile(candidate):
                conda_sh = candidate
    
    if not conda_sh:
         # Try standard location
         conda_sh = "/home/a/miniconda3/etc/profile.d/conda.sh"
    
    if not os.path.isfile(conda_sh):
        _logger.warning("[_get_conda_activate_cmd] 未找到 conda.sh，跳过环境激活。MERGENETIC_PYTHON=%s", MERGENETIC_PYTHON)
        return cmd_list 
        
    # Construct shell command
    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd_list)
    # Add debug info to verify environment
    full_cmd = f"source {shlex.quote(conda_sh)} && conda activate mergenetic && echo '[DEBUG] Python in use:' $(which python) && {cmd_str}"
    _logger.info("[_get_conda_activate_cmd] 构造的激活命令: %s", full_cmd)
    
    return ["bash", "-c", full_cmd]


def _get_merge_log_file():
    """当前按天的合并日志文件路径"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    date_suffix = time.strftime("%Y%m%d")
    return os.path.join(LOGS_DIR, f"merge_{date_suffix}.log")


def _setup_logger():
    """配置 merge_manager 使用的 logger，写入当日日志文件"""
    log_file = _get_merge_log_file()
    logger = logging.getLogger("merge_manager")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.info("日志文件位置: %s", os.path.abspath(log_file))
    return logger


def _load_prompt_cfg(prompt_yaml: str) -> dict:
    """从 YAML 文件加载提示模板配置"""
    try:
        p = Path(prompt_yaml)
        if not p.is_absolute():
            # 尝试相对于 testset_repo/yaml 或项目根目录
            if (Path(Config.PROJECT_ROOT) / "testset_repo" / "yaml" / p).exists():
                p = Path(Config.PROJECT_ROOT) / "testset_repo" / "yaml" / p
            else:
                p = Path(Config.PROJECT_ROOT) / p
        
        if not p.exists():
            _logger.warning("[_load_prompt_cfg] YAML 文件不存在: %s", p)
            return {}
            
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        _logger.error("[_load_prompt_cfg] 加载 YAML 失败: %s", e)
        return {}


# 模块加载时设置 logger，供 run_merge_task 使用
_logger = _setup_logger()


def _model_has_mergenetic_expected_keys(model_path: str) -> bool:
    """检查模型是否包含 mergenetic/mergekit 期望的键（如 model.norm.weight, model.layers.*）"""
    try:
        from mergekit.io.lazy_tensor_loader import ShardedTensorIndex
    except Exception:
        try:
            from mergekit.io import ShardedTensorIndex
        except Exception:
            _logger.warning("无法导入 ShardedTensorIndex，跳过键检查")
            return True

    try:
        index = ShardedTensorIndex.from_disk(model_path)
        keys = list(index.tensor_paths.keys())
        has_norm = any("norm.weight" in k for k in keys)
        has_layers = any("model.layers." in k for k in keys)
        _logger.info(
            "[_model_has_mergenetic_expected_keys] 通过 ShardedTensorIndex 检查: model.norm.weight=%s, model.layers.*=%s (总键数: %s)",
            has_norm,
            has_layers,
            len(keys),
        )
        return has_norm and has_layers
    except Exception as e:
        _logger.warning("_model_has_mergenetic_expected_keys 检查异常: %s", e)
        return False


def _fix_model_architecture_if_needed(model_path: str) -> None:
    """如需可在此对 config.json 做兼容性修正（当前仅占位，不修改 Llama 等）"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return
    # 保持原样，不擅自改 model_type，避免误伤 Llama
    return


def normalize_model_weights(model_path: str, force: bool = False) -> None:
    """检查模型权重键格式是否符合 mergenetic 预期；不修改键名，仅做检查与日志"""
    _logger.info("[normalize_model_weights] 开始处理模型: %s (force=%s)", model_path, force)
    model_path = os.path.abspath(model_path)
    if not os.path.isdir(model_path):
        _logger.warning("[normalize_model_weights] 路径不是目录: %s", model_path)
        return

    st_files = [
        os.path.join(model_path, f)
        for f in os.listdir(model_path)
        if f.endswith(".safetensors") and not f.endswith(".index.json")
    ]
    if not st_files:
        _logger.warning("[normalize_model_weights] 未找到 safetensors 文件")
        return

    _logger.info("[normalize_model_weights] 找到 %s 个 safetensors 文件", len(st_files))
    try:
        from mergekit.io.lazy_tensor_loader import ShardedTensorIndex
    except Exception:
        from mergekit.io import ShardedTensorIndex

    index = ShardedTensorIndex.from_disk(model_path)
    keys = list(index.tensor_paths.keys())
    has_norm = any("norm.weight" in k for k in keys)
    has_layers = any("model.layers." in k for k in keys)
    _logger.info(
        "[normalize_model_weights] 检查权重键格式，总键数: %s",
        len(keys),
    )
    _logger.info(
        "[normalize_model_weights] 验证结果 - model.norm.weight: %s, model.layers.*: %s",
        has_norm,
        has_layers,
    )


def run_merge_task(task_id, params, update_progress_callback, task_control=None):
    """
    执行一次融合任务：权重键检查 -> 生成 YAML -> mergenetic 融合 -> 移动输出 -> 写 metadata / 注册模型仓库。
    失败时写入 metadata.json status="error" 及错误信息。
    """
    if task_control is None:
        task_control = {}

    task_dir = os.path.join(MERGE_DIR, task_id)
    output_dir = os.path.join(task_dir, "output")
    yaml_config_dir = os.path.join(task_dir, "yaml_configs")
    config_yaml_path = os.path.join(yaml_config_dir, "config.yaml")
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(yaml_config_dir, exist_ok=True)

    models = params.get("models", [])
    weights = params.get("weights", [])
    method = params.get("method", "ties_dare")
    dtype = params.get("dtype", "float16")
    custom_name = params.get("custom_name", "Untitled Model")
    limit = params.get("limit", "0.5")
    created_at = params.get("created_at", time.time())

    metadata = {
        "id": task_id,
        "type": "merge",
        "custom_name": custom_name,
        "created_at": created_at,
        "models": models,
        "weights": weights,
        "method": method,
        "dtype": dtype,
        "dataset": params.get("dataset", "hellaswag"),
        "hf_dataset": params.get("hf_dataset"),
        "hf_subset": params.get("hf_subset"),
        "hf_split": params.get("hf_split"),
        "status": "pending",
    }
    meta_path = os.path.join(task_dir, "metadata.json")
    _write_metadata(task_id, task_dir, metadata)

    _logger.info("[run_merge_task] ========== 任务开始 ==========")
    task_start_time = time.time()
    _logger.info("[run_merge_task] 任务ID: %s", task_id)
    _logger.info("[run_merge_task] 日志文件: %s", _get_merge_log_file())
    _logger.info("[run_merge_task] 参数: %s", json.dumps(params, ensure_ascii=False, indent=2))

    def _write_error_status(err_msg: str):
        metadata["status"] = "error"
        metadata["error"] = err_msg
        metadata["duration_seconds"] = time.time() - task_start_time
        _write_metadata(task_id, task_dir, metadata)

    try:
        # Python 环境校验（可选）
        import subprocess
        r = subprocess.run(
            [MERGENETIC_PYTHON, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            raise RuntimeError("Python 环境验证失败: %s" % (r.stderr or r.stdout or "unknown"))
        _logger.info("[run_merge_task] Python 环境验证通过: %s", r.stdout.strip() if r.stdout else MERGENETIC_PYTHON)

        update_progress_callback(5, "开始初始化 Mergenetic 融合器...")

        # 解析模型绝对路径：优先使用传入的 model_paths，否则由 MODEL_POOL_PATH + models 解析
        model_paths = params.get("model_paths")
        if model_paths:
            model_paths = [os.path.abspath(p) for p in model_paths]
        else:
            model_paths = [os.path.abspath(os.path.join(MODEL_POOL_PATH, name)) for name in models]
        for p in model_paths:
            if not os.path.isdir(p):
                raise FileNotFoundError("模型路径不存在: %s" % p)

        # 修正模型架构（占位，不修改 Llama 等）
        _logger.info("[run_merge_task] 开始修正模型架构，共 %s 个模型", len(model_paths))
        for i, mp in enumerate(model_paths):
            _logger.info("[run_merge_task] [%s/%s] 修正架构: %s", i + 1, len(model_paths), mp)
            _fix_model_architecture_if_needed(mp)

        # 统一权重键格式（检查）
        _logger.info("[run_merge_task] ========== 开始统一权重键格式，共 %s 个模型 ==========", len(model_paths))
        for i, mp in enumerate(model_paths):
            _logger.info("[run_merge_task] [%s/%s] 处理模型: %s", i + 1, len(model_paths), mp)
            _logger.info("[run_merge_task] 融合前检查模型键...")
            has_keys = _model_has_mergenetic_expected_keys(mp)
            _logger.info("[run_merge_task] 融合前检查结果: has_keys=%s", has_keys)
            if not has_keys:
                raise ValueError("模型键不符合 mergenetic 预期: %s" % mp)
            _logger.info("[run_merge_task] 模型已有所需键，执行轻量级检查...")
            normalize_model_weights(mp, force=False)
            _logger.info("[run_merge_task] 融合后验证模型键（使用 mergekit 索引方式）...")
            has_keys2 = _model_has_mergenetic_expected_keys(mp)
            _logger.info("[run_merge_task] 融合后验证结果: has_keys=%s", has_keys2)
            if not has_keys2:
                raise ValueError("融合后验证失败: %s" % mp)
            _logger.info("[run_merge_task] ✅ 验证通过 - %s", mp)
        _logger.info("[run_merge_task] ========== 权重键格式统一完成 ==========")

        # 生成 YAML 并执行 mergenetic 融合
        update_progress_callback(15, "生成融合配置...")
        num_models = len(model_paths)
        weights_floats = [float(weights[i]) if i < len(weights) else 1.0 for i in range(num_models)]
        if (method or "").strip().lower() == "linear":
            from mergenetic.merging.linear_merger import LinearMerger
            merger = LinearMerger(
                run_id=task_id,
                path_to_base_model=model_paths[0],
                model_paths=model_paths,
                path_to_store_yaml=yaml_config_dir,
                path_to_store_merged_model=output_dir,
                dtype=dtype,
            )
            merger.create_individual_configuration(weights_floats)
        else:
            from mergenetic.merging.ties_dare_merger import TiesDareMerger
            base_path = model_paths[0]
            other_paths = model_paths[1:]
            density = float(limit) if limit else 0.5
            densities = [density] * num_models
            weights_and_densities = list(weights_floats) + list(densities)
            merger = TiesDareMerger(
                run_id=task_id,
                path_to_base_model=base_path,
                model_paths=other_paths,
                path_to_store_yaml=yaml_config_dir,
                path_to_store_merged_model=output_dir,
                dtype=dtype,
            )
            merger.create_individual_configuration(weights_and_densities)

        _logger.info("[run_merge_task] ========== 开始执行 Mergenetic 融合 ==========")
        _logger.info("[run_merge_task] 配置路径: %s", config_yaml_path)
        _logger.info("[run_merge_task] 融合方法: %s, 模型数量: %s", method, num_models)
        _logger.info("[run_merge_task] 参与融合的模型路径:")
        for i, p in enumerate(model_paths):
            _logger.info("[run_merge_task]   [%s] %s", i + 1, p)

        # 融合前索引验证
        try:
            from mergekit.io.lazy_tensor_loader import ShardedTensorIndex
        except Exception:
            from mergekit.io import ShardedTensorIndex
        for i, mp in enumerate(model_paths):
            has_k = _model_has_mergenetic_expected_keys(mp)
            idx = ShardedTensorIndex.from_disk(mp) if has_k else None
            sample = list(idx.tensor_paths.keys())[:5] if idx else []
            _logger.info(
                "[run_merge_task] 融合前索引验证 [%s] %s -> model.norm.weight=%s, 键示例=%s",
                i + 1,
                mp,
                has_k,
                sample,
            )

        # 清空 mergekit LoaderCache
        try:
            from mergekit.io.tasks import LoaderCache
            LoaderCache().loaders.clear()
            _logger.info("[run_merge_task] 已清空 mergekit LoaderCache，确保从磁盘重新加载模型索引")
        except Exception as e:
            _logger.warning("清空 LoaderCache 失败（可忽略）: %s", e)

        update_progress_callback(25, "正在执行 Mergenetic 融合...")
        out_path = merger.merge_model_from_configuration(Path(config_yaml_path))

        _logger.info("[run_merge_task] ✅ 融合成功，输出路径: %s", out_path)

        # 解析绝对路径
        out_str = str(out_path)
        if not os.path.isabs(out_str):
            project_root = Config.PROJECT_ROOT
            out_str = os.path.normpath(os.path.join(project_root, out_str))
        _logger.info(
            "[run_merge_task] mergenetic返回路径: %s -> 解析为: %s",
            str(out_path),
            out_str,
        )
        expected_dir = os.path.join(task_dir, "output")
        _logger.info("[run_merge_task] 预期路径: %s", os.path.join(expected_dir, task_id))
        _logger.info("[run_merge_task] 返回路径存在: %s", os.path.isdir(out_str))

        # 若输出在 output/task_id 下，将内容移到 output（注意：切勿删除 output_dir，否则 inner 会一并被删）
        inner = os.path.join(output_dir, task_id)
        if os.path.isdir(inner):
            _logger.info("[run_merge_task] 融合输出路径存在: %s", inner)
            _logger.info("[run_merge_task] 模型在预期位置，准备移动到 output_dir: %s", output_dir)
            _logger.info("[run_merge_task] 移动模型内容从 %s 到 %s", inner, output_dir)
            inner_abs = os.path.abspath(inner)
            output_dir_abs = os.path.abspath(output_dir)
            if not os.path.isdir(inner_abs):
                raise RuntimeError("融合输出目录在复制前已不存在: %s" % inner_abs)
            for name in os.listdir(inner_abs):
                src = os.path.join(inner_abs, name)
                dst = os.path.join(output_dir_abs, name)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
            shutil.rmtree(inner_abs)
            _logger.info("[run_merge_task] ✅ 移动成功")
        elif os.path.isdir(out_str) and out_str != output_dir:
            # 若 out_path 是别的目录，整体拷贝到 output
            for name in os.listdir(out_str):
                src = os.path.join(out_str, name)
                dst = os.path.join(output_dir, name)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    dest = os.path.join(output_dir, name)
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.copytree(src, dest)
            _logger.info("[run_merge_task] ✅ 已拷贝到 output")
        else:
            _logger.info("[run_merge_task] ✅ 最终路径验证通过: %s", output_dir)

        # 最终验证
        if not os.path.isdir(output_dir) or not any(
            f.endswith(".safetensors") or f == "config.json"
            for f in os.listdir(output_dir)
        ):
            raise RuntimeError("融合输出目录无效或缺少权重/配置: %s" % output_dir)
        _logger.info("[run_merge_task] ✅ 最终路径验证通过: %s", output_dir)

        update_progress_callback(90, "融合完成，正在注册模型仓库...")
        # 注册到模型仓库
        try:
            from model_repo import api as model_repo_api
        except ImportError:
            model_repo_api = None
        if model_repo_api and hasattr(model_repo_api, "register_merged_model"):
            display_name = custom_name or ("_".join(models[:2]) + "_" + task_id[:8])
            recipe = {
                "parent_models": models,
                "weights": weights_floats,
                "method": method,
                "library": "mergenetic",
                "dtype": dtype,
            }
            model_repo_api.register_merged_model(output_dir, display_name, recipe)
            _logger.info("[run_merge_task] ✅ 融合模型已注册到模型仓库: %s", display_name)
        else:
            _logger.info("[run_merge_task] 未注册模型仓库（无 model_repo.api.register_merged_model）")

        metadata["status"] = "success"
        metadata["duration_seconds"] = time.time() - task_start_time
        metadata["model_path"] = output_dir
        metadata["metrics"] = {
            "output_path": output_dir,
            "accuracy": None,
            "f1_score": None,
            "test_cases": 0,
            "passed": 0,
            "base_name": models[0] if models else "",
            "comparison": {"labels": [], "base_data": [], "merged_data": []},
        }
        _write_metadata(task_id, task_dir, metadata)

        update_progress_callback(100, "任务完成")
        return {"status": "success", "output_path": output_dir, "metrics": metadata.get("metrics", {})}

    except Exception as e:
        _logger.exception("[run_merge_task] 任务失败: %s", e)
        err_str = str(e)
        inner = os.path.join(output_dir, task_id)
        salvaged = False
        if "Circular reference" in err_str and os.path.isdir(inner):
            try:
                inner_abs = os.path.abspath(inner)
                output_dir_abs = os.path.abspath(output_dir)
                for name in os.listdir(inner_abs):
                    src = os.path.join(inner_abs, name)
                    dst = os.path.join(output_dir_abs, name)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    else:
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                shutil.rmtree(inner_abs)
                _logger.info("[run_merge_task] 已从 %s 抢救输出到 %s", inner, output_dir)
                if os.path.isdir(output_dir) and any(
                    f.endswith(".safetensors") or f == "config.json"
                    for f in os.listdir(output_dir)
                ):
                    salvaged = True
                    metadata["status"] = "success"
                    metadata["duration_seconds"] = time.time() - task_start_time
                    metadata["model_path"] = output_dir
                    metadata["metrics"] = {"output_path": output_dir}
                    _write_metadata(task_id, task_dir, metadata)
                    update_progress_callback(100, "任务完成（已抢救输出）")
                    return {"status": "success", "output_path": output_dir, "metrics": metadata.get("metrics", {})}
            except Exception as salvage_err:
                _logger.warning("[run_merge_task] 抢救输出失败: %s", salvage_err)
        _write_error_status(err_str)
        if task_control.get("aborted"):
            return {"status": "stopped", "message": "用户已停止"}
        return {"status": "error", "error": err_str}


def materialize_recipe_to_temp(recipe_id, parent_task_id, suffix, update_progress_callback, task_control=None):
    """
    将配方物化到临时目录，不注册到 model_repo。用于三代融合等中间步骤。
    返回 (output_path, error_msg)。成功时 error_msg 为 None。
    """
    synthetic_task_id = "%s_gen2_%s" % (parent_task_id, suffix)
    params = {"recipe_id": recipe_id}
    result = run_recipe_apply_task(
        synthetic_task_id,
        params,
        update_progress_callback,
        task_control=task_control,
        skip_register=True,
    )
    if result.get("status") == "success":
        return (result.get("output_path"), None)
    return (None, result.get("error", "未知错误"))


def cleanup_recipe_temp_dirs(parent_task_id, suffixes):
    """清理完全融合中物化配方产生的临时目录（merges/<parent_task_id>_gen2_<suffix>）。"""
    for suffix in suffixes:
        synthetic_id = "%s_gen2_%s" % (parent_task_id, suffix)
        temp_dir = os.path.join(MERGE_DIR, synthetic_id)
        if os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                _logger.info("[cleanup] 已删除临时物化目录: %s", temp_dir)
            except Exception as e:
                _logger.warning("[cleanup] 删除临时目录失败 %s: %s", temp_dir, e)


def run_recipe_apply_task(task_id, params, update_progress_callback, task_control=None, skip_register=False):
    """
    按配方执行一次合并（固定 genotype，不进化）。用于「根据配方直接融合出最终模型」或中间物化。
    params: recipe_id, custom_name（可选）
    skip_register: 为 True 时不注册到 model_repo，用于中间物化（如三代融合前的 A/B）。
    """
    if task_control is None:
        task_control = {}

    recipe_id = params.get("recipe_id")
    if not recipe_id:
        return {"status": "error", "error": "缺少 recipe_id"}

    recipe_path = os.path.join(RECIPES_DIR, "%s.json" % recipe_id)
    if not os.path.isfile(recipe_path):
        return {"status": "error", "error": "配方不存在: %s" % recipe_id}

    with open(recipe_path, "r", encoding="utf-8") as f:
        recipe = json.load(f)

    model_paths = recipe.get("model_paths") or []
    if len(model_paths) < 2:
        return {"status": "error", "error": "配方中模型数量不足"}

    model_paths = [os.path.abspath(p) for p in model_paths]
    for p in model_paths:
        if not os.path.isdir(p):
            return {"status": "error", "error": "模型路径不存在: %s" % p}

    weights = recipe.get("best_genotype")
    if not weights or len(weights) < 2:
        return {"status": "error", "error": "配方缺少 best_genotype"}
    weights = [float(weights[i]) for i in range(min(len(weights), len(model_paths)))]
    if len(weights) < len(model_paths):
        weights.extend([1.0] * (len(model_paths) - len(weights)))

    dtype = recipe.get("dtype", "bfloat16")
    custom_name = params.get("custom_name", "").strip() or recipe.get("custom_name", "") or ("配方-%s" % recipe_id)
    density = 0.5

    task_dir = os.path.join(MERGE_DIR, task_id)
    output_dir = os.path.join(task_dir, "output")
    yaml_config_dir = os.path.join(task_dir, "yaml_configs")
    config_yaml_path = os.path.join(yaml_config_dir, "config.yaml")
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(yaml_config_dir, exist_ok=True)

    meta_path = os.path.join(task_dir, "metadata.json")
    metadata = {
        "id": task_id,
        "type": "recipe_apply",
        "recipe_id": recipe_id,
        "custom_name": custom_name,
        "model_paths": model_paths,
        "weights": weights,
        "dtype": dtype,
        "status": "pending",
    }
    _write_metadata(task_id, task_dir, metadata)

    def _write_error_status(err_msg):
        metadata["status"] = "error"
        metadata["error"] = err_msg
        _write_metadata(task_id, task_dir, metadata)

    try:
        import subprocess
        r = subprocess.run(
            [MERGENETIC_PYTHON, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            raise RuntimeError("Python 环境验证失败: %s" % (r.stderr or r.stdout or "unknown"))

        update_progress_callback(10, "按配方执行 Mergenetic 融合...")
        for i, mp in enumerate(model_paths):
            if not _model_has_mergenetic_expected_keys(mp):
                raise ValueError("模型键不符合 mergenetic 预期: %s" % mp)
            normalize_model_weights(mp, force=False)

        update_progress_callback(20, "生成融合配置...")
        from mergenetic.merging.ties_dare_merger import TiesDareMerger

        base_path = model_paths[0]
        other_paths = model_paths[1:]
        densities = [density] * len(model_paths)
        weights_and_densities = list(weights) + list(densities)

        merger = TiesDareMerger(
            run_id=task_id,
            path_to_base_model=base_path,
            model_paths=other_paths,
            path_to_store_yaml=yaml_config_dir,
            path_to_store_merged_model=output_dir,
            dtype=dtype,
        )
        merger.create_individual_configuration(weights_and_densities)

        update_progress_callback(40, "正在执行合并...")
        out_path = merger.merge_model_from_configuration(Path(config_yaml_path))

        # 若输出在 output/task_id 下，移到 output
        inner = os.path.join(output_dir, task_id)
        
        # 确定源目录：优先尝试 inner，其次尝试 out_path，最后检查是否直接在 output_dir
        source_dir = None
        if os.path.isdir(inner):
            source_dir = inner
        elif out_path and str(out_path) != output_dir and os.path.isdir(str(out_path)):
            source_dir = str(out_path)
            
        if source_dir:
            _logger.info("[run_merge_task] 准备移动模型文件从 %s 到 %s", source_dir, output_dir)
            for name in os.listdir(source_dir):
                src = os.path.join(source_dir, name)
                dst = os.path.join(output_dir, name)
                
                # 忽略自身目录（如果 source_dir 是 output_dir 的子目录，虽然 logic 上 inner 是子目录）
                if os.path.abspath(src) == os.path.abspath(output_dir):
                    continue

                try:
                    if os.path.islink(src) or os.path.isfile(src):
                        if os.path.exists(dst):
                            if os.path.isdir(dst):
                                shutil.rmtree(dst)
                            else:
                                os.remove(dst)
                        shutil.copy2(src, dst, follow_symlinks=True)
                    elif os.path.isdir(src):
                        if os.path.exists(dst):
                            if os.path.isdir(dst):
                                shutil.rmtree(dst)
                            else:
                                os.remove(dst)
                        shutil.copytree(src, dst)
                except Exception as move_err:
                    _logger.warning("[run_merge_task] 移动文件失败 %s -> %s: %s", src, dst, move_err)
            
            # 尝试删除源目录
            try:
                shutil.rmtree(source_dir)
            except Exception as rm_err:
                _logger.warning("[run_merge_task] 删除源目录失败 %s: %s", source_dir, rm_err)
                
            _logger.info("[run_merge_task] ✅ 移动/清理完成")
        else:
            _logger.info("[run_merge_task] ✅ 未发现子目录结构，假设文件已在 output_dir")

        if not os.path.isdir(output_dir) or not any(
            f.endswith(".safetensors") or f == "config.json"
            for f in os.listdir(output_dir)
        ):
            raise RuntimeError("融合输出目录无效: %s" % output_dir)

        if not skip_register:
            update_progress_callback(90, "注册模型仓库...")
            try:
                from model_repo import api as model_repo_api
            except ImportError:
                model_repo_api = None
            if model_repo_api and hasattr(model_repo_api, "register_merged_model"):
                recipe_meta = {
                    "parent_models": [os.path.basename(p) for p in model_paths],
                    "weights": weights,
                    "method": "ties_dare",
                    "library": "mergenetic",
                    "dtype": dtype,
                }
                model_repo_api.register_merged_model(output_dir, custom_name, recipe_meta)
        metadata["status"] = "success"
        metadata["model_path"] = output_dir
        metadata["metrics"] = {"output_path": output_dir}
        _write_metadata(task_id, task_dir, metadata)
        update_progress_callback(100, "配方融合完成")
        return {"status": "success", "output_path": output_dir}
    except Exception as e:
        _logger.exception("[run_recipe_apply_task] 失败: %s", e)
        _write_error_status(str(e))
        return {"status": "error", "error": str(e)}


def _eval_get_context_length(model_path):
    """读取模型上下文长度（与 mergeKit_alpha 一致）。"""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        for attr in ("max_position_embeddings", "seq_length", "n_positions", "model_max_length"):
            if hasattr(config, attr):
                return getattr(config, attr)
        return 2048
    except Exception:
        return 0


def _pick_latest_json(path):
    """在路径或目录下取最新修改的 json 文件（与 mergeKit_alpha 一致）。"""
    if not path:
        return None
    candidates = []
    if os.path.isfile(path) and path.lower().endswith(".json"):
        candidates.append(path)
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fn in files:
                if fn.lower().endswith(".json"):
                    if fn in ("metadata.json", "progress.json"):
                        continue
                    candidates.append(os.path.join(root, fn))
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


# 测试集仓库：HF 数据集 -> lm_eval 任务名前缀。子集通过按科目任务名指定（如 mmlu_college_medicine）。
# 结果写入 output_path（merges/<task_id>/），_pick_latest_json 会递归查找目录内最新 results*.json，其他测试集路径与输出逻辑一致。
# 未在此映射的仓库数据集会以 hf_dataset 名作为 --tasks 传入，需为 lm_eval 支持的任务名才能正常运行。
HF_DATASET_TO_LM_EVAL_TASK = {
    "cais/mmlu": "mmlu",
    "mmlu": "mmlu",
    "m-a-p/cmmmu": "cmmmu",
    "cmmmu/cmmmu": "cmmmu",
}

def _load_testsets_dict():
    path = getattr(Config, "TESTSET_DATA_PATH", "") or os.path.join(Config.PROJECT_ROOT, "testset_repo", "data", "testsets.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("testsets") if isinstance(data, dict) else {}
    except Exception:
        return {}

def _resolve_prompt_yaml_for_testset(testset_id, hf_dataset, hf_subset):
    if not testset_id:
        return ""
    testsets = _load_testsets_dict()
    if not isinstance(testsets, dict):
        return ""
    for k, v in testsets.items():
        tid = (v.get("testset_id") or k or "").strip()
        if tid == testset_id:
            p = (v.get("yaml_template_path") or "").strip()
            return p
    return ""

def _normalize_answer_text(s):
    if s is None:
        return ""
    t = str(s)
    t = t.replace(",", " ").replace("\n", " ").replace("\t", " ")
    t = t.strip()
    t = t.rstrip(".").rstrip("。")
    return t

def _extract_gsm8k_predicted_answer(text):
    if not text:
        return ""
    s = str(text)
    m = re.search(r"final\\s+answer\\s+is\\s*:\\s*(?:\\\\boxed\\{)?([^\\}\\n]+)\\}?", s, flags=re.IGNORECASE)
    if m:
        return _normalize_answer_text(m.group(1))
    m2 = re.search(r"\\boxed\\{([^\\}]+)\\}", s)
    if m2:
        return _normalize_answer_text(m2.group(1))
    # 如果没有 boxed，尝试提取文本中出现的最后一个数字
    nums = re.findall(r"(-?\d+(?:\.\d+)?)", s)
    if nums:
        return _normalize_answer_text(nums[-1])
    return _normalize_answer_text(s.splitlines()[-1] if s.splitlines() else s)

def run_yaml_eval_stream(
    model_path,
    output_path,
    callback,
    start_prog,
    end_prog,
    task_control=None,
    hf_dataset=None,
    hf_subset=None,
    hf_split=None,
    prompt_yaml_path=None,
    limit="0.5",
):
    import math
    _logger.info("[YAML] run_yaml_eval_stream 启动. Python executable: %s", sys.executable)
    if task_control is None:
        task_control = {}
    os.makedirs(output_path, exist_ok=True)
    try:
        import yaml as _yaml
    except Exception:
        raise RuntimeError("缺少 yaml 解析依赖")
    try:
        from datasets import load_dataset as _load_dataset
    except Exception as e:
        raise RuntimeError("缺少 datasets 依赖: %s" % e)
    try:
        import torch as _torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError("缺少 transformers 依赖: %s" % e)
    cache_dir = getattr(Config, "HF_DATASETS_CACHE", None) or os.environ.get("HF_DATASETS_CACHE")
    ds = None
    if hf_dataset:
        splits_to_try = [hf_split] if hf_split else ["test", "validation", "train"]
        error_msgs = []
        for sp in splits_to_try:
            if not sp: continue
            try:
                if hf_subset:
                    ds = _load_dataset(hf_dataset, hf_subset, split=sp, trust_remote_code=True, cache_dir=cache_dir)
                else:
                    ds = _load_dataset(hf_dataset, split=sp, trust_remote_code=True, cache_dir=cache_dir)
                if ds:
                    _logger.info("[YAML] 成功加载数据集 %s (subset=%s, split=%s)", hf_dataset, hf_subset, sp)
                    break
            except Exception as e:
                error_msgs.append(f"{sp}: {e}")
        
        if ds is None:
            raise RuntimeError("加载数据集失败: %s. 尝试splits: %s. Errors: %s" % (hf_dataset, splits_to_try, "; ".join(error_msgs)))
    n = 0
    if ds is None:
        raise RuntimeError("数据集无效")
    try:
        n = len(ds)
    except Exception:
        try:
            keys_list = list(ds.keys())
            if keys_list:
                n = len(ds[keys_list[0]]) if hasattr(ds[keys_list[0]], "__len__") else 0
                ds = ds[keys_list[0]]
        except Exception:
            pass
    if n <= 0:
        raise RuntimeError("数据集样本为空")
    
    # Ensure dataset is indexable or convert to list
    if not hasattr(ds, "__getitem__") or hasattr(ds, "next"): # Check if iterable/streaming
         try:
             _logger.info("[YAML] Converting iterable dataset to list for random access...")
             ds = list(ds)
         except Exception as e:
             _logger.warning("[YAML] Failed to convert dataset to list: %s", e)

    limit_val = 1.0
    try:
        if isinstance(limit, str) and "." in limit:
            limit_val = max(0.01, min(1.0, float(limit)))
        else:
            ival = int(limit)
            limit_val = min(1.0, max(0.01, ival / float(max(1, n))))
    except Exception:
        limit_val = 1.0
    target_n = max(1, int(math.ceil(n * limit_val)))
    try:
        with open(prompt_yaml_path, "r", encoding="utf-8") as f:
            prompt_cfg = _yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError("读取 YAML 失败: %s" % e)
    tasks_cfg = prompt_cfg.get("tasks") or []
    field_spec = {}
    template_text = ""
    if tasks_cfg:
        t0 = tasks_cfg[0]
        field_spec = t0.get("field_spec") or {}
        solvers = t0.get("solvers") or []
        for s in solvers:
            if (s.get("name") or "").strip() == "prompt_template":
                args = s.get("args") or {}
                template_text = (args.get("template") or "").strip()
                break
    input_field = (field_spec.get("input") or "question").strip()
    target_field = (field_spec.get("target") or "answer").strip()
    if not template_text:
        template_text = "{prompt}"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        # 针对 Qwen 系列模型，避免使用 float16，改用 bfloat16 或 float32 以防止输出乱码
        base_name = os.path.basename(str(model_path)).lower()
        
        dtype = _torch.float16 if device == "cuda" else _torch.float32
        if "qwen" in base_name and device == "cuda":
            # 优先尝试 bfloat16
            try:
                if _torch.cuda.is_bf16_supported():
                    dtype = _torch.bfloat16
                else:
                    dtype = _torch.float32
            except Exception:
                # 如果 pytorch 版本过低不支持 is_bf16_supported，安全起见使用 float32
                dtype = _torch.float32
                
        _logger.info("[YAML] Loading model %s with dtype=%s, device=%s", base_name, dtype, device)
            
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device, torch_dtype=dtype)
        model.eval()
    except Exception as e:
        raise RuntimeError("加载模型失败: %s" % e)
    start_time = time.time()
    correct = 0
    total = 0
    per_task_acc = {}
    import math  # Fix for 'name math is not defined'
    progress_path = os.path.join(output_path, "progress.json")
    samples_log_path = os.path.join(output_path, "samples_yaml.jsonl")
    
    _logger.info("[YAML] 开始评估，目标样本数: %d, 模型: %s", target_n, model_path)
    
    # 准备写入 samples 日志
    f_samples = open(samples_log_path, "w", encoding="utf-8")
    
    try:
        # 检查字段是否存在
        if target_n > 0:
            first_ex = None
            try:
                first_ex = ds[0]
            except Exception:
                pass
            if first_ex:
                if input_field not in first_ex:
                    _logger.warning("[YAML] 警告: 数据集首条样本缺少 input_field '%s'. 现有字段: %s", input_field, list(first_ex.keys()))
                if target_field not in first_ex:
                    _logger.warning("[YAML] 警告: 数据集首条样本缺少 target_field '%s'. 现有字段: %s", target_field, list(first_ex.keys()))

        for i in range(target_n):
            if task_control.get("aborted"):
                raise RuntimeError("任务已被用户手动终止")
            try:
                ex = ds[i]
            except Exception as e:
                _logger.warning("[YAML] 获取样本 %d 失败: %s", i, e)
                continue
                
            q = ex.get(input_field)
            g = ex.get(target_field)
            inp_text = template_text.replace("{prompt}", str(q))
            try:
                ids = tokenizer(inp_text, return_tensors="pt").to(device)
                with _torch.no_grad():
                    # 增加 repetition_penalty 和 pad_token_id 以防止死循环输出
                    out = model.generate(
                        **ids, 
                        max_new_tokens=256, 
                        do_sample=False, 
                        pad_token_id=tokenizer.pad_token_id,
                        repetition_penalty=1.1
                    )
                pred_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
                # 尝试移除 prompt 部分
                if pred_text.startswith(inp_text):
                    pred_text = pred_text[len(inp_text):]
            except Exception as gen_err:
                _logger.warning("[YAML] 生成失败: %s", gen_err)
                pred_text = ""
            
            # 根据数据集类型选择解析策略
            if hf_dataset and "gsm8k" in hf_dataset.lower():
                pred_norm = _extract_gsm8k_predicted_answer(pred_text)
                # 针对 GSM8K，移除数值中的空格（原 normalize 将逗号转为空格），以便 "1 000" 能匹配 "1000"
                if pred_norm:
                    pred_norm = pred_norm.replace(" ", "")
                
                # 提取 Gold Answer (通常在 #### 之后)
                gold_str = str(g)
                if "####" in gold_str:
                    gold_norm = _normalize_answer_text(gold_str.split("####")[-1])
                else:
                    gold_norm = _normalize_answer_text(g)
                if gold_norm:
                    gold_norm = gold_norm.replace(" ", "")
            else:
                # 通用处理：简单的清理
                pred_norm = _normalize_answer_text(pred_text)
                # 尝试提取第一个非空行作为答案（针对 MMLU 等）
                if not pred_norm and pred_text:
                     lines = [l.strip() for l in pred_text.splitlines() if l.strip()]
                     if lines:
                         pred_norm = _normalize_answer_text(lines[0])
                gold_norm = _normalize_answer_text(g)
            is_correct = False
            # 宽松匹配：如果是 MMLU (单个字母)，做大小写不敏感匹配
            if pred_norm and gold_norm:
                if pred_norm == gold_norm:
                    is_correct = True
                elif len(gold_norm) == 1 and len(pred_norm) == 1 and pred_norm.lower() == gold_norm.lower():
                    is_correct = True
                # 尝试包含匹配 (例如 pred="Answer: A", gold="A")
                elif len(gold_norm) == 1 and gold_norm.lower() in pred_norm.lower().split():
                     is_correct = True
            
            if is_correct:
                correct += 1
            total += 1
            
            # 记录 sample
            try:
                log_entry = {
                    "i": i,
                    "input": str(q),
                    "gold": str(g),
                    "pred_raw": pred_text,
                    "pred_norm": pred_norm,
                    "gold_norm": gold_norm,
                    "correct": is_correct
                }
                f_samples.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                f_samples.flush()
            except Exception:
                pass

            # 更新进度
            if i % max(1, int(target_n / 50)) == 0 or i == target_n - 1:
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1)) * (target_n - (i + 1))
                prog_data = {
                    "current": i + 1,
                    "total": target_n,
                    "percent": round((i + 1) / target_n * 100, 1),
                    "eta_seconds": int(eta),
                    "status": "running"
                }
                try:
                    with open(progress_path, "w") as f_prog:
                        json.dump(prog_data, f_prog)
                except Exception:
                    pass
    except Exception as e:
        _logger.error("[YAML] 评估循环异常: %s", e)
        raise e
    finally:
        f_samples.close()

    if total == 0 and target_n > 0:
        msg = f"评估失败: 未成功处理任何样本 (target={target_n}, total={total}). 请检查数据集字段是否匹配."
        _logger.error("[YAML] %s", msg)
        raise RuntimeError(msg)

    duration = time.time() - start_time
    throughput = total / duration if duration > 0 else 0.0
    acc_val = (correct / max(1, total)) if total > 0 else 0.0
    task_key = "gsm8k" if (hf_dataset and "gsm8k" in hf_dataset.lower()) else (hf_dataset or "task")
    per_task_acc[task_key] = round(acc_val * 100, 2)
    context_len = _eval_get_context_length(model_path)
    try:
        del model
    except Exception:
        pass
    try:
        import torch as _torch2
        if hasattr(_torch2.cuda, "empty_cache"):
            _torch2.cuda.empty_cache()
    except Exception:
        pass
    return {
        "acc": round(acc_val * 100, 2),
        "f1": round(acc_val, 4),
        "samples": int(total),
        "time": round(duration, 2),
        "throughput": round(throughput, 2),
        "context": int(context_len or 0),
        "per_task_acc": per_task_acc
    }


def _load_eval_task_mapping():
    """加载可选的部署用映射文件，格式 JSON：{"hf_dataset": "前缀", "hf_dataset|subset": "lm_eval_task"}，与内置映射合并。"""
    path = os.path.join(Config.PROJECT_ROOT, "config", "eval_task_mapping.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _logger.warning("[eval] 读取映射文件失败 %s: %s", path, e)
        return {}


def _save_eval_task_mapping(mapping_dict):
    """保存映射到配置文件。"""
    path = os.path.join(Config.PROJECT_ROOT, "config", "eval_task_mapping.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping_dict, f, ensure_ascii=False, indent=2)
        _logger.info("[eval] 已保存任务映射到 %s", path)
        return True
    except Exception as e:
        _logger.warning("[eval] 保存映射文件失败 %s: %s", path, e)
        return False


def _update_leaderboard(testset_id, entry):
    """写入 leaderboard.json。DB 侧由 app.services 在评估任务完成后通过 _db_insert_eval_result 双写 EvaluationResult，与本次调用成对。"""
    if not testset_id:
        return False
    lb_path = os.path.join(Config.PROJECT_ROOT, "testset_repo", "data", "leaderboard.json")
    os.makedirs(os.path.dirname(lb_path), exist_ok=True)
    data = {}
    if os.path.isfile(lb_path):
        try:
            with open(lb_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception as e:
            _logger.warning("[eval] 读取 leaderboard 失败: %s", e)
            data = {}
    leaderboards = data.get("leaderboards", {}) if isinstance(data, dict) else {}
    lb = leaderboards.get(testset_id, []) if isinstance(leaderboards, dict) else []
    task_id = entry.get("task_id")
    if task_id:
        lb = [e for e in lb if e.get("task_id") != task_id]
    lb.append(entry)
    if not isinstance(leaderboards, dict):
        leaderboards = {}
    leaderboards[testset_id] = lb
    try:
        with open(lb_path, "w", encoding="utf-8") as f:
            json.dump({"leaderboards": leaderboards}, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        _logger.warning("[eval] 保存 leaderboard 失败: %s", e)
        return False


def _get_available_lm_eval_tasks():
    """获取所有可用的 lm_eval 任务列表（排除任务组，只返回具体任务）。"""
    try:
        from lm_eval.tasks import TaskManager
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            task_manager = TaskManager()
        # 0.4.11+: all_subtasks 只含具体任务；旧版回退 all_tasks - all_groups
        if hasattr(task_manager, 'all_subtasks'):
            return sorted(task_manager.all_subtasks)
        available_tasks = task_manager.all_tasks
        task_groups_set = set()
        try:
            if hasattr(task_manager, 'all_groups'):
                task_groups = task_manager.all_groups
                if isinstance(task_groups, list):
                    task_groups_set = set(task_groups)
                elif hasattr(task_groups, 'keys'):
                    task_groups_set = set(task_groups.keys())
        except Exception:
            pass
        if task_groups_set:
            filtered_tasks = [t for t in available_tasks if t not in task_groups_set]
            return sorted(filtered_tasks) if filtered_tasks else sorted(list(available_tasks))
        return sorted(list(available_tasks)) if available_tasks else []
    except Exception as e:
        _logger.warning("[eval] 获取 lm_eval 任务列表失败: %s", e)
        # 尝试使用命令行方式
        try:
            import subprocess
            result = subprocess.run(
                ["lm_eval", "--tasks", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                tasks = [line.strip() for line in result.stdout.split("\n") if line.strip() and not line.startswith("#")]
                return tasks
        except Exception as cmd_err:
            _logger.debug("[eval] 命令行获取任务列表也失败: %s", cmd_err)
        return []


def _auto_discover_lm_eval_task(hf_dataset, hf_subset=None):
    """
    自动发现与 hf_dataset/hf_subset 匹配的 lm_eval 任务。
    返回匹配的任务名，如果找不到返回 None。
    确保返回的是具体任务而不是任务组。
    """
    if not hf_dataset:
        return None
    
    hf_dataset_lower = hf_dataset.lower().strip()
    subset_lower = (hf_subset or "").lower().strip()
    
    # 提取数据集名称（去掉组织名）
    dataset_name_parts = hf_dataset_lower.split("/")
    dataset_base_name = dataset_name_parts[-1] if len(dataset_name_parts) > 1 else dataset_name_parts[0]
    dataset_base_name = dataset_base_name.replace("-", "_")
    
    available_tasks = _get_available_lm_eval_tasks()
    if not available_tasks:
        _logger.warning("[eval] 无法获取 lm_eval 任务列表，跳过自动发现")
        return None
    
    # 获取任务组集合，用于验证返回的不是任务组
    try:
        from lm_eval.tasks import TaskManager
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            task_manager = TaskManager()
        task_groups_set = set(task_manager.all_groups) if hasattr(task_manager, 'all_groups') else set()
    except Exception:
        task_groups_set = set()
    
    # 匹配策略（按优先级）：
    # 1. 精确匹配：hf_dataset 或 dataset_base_name
    # 2. 包含匹配：任务名包含 dataset_base_name
    # 3. 如果有 subset，尝试匹配包含 subset 的任务
    # 4. 对于 MMLU 相关，尝试匹配 mmlu_* 模式
    
    candidates = []
    
    # 策略1: 精确匹配
    for task in available_tasks:
        # 确保不是任务组
        if task in task_groups_set:
            continue
        task_lower = task.lower()
        if task_lower == hf_dataset_lower or task_lower == dataset_base_name:
            _logger.info("[eval] 自动发现精确匹配: %s -> %s", hf_dataset, task)
            return task
    
    # 策略2: 包含匹配（优先匹配更长的任务名）
    for task in available_tasks:
        # 确保不是任务组
        if task in task_groups_set:
            continue
        task_lower = task.lower()
        if dataset_base_name in task_lower or task_lower in dataset_base_name:
            # 计算相似度（优先更长的匹配）
            score = len(set(task_lower.split("_")) & set(dataset_base_name.split("_")))
            candidates.append((score, len(task), task))
    
    # 策略3: 如果有 subset，尝试匹配包含 subset 的任务
    if subset_lower:
        subset_normalized = subset_lower.replace("-", "_")
        subset_keywords = [s for s in subset_normalized.split("_") if len(s) > 3]  # 提取关键词（长度>3的词）
        for task in available_tasks:
            # 确保不是任务组
            if task in task_groups_set:
                continue
            task_lower = task.lower()
            # 精确匹配
            if subset_normalized in task_lower:
                score = len(set(task_lower.split("_")) & set([subset_normalized]))
                candidates.append((score + 15, len(task), task))  # subset 精确匹配高分
            # 关键词匹配
            elif subset_keywords and any(kw in task_lower for kw in subset_keywords):
                matched_keywords = sum(1 for kw in subset_keywords if kw in task_lower)
                score = matched_keywords * 5  # 每个匹配的关键词5分
                candidates.append((score + 10, len(task), task))  # subset 关键词匹配加分
    
    # 策略4: MMLU 特殊处理（包括 MMLU-Pro 等变体）
    if "mmlu" in dataset_base_name:
        # 对于 MMLU-Pro，尝试匹配 mmlu_pro 相关的任务
        if "pro" in dataset_base_name or "pro" in hf_dataset_lower:
            # 优先匹配：先找精确匹配 subset 的任务
            if subset_lower:
                subset_normalized = subset_lower.replace("-", "_")
                subset_keywords = [s for s in subset_normalized.split("_") if len(s) > 3]
                
                # 精确匹配 subset（如 college_medicine）
                for task in available_tasks:
                    if task in task_groups_set:
                        continue
                    task_lower = task.lower()
                    if subset_normalized in task_lower and "mmlu" in task_lower:
                        _logger.info("[eval] 自动发现 MMLU-Pro 精确 subset 匹配: %s/%s -> %s", hf_dataset, hf_subset, task)
                        return task
                
                # 智能匹配：对于 health_and_medicine，优先匹配 professional_medicine
                # 对于其他 subset，尝试找到最相关的任务
                best_matches = []
                for task in available_tasks:
                    if task in task_groups_set:
                        continue
                    task_lower = task.lower()
                    if "mmlu" in task_lower:
                        score = 0
                        matched_keywords = sum(1 for kw in subset_keywords if kw in task_lower)
                        score += matched_keywords * 5  # 每个匹配的关键词5分
                        
                        # 特殊处理：health_and_medicine -> professional_medicine
                        if "health" in subset_normalized and "medicine" in subset_normalized:
                            if "professional_medicine" in task_lower or "professional-medicine" in task_lower:
                                score += 20  # 大幅加分
                            elif "college_medicine" in task_lower or "college-medicine" in task_lower:
                                score += 10  # 中等加分
                        
                        # 优先选择不包含特定前缀的任务（如 arabic_leaderboard）
                        # 这些通常是更通用的任务
                        if not any(prefix in task_lower for prefix in ["arabic_leaderboard", "arabic_", "global_mmlu_full"]):
                            score += 5  # 通用任务加分
                        
                        # 特殊处理：college_* -> college_*
                        if "college" in subset_normalized:
                            if "college" in task_lower:
                                score += 15
                        
                        if score > 0:
                            best_matches.append((score, len(task), task))
                
                if best_matches:
                    best_matches.sort(key=lambda x: (-x[0], -x[1]))
                    best_match = best_matches[0][2]
                    _logger.info("[eval] 自动发现 MMLU-Pro subset 智能匹配: %s/%s -> %s (分数: %s)", hf_dataset, hf_subset, best_match, best_matches[0][0])
                    return best_match
            # 如果没有 subset 或匹配失败，尝试匹配 mmlu_pro 相关的任务
            for task in available_tasks:
                if task in task_groups_set:
                    continue
                task_lower = task.lower()
                # 优先匹配包含 mmlu_pro 或 mmlu-pro 的任务
                if "mmlu_pro" in task_lower or "mmlu-pro" in task_lower:
                    _logger.info("[eval] 自动发现 MMLU-Pro 匹配: %s -> %s", hf_dataset, task)
                    return task
        
        # 标准 MMLU 处理
        for task in available_tasks:
            # 确保不是任务组
            if task in task_groups_set:
                continue
            task_lower = task.lower()
            if task_lower.startswith("mmlu") and "pro" not in task_lower:
                if subset_lower:
                    # 尝试匹配 mmlu_subset
                    subset_normalized = subset_lower.replace("-", "_")
                    # 尝试精确匹配或关键词匹配
                    if subset_normalized in task_lower:
                        _logger.info("[eval] 自动发现 MMLU subset 匹配: %s/%s -> %s", hf_dataset, hf_subset, task)
                        return task
                    # 如果 subset 包含多个词，尝试匹配关键词
                    subset_keywords = [s for s in subset_normalized.split("_") if len(s) > 3]
                    if any(kw in task_lower for kw in subset_keywords):
                        candidates.append((20, len(task), task))  # MMLU subset 匹配高分
                else:
                    # 如果没有 subset，跳过通用的 mmlu（因为它是任务组），继续查找具体任务
                    if task_lower == "mmlu":
                        continue
    
    # 选择最佳候选
    if candidates:
        candidates.sort(key=lambda x: (-x[0], -x[1]))  # 按分数降序，长度降序
        best_task = candidates[0][2]
        # 再次验证不是任务组
        if best_task in task_groups_set:
            _logger.warning("[eval] 最佳匹配是任务组，跳过: %s", best_task)
            # 尝试下一个候选
            for score, length, task in candidates[1:]:
                if task not in task_groups_set:
                    best_task = task
                    _logger.info("[eval] 自动发现最佳匹配（跳过任务组）: %s/%s -> %s (分数: %s)", hf_dataset, hf_subset or "", best_task, score)
                    return best_task
            return None
        _logger.info("[eval] 自动发现最佳匹配: %s/%s -> %s (分数: %s)", hf_dataset, hf_subset or "", best_task, candidates[0][0])
        return best_task
    
    # 策略5: 如果所有策略都失败，尝试模糊匹配（用于 MMLU-Pro 等新数据集）
    # 对于 MMLU 相关数据集，尝试找到最接近的 MMLU 任务
    if "mmlu" in dataset_base_name:
        _logger.info("[eval] 尝试模糊匹配 MMLU 相关任务: %s/%s", hf_dataset, hf_subset or "")
        # 收集所有 MMLU 任务作为候选
        mmlu_candidates = []
        for task in available_tasks:
            if task in task_groups_set:
                continue
            task_lower = task.lower()
            if "mmlu" in task_lower:
                score = 0
                # 如果有 subset，尝试匹配 subset 的关键词
                if subset_lower:
                    subset_keywords = [s for s in subset_lower.replace("-", "_").split("_") if len(s) > 3]
                    matched = sum(1 for kw in subset_keywords if kw in task_lower)
                    score = matched * 3  # 每个匹配的关键词3分
                mmlu_candidates.append((score, len(task), task))
        
        if mmlu_candidates:
            mmlu_candidates.sort(key=lambda x: (-x[0], -x[1]))
            best_mmlu = mmlu_candidates[0][2]
            _logger.info("[eval] 模糊匹配到 MMLU 任务: %s/%s -> %s (分数: %s)", hf_dataset, hf_subset or "", best_mmlu, mmlu_candidates[0][0])
            return best_mmlu
    
    _logger.debug("[eval] 无法为 %s/%s 找到匹配的 lm_eval 任务", hf_dataset, hf_subset or "")
    return None


def _find_or_download_eval_yaml(hf_dataset, hf_subset):
    """
    尝试查找或下载数据集对应的 eval.yaml。
    查找顺序：
    1. testset_repo/yaml 下的缓存文件
    2. ~/.cache/huggingface/hub 下的数据集快照
    3. 从 HuggingFace Hub 下载
    返回: YAML 文件绝对路径 或 None
    """
    try:
        # Ensure mirror is used
        os.environ["HF_ENDPOINT"] = Config.HF_ENDPOINT
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        _logger.warning("[eval] huggingface_hub 未安装，无法自动下载 YAML")
        return None

    # 规范化命名：openai/gsm8k -> openai___gsm8k__main__eval.yaml
    # 注意：需与 testsets.json 中的命名习惯保持一致或兼容
    subset_str = hf_subset if hf_subset else "default"
    safe_base = f"{hf_dataset.replace('/', '___')}__{subset_str.replace('-', '_')}"
    repo_yaml_dir = Path(Config.PROJECT_ROOT) / "testset_repo" / "yaml"
    repo_yaml_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 检查 testset_repo 缓存 (支持 yaml, py, md)
    for ext in ["eval.yaml", ".py", ".md"]:
        # 构造可能的文件名模式
        # 1. {safe_base}__eval.yaml / {safe_base}__{filename}
        # 2. 直接查找以 safe_base 开头的文件
        candidates = []
        for f in os.listdir(repo_yaml_dir):
            if f.startswith(safe_base + "__") and (f.endswith(ext) or (ext == ".py" and f.endswith(".py"))):
                candidates.append(f)
        
        if candidates:
            # 优先选择匹配度高的
            # 这里简单返回第一个找到的
            return str(repo_yaml_dir / candidates[0])
        
    # Simplify: Check if we have mapped this dataset before in testsets.json?
    # No, we are in discover mode.
    
    # Let's search directory for matching prefix
    for f in os.listdir(repo_yaml_dir):
        if f.startswith(safe_base + "__"):
             return str(repo_yaml_dir / f)

    # 2. 检查 HuggingFace Hub 缓存 (snapshots)
    try:
        dataset_cache_path = Path(Config.EVAL_HF_DATASETS_CACHE) / f"{hf_dataset.replace('/', '___')}"
        if dataset_cache_path.exists():
            _logger.info("[eval] 检查 HF 数据集缓存: %s", dataset_cache_path)
            for snapshot in dataset_cache_path.iterdir():
                if snapshot.is_dir():
                    # 查找优先级: eval.yaml > hendrycks_test.py > {dataset}.py > *.py > README.md
                    candidates = []
                    
                    # 1. eval.yaml
                    if (snapshot / "eval.yaml").exists():
                        candidates.append(("eval.yaml", 100))
                    
                    # 2. .py files
                    py_files = list(snapshot.glob("*.py"))
                    for py in py_files:
                        if py.name == "hendrycks_test.py":
                            candidates.append((py.name, 90))
                        elif py.name == f"{hf_dataset.split('/')[-1]}.py":
                            candidates.append((py.name, 80))
                        else:
                            candidates.append((py.name, 50))
                            
                    # 3. README.md
                    if (snapshot / "README.md").exists():
                        candidates.append(("README.md", 10))
                    
                    if candidates:
                        # Sort by priority
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        best_file = candidates[0][0]
                        source_path = snapshot / best_file
                        
                        _logger.info("[eval] 在 HF Cache 找到配置文件: %s", source_path)
                        
                        # 复制到 testset_repo/yaml 并重命名
                        dest_name = f"{safe_base}__{best_file}"
                        dest_path = repo_yaml_dir / dest_name
                        shutil.copy2(source_path, dest_path)
                        return str(dest_path)
    except Exception as e:
        _logger.debug("[eval] 检查 HF Cache 失败: %s", e)
    
    # 3. 从 HF 下载
    try:
        _logger.info("[eval] 正在查询 HF 仓库文件列表: %s", hf_dataset)
        repo_files = list_repo_files(repo_id=hf_dataset, repo_type="dataset")
        
        target_file = None
        if "eval.yaml" in repo_files:
            target_file = "eval.yaml"
        else:
            # 优先查找 .py (如 hendrycks_test.py)
            py_files = [f for f in repo_files if f.endswith(".py")]
            if py_files:
                # 优先匹配 hendrycks_test.py (MMLU) 或 与数据集同名
                if "hendrycks_test.py" in py_files:
                    target_file = "hendrycks_test.py"
                else:
                    dataset_name = hf_dataset.split("/")[-1]
                    if f"{dataset_name}.py" in py_files:
                        target_file = f"{dataset_name}.py"
                    else:
                        target_file = py_files[0]
            elif "README.md" in repo_files:
                target_file = "README.md"
        
        if target_file:
            _logger.info("[eval] 发现配置文件: %s", target_file)
            local_path = hf_hub_download(repo_id=hf_dataset, filename=target_file, repo_type="dataset")
            
            # 复制到 testset_repo/yaml 并重命名
            # 命名格式: {safe_base}__{filename}
            dest_name = f"{safe_base}__{target_file}"
            dest_path = repo_yaml_dir / dest_name
            shutil.copy2(local_path, dest_path)
            _logger.info("[eval] 已部署配置文件到: %s", dest_path)
            return str(dest_path)
            
    except Exception as e:
        _logger.warning("[eval] 自动下载失败: %s", e)

    return None


def _prepare_custom_eval_task(hf_dataset, hf_subset, hf_split, task_dir):
    """
    检查是否有自定义 YAML 模板，若有则生成临时 lm_eval 任务 YAML。
    返回 (task_name, include_path) 或 (None, None)。
    """
    try:
        testsets_path = Path(Config.PROJECT_ROOT) / "testset_repo" / "data" / "testsets.json"
        
        target_entry = None
        if testsets_path.exists():
            with open(testsets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.get("testsets", {}).items():
                if v.get("hf_dataset") == hf_dataset and v.get("hf_subset") == hf_subset:
                    target_entry = v
                    break
        
        yaml_path = None
        if target_entry:
            yaml_path = target_entry.get("yaml_template_path")
            
        # 如果未配置或文件不存在，尝试自动查找/下载
        if not yaml_path or not os.path.exists(yaml_path):
             found = _find_or_download_eval_yaml(hf_dataset, hf_subset)
             if found:
                 yaml_path = found
        
        if not yaml_path:
            return None, None
            
        # 简单检查文件扩展名
        if yaml_path.endswith(".py"):
             _logger.info("[eval] 发现 Python 配置文件: %s，将作为 include_path 使用", yaml_path)
             return None, os.path.dirname(yaml_path)

        if not (yaml_path.endswith(".yaml") or yaml_path.endswith(".yml")):
             _logger.info("[eval] 发现非 YAML 配置文件: %s，跳过自动任务生成", yaml_path)
             return None, None

        cfg = _load_prompt_cfg(yaml_path)
        if not cfg:
            return None, None
            
        # 提取模板
        solvers = cfg.get("solvers", [])
        template_str = None
        for s in solvers:
            if s.get("name") == "prompt_template":
                template_str = s.get("args", {}).get("template")
                break
        
        if not template_str:
            return None, None
            
        # 提取字段映射
        field_spec = cfg.get("field_spec", {})
        input_field = field_spec.get("input", "question")
        target_field = field_spec.get("target", "answer")
        
        # 构造 doc_to_text
        # 将 {prompt} 替换为 {{input_field}}
        # 注意：YAML 中的 {prompt} 可能是 python format 风格，而 lm_eval 使用 Jinja2
        doc_to_text = template_str.replace("{prompt}", "{{%s}}" % input_field)
        
        # 构造任务定义
        # 我们创建一个继承自 huggingface 自动配置的任务
        # 或者尽可能复用已有信息
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", f"{hf_dataset}_{hf_subset}")
        
        task_def = {
            "task": safe_name,
            "dataset_path": hf_dataset,
            "dataset_name": hf_subset if hf_subset else None,
            "output_type": "generate_until",
            "training_split": "train",
            "test_split": hf_split or "test",
            "doc_to_text": doc_to_text,
            "doc_to_target": "{{%s}}" % target_field,
            # 添加默认的 generation_kwargs
            "generation_kwargs": {
                "until": ["\n\n", "Therefore"], # 简单的停止词，防止无限生成
                "do_sample": False,
                "temperature": 0.0
            },
             # 如果是 math 任务，通常需要 regex 提取
            "metric_list": [
                {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
            ]
        }
        
        # 清理 None 值
        task_def = {k: v for k, v in task_def.items() if v is not None}
        
        # 写入临时文件
        custom_yaml_dir = Path(task_dir) / "custom_tasks"
        custom_yaml_dir.mkdir(parents=True, exist_ok=True)
        custom_yaml_path = custom_yaml_dir / f"{safe_name}.yaml"
        
        with open(custom_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(task_def, f, allow_unicode=True)
            
        _logger.info("[eval] 已生成自定义任务 YAML: %s (基于 %s)", custom_yaml_path, yaml_path)
        return safe_name, str(custom_yaml_dir)
        
    except Exception as e:
        _logger.warning("[eval] 生成自定义任务失败: %s", e)
        return None, None


def _prepare_multimodal_model_for_eval(model_path, output_path):
    """为 hf-multimodal 准备兼容模型目录（最小侵入：仅在评估目录生成临时配置）。"""
    try:
        cfg_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(cfg_path):
            return model_path
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        top_type = (cfg.get("model_type") or "").strip().lower()
        vcfg = cfg.get("vision_config") if isinstance(cfg.get("vision_config"), dict) else {}
        vision_type = (vcfg.get("model_type") or "").strip().lower()

        # 仅处理常见的 Qwen VL 退化配置（top-level 仍是 qwen2）
        if top_type != "qwen2" or vision_type not in ("qwen2_5_vl", "qwen2_vl", "qwen3_vl", "qwen3_vl_moe"):
            return model_path

        arch_map = {
            "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
            "qwen2_vl": "Qwen2VLForConditionalGeneration",
            "qwen3_vl": "Qwen3VLForConditionalGeneration",
            "qwen3_vl_moe": "Qwen3VLMoeForConditionalGeneration",
        }

        patched_dir = os.path.join(output_path, "_mm_model")
        os.makedirs(patched_dir, exist_ok=True)

        # 除 config.json 外尽量复用原文件（软链接）
        for name in os.listdir(model_path):
            if name == "config.json":
                continue
            src = os.path.join(model_path, name)
            dst = os.path.join(patched_dir, name)
            try:
                if os.path.islink(dst) or os.path.isfile(dst):
                    os.remove(dst)
                elif os.path.isdir(dst):
                    continue
            except Exception:
                pass
            try:
                os.symlink(src, dst)
            except Exception:
                # 软链接失败时回退复制（小文件）
                if os.path.isfile(src):
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass

        patched_cfg = dict(cfg)
        patched_cfg["model_type"] = vision_type
        patched_cfg["architectures"] = [arch_map.get(vision_type, patched_cfg.get("architectures", ["Qwen2_5_VLForConditionalGeneration"])[0])]
        with open(os.path.join(patched_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(patched_cfg, f, ensure_ascii=False, indent=2)

        _logger.info("[eval] 已生成多模态兼容配置: %s (model_type=%s)", patched_dir, vision_type)
        return patched_dir
    except Exception as e:
        _logger.warning("[eval] 生成多模态兼容配置失败，使用原模型目录: %s", e)
        return model_path


def run_lm_eval_stream(
    model_path,
    output_path,
    task_name,
    callback,
    start_prog,
    end_prog,
    task_control=None,
    limit="0.5",
    hf_dataset=None,
    hf_subset=None,
    hf_split=None,
    lm_eval_task=None,
    custom_include_path=None,
    sampling="sequential",
):
    """
    与 mergeKit_alpha 一致的评估流程：Popen 流式执行 lm_eval，结果写入 output_path。
    若提供 lm_eval_task（测试集创建/选择时自动写入），优先使用；否则用 hf_dataset/hf_subset 推断或内置/配置文件映射。
    返回 { acc, f1, samples, time, context, per_task_acc }。
    """
    if task_control is None:
        task_control = {}
    os.makedirs(output_path, exist_ok=True)

    # 保留调用方传入的任务名（例如 hellaswag）；仅在为空时走后续推断逻辑
    task_name = (task_name or "").strip()
    # 获取任务组集合，用于验证（提前获取，避免重复初始化）
    task_groups_set = None
    try:
        from lm_eval.tasks import TaskManager
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            task_manager_temp = TaskManager()
        task_groups_set = set(task_manager_temp.all_groups) if hasattr(task_manager_temp, 'all_groups') else set()
    except Exception:
        task_groups_set = set()
    
    # 尝试生成自定义任务（基于 YAML 模板）
    custom_task_name, custom_include_path = _prepare_custom_eval_task(hf_dataset, hf_subset, hf_split, output_path)
    if custom_task_name:
        task_name = custom_task_name
        _logger.info("[eval] 使用自定义任务: %s (include_path=%s)", task_name, custom_include_path)
    
    if (lm_eval_task or "").strip() and not custom_task_name:
        task_name = (lm_eval_task or "").strip()
    elif (hf_dataset or "").strip():
        # 测试集仓库分支：先查部署用映射文件，再内置映射，最后按 MMLU 规则推断
        hf_key = (hf_dataset or "").strip()
        subset = (hf_subset or "").strip()
        extra = _load_eval_task_mapping()
        exact_key = "%s|%s" % (hf_key, subset) if subset else hf_key
        if exact_key in extra:
            task_name = extra[exact_key]
        elif hf_key in extra:
            task_name = extra[hf_key] if not subset else (extra[hf_key] + "_" + subset.replace("-", "_"))
        else:
            mapped = HF_DATASET_TO_LM_EVAL_TASK.get(hf_key.lower(), None)
            if mapped:
                # 注意：内置映射中的 "mmlu" 是任务组，不能直接使用
                if subset and mapped == "mmlu":
                    task_name = "mmlu_" + subset.replace("-", "_")
                elif mapped == "mmlu":
                    # 如果没有 subset 且映射是 "mmlu"（任务组），不能使用，需要自动发现或报错
                    task_name = ""  # 保持为空，让自动发现处理
                else:
                    task_name = mapped
            # 如果 mapped 为 None，说明找不到映射，task_name 保持为空
        
        # CMMMU 数据集仅接受 cmmmu_* / mmmu_val_* 任务，防止历史错误映射污染到其它基准
        hf_ds_lower = (hf_dataset or "").strip().lower()
        if hf_ds_lower in ("m-a-p/cmmmu", "cmmmu/cmmmu"):
            t = (task_name or "").strip()
            if t and not (t == "cmmmu" or t.startswith("cmmmu_") or t.startswith("mmmu_val_") or "," in t):
                _logger.warning("[eval] 忽略不兼容的 CMMMU 任务映射: %s", t)
                task_name = "cmmmu"

        # 验证从映射获取的任务名不是任务组
        if task_name and task_name.strip() and task_groups_set:
            if task_name in task_groups_set:
                _logger.warning("[eval] 从映射获取的任务名是任务组，无效: %s，将尝试自动发现", task_name)
                task_name = ""  # 重置为空，让自动发现处理
        if task_name and task_name.strip():
            available_tasks = _get_available_lm_eval_tasks()
            if available_tasks and task_name not in available_tasks:
                # cmmmu 是任务组占位符，后续会进入专门解析逻辑，不能在这里清空
                if (task_name or "").strip().lower() != "cmmmu":
                    _logger.warning("[eval] 从映射获取的任务名不存在，尝试自动发现: %s", task_name)
                    task_name = ""

    # 若传入 lm_eval_task 为任务组名 "cmmmu"，按本机可用任务动态解析（避免硬编码到不存在任务）
    if (task_name or "").strip() == "cmmmu":
        available_tasks = _get_available_lm_eval_tasks() or []
        subset_key = (hf_subset or "").strip().lower().replace("-", "_")

        # 1) 优先使用原生 cmmmu_*（若环境已安装对应任务）
        cmmmu_tasks = sorted(t for t in available_tasks if t.startswith("cmmmu_"))
        if cmmmu_tasks:
            if subset_key and subset_key != "all":
                # 尝试按子集名匹配；匹配不到则回退全部，保证任务可执行
                chosen = [t for t in cmmmu_tasks if subset_key in t.lower()]
                task_name = ",".join(chosen or cmmmu_tasks)
            else:
                task_name = ",".join(cmmmu_tasks)
            _logger.info("[eval] CMMMU 解析到原生任务: %s", task_name)
        else:
            # 2) 兼容回退：当前环境若仅有 mmmu_val_*，则使用其可用子任务（不影响其它数据集）
            mmmu_tasks = sorted(t for t in available_tasks if t.startswith("mmmu_val_"))
            if mmmu_tasks:
                # 仅保留当前环境可加载到数据集配置的任务，避免部分 config 缺失导致进程中断
                try:
                    cfg_norm = set()
                    eval_cache = getattr(Config, "EVAL_HF_DATASETS_CACHE", None) or os.path.join(Config.PROJECT_ROOT, "cache", "eval_datasets")
                    local_cfg_root = Path(eval_cache) / "MMMU___mmmu"
                    if local_cfg_root.exists() and local_cfg_root.is_dir():
                        for p in local_cfg_root.iterdir():
                            if p.is_dir():
                                cfg_norm.add(p.name.strip().lower().replace("-", "_"))
                        if cfg_norm:
                            _logger.info("[eval] MMMU 使用本地缓存配置过滤，共 %d 项", len(cfg_norm))
                    if not cfg_norm:
                        from datasets import get_dataset_config_names
                        cfgs = get_dataset_config_names("MMMU/MMMU", trust_remote_code=True) or []
                        cfg_norm = {str(c).strip().lower().replace("-", "_") for c in cfgs}
                    filtered = []
                    for t in mmmu_tasks:
                        suffix = t.replace("mmmu_val_", "", 1).strip().lower().replace("-", "_")
                        if suffix in cfg_norm:
                            filtered.append(t)
                    if filtered:
                        mmmu_tasks = filtered
                except Exception as e:
                    _logger.warning("[eval] 读取 MMMU 可用 configs 失败，将使用全部 mmmu_val_* 任务: %s", e)

                if subset_key and subset_key != "all":
                    # 领域到关键词映射（匹配不到则回退全部可用任务，确保可执行）
                    keyword_map = {
                        "art_and_design": ["art", "design"],
                        "business": ["business", "finance", "economics", "manage"],
                        "health_and_medicine": ["medical", "medicine", "health", "pharmacy", "biology"],
                        "humanities_and_social_sciences": ["history", "literature", "sociology", "psychology", "law", "politics"],
                        "science": ["physics", "chemistry", "math", "biology", "science"],
                        "technology_and_engineering": ["computer", "engineering", "electronics", "energy"],
                    }
                    kws = keyword_map.get(subset_key, [subset_key])
                    chosen = [t for t in mmmu_tasks if any(k in t.lower() for k in kws)]
                    task_name = ",".join(chosen or mmmu_tasks)
                else:
                    task_name = ",".join(mmmu_tasks)
                _logger.warning("[eval] 环境无 cmmmu_*，回退使用 mmmu_val_* 任务: %s", task_name)
            else:
                # CMMMU 请求下若环境既无 cmmmu_* 也无 mmmu_val_*，直接报错，避免误匹配到其它 mmlu 任务
                raise ValueError("当前 lm_eval 环境未注册 CMMMU 兼容任务（缺少 cmmmu_* 与 mmmu_val_*）。请安装含多模态任务的 lm_eval 版本后重试。")

    # 验证任务名是否有效，如果无效则尝试自动发现
    if not task_name or task_name.strip() == "":
        _logger.info("[eval] 无法推断任务名，尝试自动发现: hf_dataset=%s, hf_subset=%s", hf_dataset, hf_subset)
        discovered_task = _auto_discover_lm_eval_task(hf_dataset, hf_subset)
        if discovered_task:
            # 最终验证：确保不是任务组（使用之前获取的 task_groups_set）
            if task_groups_set and discovered_task in task_groups_set:
                _logger.error("[eval] 自动发现的任务是任务组，无效: %s", discovered_task)
                raise ValueError(f"自动发现的任务 '{discovered_task}' 是任务组而非具体任务，无法使用。请手动配置任务映射。")
            
            task_name = discovered_task
            # 自动保存映射到配置文件
            try:
                extra = _load_eval_task_mapping()
                hf_key = (hf_dataset or "").strip()
                subset = (hf_subset or "").strip()
                exact_key = "%s|%s" % (hf_key, subset) if subset else hf_key
                hf_ds_lower2 = (hf_dataset or "").strip().lower()
                allow_save = True
                if hf_ds_lower2 in ("m-a-p/cmmmu", "cmmmu/cmmmu"):
                    dt = (discovered_task or "").strip()
                    allow_save = bool(dt.startswith("cmmmu_") or dt.startswith("mmmu_val_"))
                if allow_save:
                    extra[exact_key] = discovered_task
                    # 如果没有 subset，也保存通用映射
                    if not subset:
                        extra[hf_key] = discovered_task
                    _save_eval_task_mapping(extra)
                    _logger.info("[eval] 已自动保存映射: %s -> %s", exact_key, discovered_task)
                else:
                    _logger.warning("[eval] CMMMU 自动发现结果不兼容，跳过保存映射: %s", discovered_task)
            except Exception as save_err:
                _logger.warning("[eval] 自动保存映射失败: %s", save_err)
        else:
            error_msg = f"无法推断有效的 lm_eval 任务名。hf_dataset={hf_dataset}, hf_subset={hf_subset}, lm_eval_task={lm_eval_task}"
            _logger.error("[eval] %s", error_msg)
            raise ValueError(error_msg + "。请检查测试集配置或添加任务映射到 config/eval_task_mapping.json")
    
    # 最终验证：确保任务名不是任务组（使用之前获取的 task_groups_set）
    # 如果 task_name 包含逗号，说明是任务列表，跳过验证直接传递给 CLI
    if task_name and task_name.strip() and "," not in task_name:
        if task_groups_set and task_name in task_groups_set:
            error_msg = f"任务名 '{task_name}' 是任务组而非具体任务，无法使用。请使用具体任务名或添加任务映射。"
            _logger.error("[eval] %s", error_msg)
            raise ValueError(error_msg)
        
        # 额外验证：尝试加载任务，确保可以正常加载（不是任务组）
        try:
            from lm_eval.tasks import TaskManager
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                verify_tm = TaskManager()
            # 0.4.11+: 优先用 load()；旧版回退 load_task_or_group
            if hasattr(verify_tm, 'load'):
                result = verify_tm.load(task_name)
                if result.get("groups") and task_name in result["groups"]:
                    error_msg = f"任务名 '{task_name}' 是任务组而非具体任务，无法直接使用。请使用具体任务名。"
                    _logger.error("[eval] %s", error_msg)
                    raise ValueError(error_msg)
            else:
                loaded = verify_tm.load_task_or_group(task_name)
                from collections import ChainMap
                if isinstance(loaded, ChainMap):
                    error_msg = f"任务名 '{task_name}' 是任务组（返回 ChainMap），无法直接使用。"
                    _logger.error("[eval] %s", error_msg)
                    raise ValueError(error_msg)
        except ValueError:
            raise
        except Exception as verify_err:
            _logger.warning("[eval] 验证任务加载失败（非致命）: %s", verify_err)

    if task_name == "all":
        target_tasks_list = list(STANDARD_BENCHMARKS)
        cmd_task_str = ",".join(target_tasks_list)
    elif "," in (task_name or ""):
        target_tasks_list = [t.strip() for t in task_name.split(",") if t.strip()]
        cmd_task_str = ",".join(target_tasks_list)
    else:
        target_tasks_list = [task_name]
        cmd_task_str = task_name

    # 改进：不再依赖当前环境的 torch 判断设备，而是检查系统是否有 GPU (nvidia-smi)
    # 因为后端环境可能没有 torch，但目标环境 (mergenetic) 有 GPU
    try:
        subprocess.check_call("nvidia-smi", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        device = "cuda"
    except Exception:
        _logger.warning("[eval] 未检测到 GPU (nvidia-smi 失败)，回退到 CPU")
        device = "cpu"

    # 注意：当前版本的 lm_eval CLI 可能不支持 --samples 参数
    # 对于随机采样，我们使用 --limit 配合随机种子，或者回退到顺序采样
    # 由于 lm_eval CLI 的限制，随机采样功能暂时使用顺序采样 + limit 实现
    # 真正的随机采样需要等待 lm_eval 支持或使用 Python API
    if sampling == "random":
        _logger.info("[eval] 随机采样模式：由于 lm_eval CLI 限制，使用顺序采样 + limit 实现近似随机效果")
        # 使用 limit 参数，虽然不能完全随机，但可以限制样本数量
        # 注意：这实际上是顺序采样，不是真正的随机采样

    # 记录最终使用的任务名，便于调试
    _logger.info("[eval] 最终使用的任务名: %s (cmd_task_str=%s)", task_name, cmd_task_str)
    
    # 针对部分模型（如 Qwen 系列）在 bf16 上可能出现退化输出的情况，按路径名进行最小化参数修正
    model_args = "pretrained=%s,trust_remote_code=True" % model_path
    try:
        base_name = os.path.basename(str(model_path)).lower()
        if "qwen" in base_name and "math" in base_name:
            # Qwen2.5 Math 在 RTX 4090 (bf16) 上表现正常，强制 float16 反而导致溢出 (!!!!)
            # 因此移除之前的强制 float16 逻辑，让其使用默认的 bfloat16
            # model_args += ",dtype=float16"
            # _logger.info("[eval] 检测到 Qwen Math 模型，强制 dtype=float16 以避免生成异常")
            pass
    except Exception:
        pass

    # 尝试使用 MERGENETIC_PYTHON 所在环境的 lm_eval
    lm_eval_bin = "lm_eval"
    if MERGENETIC_PYTHON and os.path.isabs(MERGENETIC_PYTHON):
        bin_dir = os.path.dirname(MERGENETIC_PYTHON)
        candidate = os.path.join(bin_dir, "lm_eval")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            lm_eval_bin = candidate
            _logger.info("[eval] 使用指定环境的 lm_eval: %s", lm_eval_bin)
        else:
            _logger.warning("[eval] 未在 %s 找到 lm_eval，将尝试系统 PATH", bin_dir)

    # 多模态任务需使用 hf-multimodal / vllm-vlm；这里按任务名自动切换到 hf-multimodal
    is_multimodal_task = any((t or "").startswith("mmmu_val_") or (t or "").startswith("cmmmu_") for t in (target_tasks_list or []))
    model_backend = "hf-multimodal" if is_multimodal_task else "hf"
    eval_model_path = model_path
    if is_multimodal_task:
        _logger.info("[eval] 检测到多模态任务，使用模型后端: %s", model_backend)
        eval_model_path = _prepare_multimodal_model_for_eval(model_path, output_path)
        model_args = model_args.replace("pretrained=%s" % model_path, "pretrained=%s" % eval_model_path, 1)

    batch_size_arg = "1" if is_multimodal_task else "auto"
    cmd = [
        lm_eval_bin, "--model", model_backend,
        "--model_args", model_args,
        "--tasks", cmd_task_str,
        "--device", device,
        "--batch_size", batch_size_arg,
        "--output_path", output_path,
        "--log_samples",
    ]
    
    if custom_include_path:
        cmd.extend(["--include_path", custom_include_path])

    # 使用 --limit 参数（顺序采样和随机采样都使用 limit，因为 CLI 不支持 --samples）
    if str(limit) != "1.0":
        cmd.extend(["--limit", str(limit)])

    # Wrap with conda activate
    cmd = _get_conda_activate_cmd(cmd)

    eval_cache = getattr(Config, "EVAL_HF_DATASETS_CACHE", None) or os.path.join(Config.PROJECT_ROOT, "cache", "eval_datasets")
    os.makedirs(eval_cache, exist_ok=True)
    eval_env = os.environ.copy()
    eval_env["HF_DATASETS_CACHE"] = eval_cache

    _logger.info("[eval] 执行: %s (HF_DATASETS_CACHE=%s)", " ".join(cmd), eval_cache)
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=eval_env,
        **_popen_group_kwargs(),
    )
    task_control["process"] = process

    last_update_time = 0.0
    last_lines = []
    max_tail = 40
    last_count = 0
    total_count = 0
    last_rate = 0.0
    last_rate_time = 0.0
    pct = None
    progress_path = os.path.join(output_path, "progress.json")
    
    # 使用字符级读取以处理 \r (tqdm 进度条)
    buf = []
    while True:
        if task_control.get("aborted"):
            ProcessManager.kill_process_tree(process)
            raise RuntimeError("任务已被用户手动终止")
        
        char = process.stdout.read(1)
        if not char:
            if process.poll() is not None:
                break
            continue

        if char != '\n' and char != '\r':
            buf.append(char)
            continue
            
        # 遇到换行或回车，处理当前缓冲区作为一行
        line = "".join(buf)
        buf = []
        
        # 即使是空行也继续处理（虽然下面 check 了 line）
        if True: # 保持缩进结构
            _logger.info("[eval] %s", line)
            last_lines.append(line)
            if len(last_lines) > max_tail:
                last_lines.pop(0)
            now = time.time()
            if now - last_update_time > 0.5:
                clean = line.strip()[:80]
                if clean:
                    prog_val = start_prog
                    m = re.search(r"(\d+)\s*%\s*\|\s*(\d+)\s*/\s*(\d+)", line)
                    if m:
                        try:
                            pct = int(m.group(1))
                            span = max(0, int(end_prog) - int(start_prog))
                            prog_val = int(start_prog) + int((span * pct) / 100)
                            if prog_val > int(end_prog):
                                prog_val = int(end_prog)
                            total_count = int(m.group(3))
                            last_count = int(m.group(2))
                        except Exception:
                            prog_val = start_prog
                    else:
                        # Try evolutionary pattern: "Generation 5/10"
                        m3 = re.search(r"Generation\s+(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
                        if m3:
                            try:
                                cur = int(m3.group(1))
                                tot = int(m3.group(2))
                                pct = int((cur * 100) / max(1, tot))
                                total_count = tot
                                last_count = cur
                            except:
                                pass
                                
                        m2 = re.search(r"(\d+)\s*/\s*(\d+).*?it/s", line)
                        if m2:
                            try:
                                cur = int(m2.group(1))
                                tot = max(1, int(m2.group(2)))
                                pct = int((cur * 100) / tot)
                                span = max(0, int(end_prog) - int(start_prog))
                                prog_val = int(start_prog) + int((span * pct) / 100)
                                if prog_val > int(end_prog):
                                    prog_val = int(end_prog)
                                total_count = tot
                                last_count = cur
                            except Exception:
                                prog_val = start_prog
                    r = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*it/s", line)
                    if r:
                        try:
                            last_rate = float(r.group(1))
                            last_rate_time = now
                        except Exception:
                            pass
                    eta_txt = ""
                    rate = last_rate
                    if rate <= 0 and total_count > 0 and last_count > 0:
                        dt = max(0.1, now - (last_rate_time or (now - 1)))
                        rate = (last_count / dt) if dt > 0 else 0.0
                    if rate > 0 and total_count > 0 and last_count >= 0 and total_count >= last_count:
                        remain = (total_count - last_count) / rate
                        mins = int(remain // 60)
                        secs = int(remain % 60)
                        eta_txt = " ETA %d:%02d" % (mins, secs)
                    # 始终写入进度快照（即使速率未知），以便前端刷新早期阶段的进度
                    try:
                        if total_count > 0:
                            # Read existing first to preserve custom fields (like best_genotype/current_best)
                            existing_pd = {}
                            if os.path.exists(progress_path):
                                try:
                                    with open(progress_path, "r", encoding="utf-8") as f:
                                        existing_pd = json.load(f)
                                except:
                                    pass

                            pd = {
                                "status": "running",
                                "current": int(last_count),
                                "total": int(total_count),
                                "percent": int(max(0, min(100, pct))) if isinstance(pct, int) else (int((last_count * 100) / max(1, total_count)) if total_count > 0 else 0),
                            }
                            if rate > 0 and total_count > 0 and last_count >= 0 and total_count >= last_count:
                                pd["eta_seconds"] = float((total_count - last_count) / rate)
                            
                            existing_pd.update(pd)
                            
                            with open(progress_path, "w", encoding="utf-8") as f:
                                json.dump(existing_pd, f, ensure_ascii=False)
                    except Exception:
                        pass
                    callback(prog_val, "[%s] %s%s" % (cmd_task_str, clean, eta_txt))
                last_update_time = now
    process.wait()
    duration = time.time() - start_time

    # Fix: Ensure progress is marked as completed/failed at the end
    try:
        final_status = "completed" if process.returncode == 0 else "failed"
        final_percent = 100 if process.returncode == 0 else 0
        
        # Read existing to preserve evolutionary data
        existing_data = {}
        if os.path.exists(progress_path):
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except:
                pass
        
        existing_data.update({
            "status": final_status,
            "percent": final_percent if final_status == "completed" else existing_data.get("percent", 0),
            "eta_seconds": 0
        })
        
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False)
    except Exception as e:
        _logger.warning("[eval] Failed to write final progress: %s", e)

    def _fail_msg(tip):
        tail = "\n".join(last_lines[-30:]) if last_lines else ""
        return "%s\n--- lm_eval 输出尾行 ---\n%s" % (tip, tail)

    if process.returncode != 0:
        _logger.warning("[eval] lm_eval 退出码 %s", process.returncode)
        tail_text = "\n".join(last_lines[-30:]) if last_lines else ""
        hint = ""
        if "dataclass" in tail_text or "Features.from_dict" in tail_text:
            hint = " 若为 datasets 缓存兼容问题，可尝试删除 HF 数据集缓存或 pip install -U datasets。"
        elif "does not recognize this architecture" in tail_text or ("model type" in tail_text and "Transformers" in tail_text):
            hint = " 该架构可能未被当前 transformers 版本识别，请确认 pip install --upgrade 'lm_eval[hf]' transformers 已升级到最新。"
        elif "canUse32BitIndexMath" in tail_text:
            hint = " 大张量索引溢出：可尝试 batch_size=1 或减小 --limit。"
        err_path = os.path.join(output_path, "eval_stderr.txt")
        try:
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(_fail_msg("lm_eval 退出码: %s%s" % (process.returncode, hint)))
        except Exception:
            pass
        raise RuntimeError("评测进程退出码 %s，详见任务目录 eval_stderr.txt.%s\n--- 输出尾行 ---\n%s" % (process.returncode, hint, tail_text[-2000:] if len(tail_text) > 2000 else tail_text))

    json_path = _pick_latest_json(output_path)
    if not json_path:
        _logger.warning("[eval] 未在 %s 找到 json 结果", output_path)
        raise RuntimeError(_fail_msg("评测未在 %s 生成结果文件，请检查 lm_eval 与数据集环境" % output_path))

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_results = data.get("results", data)
        total_acc = 0.0
        total_samples = 0
        valid_tasks_count = 0
        per_task_acc = {}
        _logger.info("[eval] DEBUG: target_tasks_list=%s", target_tasks_list)
        for t in target_tasks_list:
            task_res = raw_results.get(t, {})
            _logger.info("[eval] DEBUG: checking task %s, found in results: %s", t, t in raw_results)
            curr_acc = 0.0
            found = False
            for key in (
                "acc_norm,none",
                "acc_norm",
                "acc,none",
                "acc",
                "exact_match,strict-match",
                "exact_match,flexible-extract",
                "exact_match",
            ):
                if key in task_res:
                    curr_acc = float(task_res[key])
                    found = True
                    break
            if found:
                total_acc += curr_acc
                valid_tasks_count += 1
            per_task_acc[t] = round(curr_acc * 100, 2)
            curr_samples = 0
            if "n-samples" in data and t in data["n-samples"]:
                s = data["n-samples"][t]
                curr_samples = s.get("effective", 0) if isinstance(s, dict) else int(s)
                _logger.info("[eval] DEBUG: Found n-samples for %s: %s (effective=%s)", t, s, curr_samples)
            elif "n_samples" in data and t in data.get("n_samples", {}):
                s = data["n_samples"][t]
                curr_samples = s.get("effective", 0) if isinstance(s, dict) else int(s)
                _logger.info("[eval] DEBUG: Found n_samples for %s: %s", t, s)
            elif task_res.get("n_samples") is not None:
                curr_samples = int(task_res["n_samples"])
                _logger.info("[eval] DEBUG: Found task_res n_samples for %s: %s", t, curr_samples)
            else:
                _logger.warning("[eval] DEBUG: No n-samples found for %s. Keys in data: %s", t, list(data.keys()))
            total_samples += curr_samples

        avg_acc = (total_acc / valid_tasks_count) if valid_tasks_count > 0 else 0.0
        
        # 尝试从 samples_*.jsonl 文件修正 total_samples
        if total_samples == 0:
            try:
                jsonl_files = [f for f in os.listdir(output_path) if f.startswith("samples_") and f.endswith(".jsonl")]
                if jsonl_files:
                    # 取最新的文件
                    jsonl_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_path, x)), reverse=True)
                    latest_jsonl = os.path.join(output_path, jsonl_files[0])
                    with open(latest_jsonl, "r", encoding="utf-8") as jf:
                        # 计算行数 (排除空行)
                        line_count = sum(1 for line in jf if line.strip())
                        if line_count > 0:
                            total_samples = line_count
                            _logger.info("[eval] 从 jsonl 文件修正 samples count: %s", total_samples)
            except Exception as e:
                _logger.warning("[eval] 尝试从 jsonl 统计样本数失败: %s", e)

        if total_samples == 0:
            total_samples = int(data.get("config", {}).get("limit", 0)) * len(target_tasks_list)
        context_len = _eval_get_context_length(model_path)
        return {
            "acc": round(avg_acc * 100, 2),
            "f1": round(avg_acc, 4),
            "samples": int(total_samples),
            "time": round(duration, 2),
            "context": int(context_len or 0),
            "per_task_acc": per_task_acc,
        }
    except Exception as e:
        _logger.exception("[eval] 解析结果失败: %s", e)
        raise RuntimeError("解析评测结果失败: %s" % e) from e


def run_eval_only_task(task_id, params, update_progress_callback, task_control=None):
    """仅评估任务：与 mergeKit_alpha 一致，调用 run_lm_eval_stream 后按 alpha 格式写 metadata 并返回。
    当选择「测试集仓库」时，使用 params 中的 hf_dataset / hf_subset / hf_split，不再使用内置 benchmark。"""
    if task_control is None:
        task_control = {}
    task_dir = os.path.join(MERGE_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    model_path = (params.get("model_path") or "").strip()
    # 优先使用测试集仓库参数：有 hf_dataset 或 testset_id 时走测试集分支，否则用 dataset（内置 all/hellaswag 等）
    hf_dataset = (params.get("hf_dataset") or "").strip() or None
    hf_subset = (params.get("hf_subset") or "").strip() or None
    hf_split = (params.get("hf_split") or "").strip() or None
    if not hf_dataset and params.get("testset_id"):
        # 从测试集仓库列表解析出 hf_dataset（与 app 端一致：api 已写入 dataset/hf_*，此处仅作兜底）
        for key in ("dataset", "hf_dataset"):
            v = (params.get(key) or "").strip()
            if v and "/" in v:
                hf_dataset = v
                if hf_subset is None:
                    hf_subset = (params.get("hf_subset") or "").strip() or None
                if hf_split is None:
                    hf_split = (params.get("hf_split") or "validation").strip() or "validation"
                break
    dataset_name = params.get("dataset", "all")
    # 若为测试集仓库选定数据集，展示名与任务名由 run_lm_eval_stream 内根据 hf_* 决定
    if hf_dataset:
        dataset_name = hf_dataset + ((" (" + (hf_subset or "") + ")") if hf_subset else "")
        _logger.info("[eval] 使用测试集仓库: hf_dataset=%s hf_subset=%s hf_split=%s", hf_dataset, hf_subset, hf_split)
    limit_val = params.get("limit", "0.5")
    model_name = params.get("model_name", "Eval Task")

    metadata = {
        "id": task_id,
        "type": "eval_only",
        "custom_name": model_name,
        "created_at": params.get("created_at", time.time()),
        "dataset": dataset_name,
        "status": "running",
        "testset_id": (params.get("testset_id") or None),
        "hf_dataset": hf_dataset,
        "hf_subset": hf_subset,
        "hf_split": hf_split,
    }

    def _write_meta(status, metrics=None, error=None):
        metadata["status"] = status
        if metrics is not None:
            metadata["metrics"] = metrics
        if error:
            metadata["error"] = error
        metadata["model_path"] = model_path
        _write_metadata(task_id, task_dir, metadata)

    try:
        _write_meta("running")
        if not model_path or not os.path.isdir(model_path):
            msg = "模型路径无效或不是目录: %s" % (model_path or "(空)")
            _write_meta("error", error=msg)
            return {"status": "error", "error": msg}

        update_progress_callback(10, "准备评估模型: %s" % model_name)
        display_task = "Standard Benchmark" if (dataset_name == "all" and not hf_dataset) else dataset_name
        update_progress_callback(30, "正在启动评估 (%s)..." % display_task)

        prompt_yaml = _resolve_prompt_yaml_for_testset(params.get("testset_id"), hf_dataset, hf_subset)
        use_yaml = bool(prompt_yaml and os.path.isfile(prompt_yaml) and (prompt_yaml.endswith(".yaml") or prompt_yaml.endswith(".yml")))
        if use_yaml:
            metadata["yaml_template"] = prompt_yaml
            _write_metadata(task_id, task_dir, metadata)
            
            # 直接调用 lm_eval，不再使用 eval_worker.py 避免递归死循环
            # 解析任务名和包含路径
            custom_task_name = os.path.splitext(os.path.basename(prompt_yaml))[0]
            custom_include_path = os.path.dirname(prompt_yaml)
            
            # 复用 run_lm_eval_stream 的逻辑，但传入 custom_include_path
            metrics = run_lm_eval_stream(
                model_path,
                task_dir,
                custom_task_name,
                update_progress_callback,
                30,
                95,
                task_control,
                limit=limit_val,
                hf_dataset=hf_dataset,
                hf_subset=hf_subset,
                hf_split=hf_split,
                custom_include_path=custom_include_path, # 传入包含路径
                sampling=(params.get("sampling") or "sequential").strip() or "sequential",
            )


        else:
            metrics = run_lm_eval_stream(
                model_path,
                task_dir,
                dataset_name,
                update_progress_callback,
                30,
                95,
                task_control,
                limit=limit_val,
                hf_dataset=hf_dataset,
                hf_subset=hf_subset,
                hf_split=hf_split,
                lm_eval_task=(params.get("lm_eval_task") or "").strip() or None,
                sampling=(params.get("sampling") or "sequential").strip() or "sequential",
            )

        # 与 mergeKit_alpha 一致：雷达图固定为三维 [Accuracy, Efficiency, Context]
        context_val = metrics.get("context", 0) or 0
        context_score = min(100, (int(context_val) / 1024) * 10) if context_val else 0
        # Try to find base model name from config
        base_model_name = "Baseline"
        try:
            cfg_path = os.path.join(model_path, "config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                    nm = cfg.get("_name_or_path")
                    if nm and isinstance(nm, str):
                        # 如果是绝对路径且不存在，则忽略（避免显示训练时的临时路径，如 WiNGPT...）
                        if nm.startswith("/") and not os.path.exists(nm):
                            pass
                        # 如果不包含路径分隔符且本地不存在（且不是常用HF模型名格式），也忽略
                        # 避免显示如 "WiNGPT2-Llama-3-8B-Base" 这样的内部训练名称
                        elif "/" not in nm and not os.path.exists(nm):
                            pass
                        elif "/" in nm:
                            base_model_name = os.path.basename(nm)
                        else:
                             base_model_name = str(nm)
        except:
            pass

        # Efficiency：基于 throughput 或 samples/time 计算速度，归一化到 1-100
        # 默认基准速度设为 10.0 samples/s (生成任务通常较慢)
        eval_time = metrics.get("time", 0) or 0
        eval_samples = metrics.get("samples", 0) or 0
        throughput = metrics.get("throughput", 0)

        base_sps = float(getattr(Config, "EFFICIENCY_BASE_SPS", 10.0) or 10.0)
        base_sps = max(0.1, base_sps)
        
        efficiency_score = 0
        if throughput > 0:
             efficiency_score = (throughput / base_sps) * 100
        elif eval_time > 0 and eval_samples > 0:
            speed = eval_samples / eval_time
            efficiency_score = (speed / base_sps) * 100
            
        efficiency_score = min(100, max(1, efficiency_score)) if efficiency_score > 0 else 0

        final_result = {
            "accuracy": metrics["acc"],
            "f1_score": metrics["f1"],
            "test_cases": metrics["samples"],
            "context": str(metrics["context"]) if metrics["context"] else "N/A",
            "base_name": base_model_name,
            "comparison": {
                "labels": ["Accuracy", "Efficiency", "Context"],
                "base_data": [0, 0, 0], # 0 will be treated as 'No Data' in UI
                "merged_data": [
                    metrics["acc"],
                    round(efficiency_score, 2),
                    context_score,
                ],
            },
        }
        update_progress_callback(100, "测试完成")
        _write_meta("success", metrics=final_result)
        
        # 再次检查 metrics 是否全为 0，若是则记录警告
        if final_result["test_cases"] == 0 or final_result["accuracy"] == 0:
            _logger.warning("[run_eval_only_task] 警告: 评估结果为 0 (samples=%s, acc=%s). 请检查数据集或模型输出.", 
                            final_result["test_cases"], final_result["accuracy"])
            # 如果是 run_yaml_eval_stream 返回的 0，可能是读取失败
            if use_yaml:
                 _logger.warning("[run_eval_only_task] 使用 YAML 评估，请检查 YAML 文件路径及 dataset 字段是否正确.")

        testset_id = params.get("testset_id") or None
        if testset_id:
            _update_leaderboard(testset_id, {
                "task_id": task_id,
                "model_name": model_name,
                "accuracy": final_result.get("accuracy"),
                "f1_score": final_result.get("f1_score"),
                "test_cases": final_result.get("test_cases"),
                "created_at": params.get("created_at", time.time()),
                "hf_dataset": hf_dataset,
                "hf_subset": hf_subset,
                # Add extra metrics for baseline usage
                "throughput": metrics.get("throughput", 0),
                "time": metrics.get("time", 0),
                "context": metrics.get("context", 0),
                "efficiency_score": round(efficiency_score, 2)
            })
        return {"status": "success", "metrics": final_result}
    except Exception as e:
        _logger.exception("[eval] 异常: %s", e)
        # Ensure process is killed if running
        if task_control.get("process"):
            try:
                ProcessManager.kill_process_tree(task_control["process"])
            except:
                pass
        _write_meta("error", error=str(e))
        if task_control.get("aborted"):
            return {"status": "stopped", "message": "用户已停止"}
        return {"status": "error", "error": str(e)}
