"""
融合任务执行模块 - 使用 mergenetic + mergekit 执行模型融合，并维护 metadata.json
"""
import json
import logging
import os
import shutil
import time
from pathlib import Path

from config import Config
from core.process_manager import ProcessManager

# 路径与配置
MODEL_POOL_PATH = Config.MODEL_POOL_PATH
MERGE_DIR = Config.MERGE_DIR
LOGS_DIR = Config.LOGS_DIR
RECIPES_DIR = getattr(Config, "RECIPES_DIR", None) or os.path.join(Config.PROJECT_ROOT, "recipes")
MERGENETIC_PYTHON = Config.MERGENETIC_PYTHON

os.makedirs(MERGE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def _popen_group_kwargs():
    return ProcessManager.create_process_group_kwargs()


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
        "status": "pending",
    }
    meta_path = os.path.join(task_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    _logger.info("[run_merge_task] ========== 任务开始 ==========")
    _logger.info("[run_merge_task] 任务ID: %s", task_id)
    _logger.info("[run_merge_task] 日志文件: %s", _get_merge_log_file())
    _logger.info("[run_merge_task] 参数: %s", json.dumps(params, ensure_ascii=False, indent=2))

    def _write_error_status(err_msg: str):
        metadata["status"] = "error"
        metadata["error"] = err_msg
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

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
        from mergenetic.merging.ties_dare_merger import TiesDareMerger

        base_path = model_paths[0]
        other_paths = model_paths[1:]
        # weights: [w1, w2, ...], density 用 limit 如 0.5
        num_models = len(model_paths)
        weights_floats = [float(weights[i]) if i < len(weights) else 1.0 for i in range(num_models)]
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

        # 若输出在 output/task_id 下，将内容移到 output
        inner = os.path.join(output_dir, task_id)
        if os.path.isdir(inner):
            _logger.info("[run_merge_task] 融合输出路径存在: %s", inner)
            _logger.info("[run_merge_task] 模型在预期位置，准备移动到 output_dir: %s", output_dir)
            _logger.info("[run_merge_task] 移动模型内容从 %s 到 %s", inner, output_dir)
            for name in os.listdir(inner):
                src = os.path.join(inner, name)
                dst = os.path.join(output_dir, name)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
            shutil.rmtree(inner)
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
        metadata["metrics"] = {
            "output_path": output_dir,
            "accuracy": None,
            "f1_score": None,
            "test_cases": 0,
            "passed": 0,
            "base_name": models[0] if models else "",
            "comparison": {"labels": [], "base_data": [], "merged_data": []},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        update_progress_callback(100, "任务完成")
        return {"status": "success", "output_path": output_dir, "metrics": metadata.get("metrics", {})}

    except Exception as e:
        _logger.exception("[run_merge_task] 任务失败: %s", e)
        _write_error_status(str(e))
        if task_control.get("aborted"):
            return {"status": "stopped", "message": "用户已停止"}
        return {"status": "error", "error": str(e)}


def run_recipe_apply_task(task_id, params, update_progress_callback, task_control=None):
    """
    按配方执行一次合并（固定 genotype，不进化）。用于「根据配方直接融合出最终模型」。
    params: recipe_id, custom_name（可选）
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
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _write_error_status(err_msg):
        metadata["status"] = "error"
        metadata["error"] = err_msg
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

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
        if os.path.isdir(inner):
            for name in os.listdir(inner):
                src = os.path.join(inner, name)
                dst = os.path.join(output_dir, name)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
            shutil.rmtree(inner)
        elif str(out_path) != output_dir and os.path.isdir(str(out_path)):
            for name in os.listdir(str(out_path)):
                src = os.path.join(str(out_path), name)
                dst = os.path.join(output_dir, name)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    dest = os.path.join(output_dir, name)
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.copytree(src, dest)

        if not os.path.isdir(output_dir) or not any(
            f.endswith(".safetensors") or f == "config.json"
            for f in os.listdir(output_dir)
        ):
            raise RuntimeError("融合输出目录无效: %s" % output_dir)

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
        metadata["metrics"] = {"output_path": output_dir}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        update_progress_callback(100, "配方融合完成")
        return {"status": "success", "output_path": output_dir}
    except Exception as e:
        _logger.exception("[run_recipe_apply_task] 失败: %s", e)
        _write_error_status(str(e))
        return {"status": "error", "error": str(e)}


def run_eval_only_task(task_id, params, update_progress_callback, task_control=None):
    """仅评估任务（与 alpha 兼容）：可调用 lm_eval 或占位返回。"""
    if task_control is None:
        task_control = {}
    task_dir = os.path.join(MERGE_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    metadata = {
        "id": task_id,
        "type": "eval_only",
        "custom_name": params.get("model_name", "Eval Task"),
        "created_at": params.get("created_at", time.time()),
        "dataset": params.get("dataset", "hellaswag"),
        "status": "error",
        "error": "mergeKit_beta 未实现 run_eval_only_task，请使用 mergeKit_alpha 或单独评测脚本",
    }
    meta_path = os.path.join(task_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return {"status": "error", "error": metadata["error"]}
