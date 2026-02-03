#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化融合桥接脚本：读取 mergeKit_beta merges/<task_id>/metadata.json，
调用 VLM_merge_total/VLM_merge/run_vlm_search.py 执行进化搜索。
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 确保 mergeKit_beta 在 path 中
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config

MERGE_DIR = Config.MERGE_DIR
# run_vlm_search.py 所在目录（可环境变量覆盖）
VLM_SEARCH_DIR = os.environ.get(
    "VLM_SEARCH_DIR",
    str(Path(PROJECT_ROOT).parent / "modelmerge_visual" / "VLM_merge_total" / "VLM_merge"),
)
RUN_VLM_SEARCH_PY = os.path.join(VLM_SEARCH_DIR, "run_vlm_search.py")
PROMPT_MMLU = "eval/prompt_mmlu.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("桥接脚本")


def main():
    ap = argparse.ArgumentParser(description="进化融合桥接：按 task_id 读取参数并调用 run_vlm_search.py")
    ap.add_argument("--task-id", required=True, help="任务 ID，对应 merges/<task_id>")
    args = ap.parse_args()
    task_id = args.task_id
    merge_dir = os.path.join(MERGE_DIR, task_id)
    meta_path = os.path.join(merge_dir, "metadata.json")
    progress_path = os.path.join(merge_dir, "progress.json")
    bridge_log_path = os.path.join(merge_dir, "bridge.log")
    final_vlm_output = os.path.join(merge_dir, "final_vlm")

    os.makedirs(merge_dir, exist_ok=True)
    fh = logging.FileHandler(bridge_log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    if not os.path.isfile(meta_path):
        logger.error("metadata.json 不存在: %s", meta_path)
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_paths = meta.get("model_paths") or []
    if len(model_paths) < 2:
        logger.error("至少需要 2 个模型路径")
        sys.exit(1)

    hf_dataset = meta.get("hf_dataset", "cais/mmlu")
    hf_subsets = meta.get("hf_subsets")
    if not hf_subsets or not isinstance(hf_subsets, list):
        hf_subsets = [meta.get("hf_subset", "college_medicine")]
    hf_split = meta.get("hf_split", "test")
    pop_size = int(meta.get("pop_size", 20))
    n_iter = int(meta.get("n_iter", 15))
    max_samples = int(meta.get("max_samples", 64))
    dtype = meta.get("dtype", "bfloat16")
    ray_num_gpus = int(meta.get("ray_num_gpus", 1))
    eval_mode = meta.get("eval_mode", "text")
    vlm_path = meta.get("vlm_path", "")

    logger.info("=" * 80)
    logger.info("进化融合桥接脚本启动")
    logger.info("=" * 80)
    logger.info("任务ID: %s", task_id)
    logger.info("模型路径: %s", model_paths)
    logger.info("VLM路径: %s", vlm_path or "(无)")
    logger.info("数据集: %s, 子集: %s, 分割: %s", hf_dataset, hf_subsets, hf_split)
    logger.info("参数: pop_size=%s, n_iter=%s, max_samples=%s", pop_size, n_iter, max_samples)
    logger.info("精度: %s, GPU并行数: %s", dtype, ray_num_gpus)
    logger.info("输出目录: %s", final_vlm_output)
    logger.info("进度文件: %s", progress_path)
    logger.info("桥接日志: %s", bridge_log_path)
    logger.info("-" * 80)

    if not os.path.isfile(RUN_VLM_SEARCH_PY):
        logger.error("run_vlm_search.py 不存在: %s (可设置 VLM_SEARCH_DIR)", RUN_VLM_SEARCH_PY)
        sys.exit(1)

    python_cmd = (getattr(Config, "MERGENETIC_PYTHON", None) or "").strip() or "python"
    cmd = [
        python_cmd,
        RUN_VLM_SEARCH_PY,
        "--ray-num-gpus", str(ray_num_gpus),
        "--model-paths", *model_paths,
        "--eval-mode", eval_mode,
        "--hf-dataset", hf_dataset,
        "--hf-split", hf_split,
    ]
    # 多个子集用单次 --hf-subset A B 传递，避免 argparse nargs="+" 被多次覆盖
    if hf_subsets:
        cmd.extend(["--hf-subset", *hf_subsets])
    hf_subset_group = meta.get("hf_subset_group", "").strip()
    if hf_subset_group:
        cmd.extend(["--hf-subset-group", hf_subset_group])
    cmd.extend([
        "--prompt-yaml", PROMPT_MMLU,
        "--pop-size", str(pop_size),
        "--n-iter", str(n_iter),
        "--max-samples", str(max_samples),
        "--dtype", dtype,
        "--device", "cuda",
        "--final-vlm-output", final_vlm_output,
        "--progress-file", progress_path,
    ])
    if vlm_path:
        cmd.extend(["--vlm-path", vlm_path])

    logger.info("启动 run_vlm_search.py 子进程...")
    logger.debug("命令: %s", " ".join(cmd))
    logger.info("工作目录: %s", VLM_SEARCH_DIR)

    # 子进程继承当前环境（含 HF_ENDPOINT 镜像、HF_DATASETS_CACHE），便于使用已下载数据集
    env = os.environ.copy()
    if getattr(Config, "HF_ENDPOINT", None):
        env["HF_ENDPOINT"] = Config.HF_ENDPOINT
    if getattr(Config, "HF_DATASETS_CACHE", None):
        env["HF_DATASETS_CACHE"] = Config.HF_DATASETS_CACHE
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=VLM_SEARCH_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        logger.info("子进程已启动，PID: %s", proc.pid)
        out_lines = []
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line and proc.poll() is not None:
                break
            if line:
                out_lines.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait()
        if proc.returncode != 0:
            tail = "".join(out_lines[-50:]) if len(out_lines) > 50 else "".join(out_lines)
            logger.warning("标准输出（最后500字符）: %s", tail[-500:])
            try:
                with open(progress_path, "w", encoding="utf-8") as f:
                    json.dump({"status": "error", "message": tail[-500:], "returncode": proc.returncode}, f, ensure_ascii=False)
            except Exception:
                pass
            logger.error("run_vlm_search 执行失败（返回码: %s）", proc.returncode)
            sys.exit(1)

        # 完全融合成功：将 final_vlm 同步到 merges/<task_id>/output，并更新 metadata.json
        output_dir = os.path.join(merge_dir, "output")
        if os.path.isdir(final_vlm_output) and os.listdir(final_vlm_output):
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
            shutil.copytree(final_vlm_output, output_dir)
            logger.info("已复制最终模型到 %s", output_dir)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["status"] = "success"
            meta.pop("error", None)
            meta["message"] = "任务完成"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info("已更新 metadata.json status=success")
        except Exception as e:
            logger.warning("更新 metadata.json 失败: %s", e)

        logger.info("=" * 80)
        logger.info("桥接脚本结束（成功）")
        logger.info("=" * 80)
    except Exception as e:
        logger.exception("桥接异常: %s", e)
        try:
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({"status": "error", "message": str(e)}, f, ensure_ascii=False)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
