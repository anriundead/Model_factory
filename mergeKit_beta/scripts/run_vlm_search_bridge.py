#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化融合桥接脚本：读取 mergeKit_beta merges/<task_id>/metadata.json，
调用 VLM_merge_total/VLM_merge/run_vlm_search.py 执行进化搜索。
成功后将输出命名为「父模型1_父模型2_结束时间戳」，并写入 fusion_info.json 与 README.md。
"""
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
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
    # 双 split：进化阶段用 hf_split（建议 val），最终评测用 hf_split_final（建议 test）
    hf_split_final = meta.get("hf_split_final", "").strip() or (None if hf_split == "test" else "test")
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
    logger.info("数据集: %s, 子集: %s, 分割: %s, 最终分割: %s", hf_dataset, hf_subsets, hf_split, hf_split_final or "(同训练)")
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
    # 双 split：若指定了最终评测 split，传参给 run_vlm_search（脚本需支持 --hf-split-final 与 --final-acc-file）
    final_acc_file = os.path.join(merge_dir, "final_test_acc.json")
    if hf_split_final and hf_split_final != hf_split:
        cmd.extend(["--hf-split-final", hf_split_final, "--final-acc-file", final_acc_file])
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

        # 完全融合成功：输出命名为「父模型1_父模型2_结束时间戳」，写入详细信息，并保留 output 以兼容
        output_dir = os.path.join(merge_dir, "output")
        if os.path.isdir(final_vlm_output) and os.listdir(final_vlm_output):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_paths = meta.get("model_paths") or []
            # 父辈模型名（目录名，做文件名安全处理）
            def _safe_name(s):
                return re.sub(r"[^\w\-.]", "_", (os.path.basename(s) if s else "unknown").strip())[:64]
            name1 = _safe_name(model_paths[0]) if model_paths else "base"
            name2 = _safe_name(model_paths[1]) if len(model_paths) > 1 else "other"
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            named_dir_name = f"{name1}_{name2}_{ts}"
            named_dir = os.path.join(merge_dir, named_dir_name)
            shutil.copytree(final_vlm_output, named_dir)
            logger.info("已复制最终模型到命名目录 %s", named_dir)

            # 融合详细信息（供下游与人工查阅）
            completed_at = datetime.now().isoformat()
            # 从 progress.json 读取进化得到的最优 genotype（完整配方）
            best_genotype = None
            current_best_acc = None
            try:
                with open(progress_path, "r", encoding="utf-8") as pf:
                    prog = json.load(pf)
                best_genotype = prog.get("best_genotype")
                current_best_acc = prog.get("current_best")
            except Exception:
                pass
            # 双 split：若 run_vlm_search 写入了最终 test 准确率，读入并写入配方
            final_test_acc = None
            if hf_split_final and hf_split_final != hf_split:
                acc_path = os.path.join(merge_dir, "final_test_acc.json")
                if os.path.isfile(acc_path):
                    try:
                        with open(acc_path, "r", encoding="utf-8") as af:
                            acc_data = json.load(af)
                        final_test_acc = acc_data.get("final_test_acc") or acc_data.get("accuracy")
                    except Exception:
                        pass
            fusion_info = {
                "task_id": task_id,
                "output_dir_name": named_dir_name,
                "custom_name": meta.get("custom_name", ""),
                "model_paths": model_paths,
                "parent_names": [name1, name2],
                "best_genotype": best_genotype,
                "hf_dataset": meta.get("hf_dataset", ""),
                "hf_subsets": meta.get("hf_subsets", []),
                "hf_subset_group": meta.get("hf_subset_group", ""),
                "hf_split": meta.get("hf_split", ""),
                "hf_split_final": meta.get("hf_split_final", "") or (hf_split_final or ""),
                "hf_split_train": meta.get("hf_split_train", ""),
                "hf_split_test": meta.get("hf_split_test", ""),
                "pop_size": meta.get("pop_size"),
                "n_iter": meta.get("n_iter"),
                "max_samples": meta.get("max_samples"),
                "dtype": meta.get("dtype", ""),
                "ray_num_gpus": meta.get("ray_num_gpus"),
                "eval_mode": meta.get("eval_mode", ""),
                "completed_at": completed_at,
                "current_best_acc": current_best_acc,
                "final_test_acc": final_test_acc,
            }
            with open(os.path.join(named_dir, "fusion_info.json"), "w", encoding="utf-8") as f:
                json.dump(fusion_info, f, ensure_ascii=False, indent=2)
            readme_lines = [
                "# 融合输出",
                "",
                f"- **任务 ID**: {task_id}",
                f"- **自定义名称**: {meta.get('custom_name', '')}",
                f"- **父模型**: {name1}, {name2}",
                f"- **完成时间**: {completed_at}",
                "",
                "## 配方（可复现）",
                f"- **best_genotype**: {best_genotype}",
                f"- **验证准确率 (current_best)**: {current_best_acc}",
                f"- **最终 test 准确率 (final_test_acc)**: {fusion_info.get('final_test_acc')}",
                "",
                "## 参数",
                f"- 数据集: {meta.get('hf_dataset', '')}",
                f"- 子集: {meta.get('hf_subsets', [])}",
                f"- 种群大小: {meta.get('pop_size')}, 迭代: {meta.get('n_iter')}, 最大样本: {meta.get('max_samples')}",
                f"- 精度: {meta.get('dtype', '')}, Ray GPU: {meta.get('ray_num_gpus')}",
                "",
                "详见 fusion_info.json。",
            ]
            with open(os.path.join(named_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write("\n".join(readme_lines))

            # 保存到配方目录，供「根据配方直接融合」与前端查询
            recipes_dir = getattr(Config, "RECIPES_DIR", None) or os.path.join(PROJECT_ROOT, "recipes")
            os.makedirs(recipes_dir, exist_ok=True)
            recipe_path = os.path.join(recipes_dir, "%s.json" % task_id)
            with open(recipe_path, "w", encoding="utf-8") as f:
                json.dump(fusion_info, f, ensure_ascii=False, indent=2)
            logger.info("已保存配方到 %s", recipe_path)

            # 兼容：output 指向命名目录（符号链接或拷贝）
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
            try:
                os.symlink(named_dir_name, output_dir)
                logger.info("已创建 output -> %s 符号链接", named_dir_name)
            except OSError:
                shutil.copytree(named_dir, output_dir)
                logger.info("已复制命名目录到 output（无符号链接时回退）")

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
