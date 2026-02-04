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

    # 数据集验证：在启动 run_vlm_search.py 前验证数据集是否可以成功加载
    logger.info("验证数据集加载...")
    try:
        # 导入数据集加载函数（与 run_vlm_search.py 使用相同的逻辑）
        sys.path.insert(0, VLM_SEARCH_DIR)
        from run_vlm_search import _load_mmlu_samples_one_split
        cache_dir = os.environ.get("HF_DATASETS_CACHE") or (getattr(Config, "HF_DATASETS_CACHE", None) or None)
        hf_subset_group = meta.get("hf_subset_group", "").strip()
        test_samples = _load_mmlu_samples_one_split(
            hf_dataset,
            hf_subsets,
            hf_split,
            max_samples=min(4, max_samples) if max_samples else 4,  # 只加载少量样本用于验证
            seed=42,
            cache_dir=cache_dir,
            hf_subset_group=hf_subset_group,
        )
        if not test_samples or len(test_samples) == 0:
            logger.error("数据集验证失败: 无法加载任何样本 (数据集=%s, 子集=%s, 分割=%s)", hf_dataset, hf_subsets, hf_split)
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({"status": "error", "message": f"数据集验证失败: 无法加载样本 (数据集={hf_dataset}, 子集={hf_subsets}, 分割={hf_split})"}, f, ensure_ascii=False)
            sys.exit(1)
        logger.info("数据集验证成功: 成功加载 %d 个样本 (验证用样本数)", len(test_samples))
        # 验证样本结构
        sample_keys = set(test_samples[0].keys()) if test_samples else set()
        required_keys = {"question", "choices", "answer"}
        missing_keys = required_keys - sample_keys
        if missing_keys:
            logger.warning("数据集样本缺少部分字段: %s (现有字段: %s)", missing_keys, sample_keys)
        else:
            logger.info("数据集样本结构验证通过 (包含必要字段: question, choices, answer)")
    except ImportError as e:
        logger.warning("无法导入数据集加载函数进行验证: %s (将跳过验证)", e)
    except Exception as e:
        logger.error("数据集验证过程出错: %s", e)
        logger.exception("数据集验证异常详情:")
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"status": "error", "message": f"数据集验证失败: {str(e)}"}, f, ensure_ascii=False)
        sys.exit(1)

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
        # 用于解析进度信息和计算 ETA
        import re
        import time as time_module
        eval_times = []  # 记录每次评估的耗时
        last_step = 0
        start_time = time_module.time()
        
        # 创建子进程输出日志文件
        subprocess_log_path = os.path.join(merge_dir, "subprocess_output.log")
        subprocess_log_file = open(subprocess_log_path, "w", encoding="utf-8")
        logger.info("子进程输出将保存到: %s", subprocess_log_path)
        
        try:
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line and proc.poll() is not None:
                    break
                if line:
                    out_lines.append(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    # 写入子进程输出日志文件
                    subprocess_log_file.write(line)
                    subprocess_log_file.flush()
                    
                    # 同时写入 bridge.log（INFO 级别的重要信息）
                    if "[eval]" in line or "[main]" in line or "step=" in line or "ERROR" in line.upper() or "WARNING" in line.upper():
                        logger.info("子进程: %s", line.rstrip())
                
                # 解析评估日志：提取 step 和耗时信息
                # 格式示例: "[eval] step=1 acc=0.1250 best_acc=0.1250 genotype=[...]"
                # 或: "[eval-print] step=1 pid=12345 cuda_visible=0,1 genotype=[...]"
                eval_match = re.search(r'\[eval\]\s+step=(\d+).*?acc=([\d.]+)', line)
                if eval_match:
                    current_step = int(eval_match.group(1))
                    if current_step > last_step:
                        # 计算本次评估的耗时（基于时间差）
                        current_time = time_module.time()
                        if last_step > 0:
                            elapsed = current_time - start_time
                            avg_time_per_step = elapsed / current_step if current_step > 0 else 0
                            eval_times.append(avg_time_per_step)
                        last_step = current_step
                
                # 尝试解析 ETA 信息（如果 run_vlm_search.py 输出）
                eta_match = re.search(r'eta[=:]?\s*([\d.]+)\s*(s|sec|second)', line, re.IGNORECASE)
                if eta_match:
                    eta_seconds = float(eta_match.group(1))
                    # 更新 progress.json 中的 ETA
                    try:
                        with open(progress_path, "r", encoding="utf-8") as pf:
                            prog = json.load(pf)
                        prog["eta_seconds"] = eta_seconds
                        prog["estimated_completion"] = time_module.time() + eta_seconds
                        with open(progress_path, "w", encoding="utf-8") as pf:
                            json.dump(prog, pf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                
                # 基于评估次数和平均耗时计算 ETA（如果没有直接输出）
                if last_step > 0 and len(eval_times) > 0:
                    avg_eval_time = sum(eval_times[-min(5, len(eval_times)):]) / min(5, len(eval_times))  # 使用最近5次的平均
                    total_expected_steps = pop_size * n_iter  # 估算总步数
                    remaining_steps = max(0, total_expected_steps - last_step)
                    estimated_eta = avg_eval_time * remaining_steps
                    
                    # 每10步更新一次 progress.json 中的 ETA
                    if last_step % 10 == 0 or last_step == 1:
                        try:
                            with open(progress_path, "r", encoding="utf-8") as pf:
                                prog = json.load(pf)
                            prog["eta_seconds"] = estimated_eta
                            prog["estimated_completion"] = time_module.time() + estimated_eta
                            prog["current_step"] = last_step
                            prog["total_expected_steps"] = total_expected_steps
                            with open(progress_path, "w", encoding="utf-8") as pf:
                                json.dump(prog, pf, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
        finally:
            subprocess_log_file.close()
            logger.info("子进程输出已保存到: %s", subprocess_log_path)
        
        proc.wait()
        
        # 任务完成后，记录子进程输出的统计信息
        if out_lines:
            eval_count = len([l for l in out_lines if "[eval]" in l or "step=" in l])
            logger.info("子进程输出统计: 总行数=%d, 评估日志行数=%d", len(out_lines), eval_count)
            # 记录最后100行输出（便于排查问题）
            tail_lines = out_lines[-100:] if len(out_lines) > 100 else out_lines
            logger.info("子进程输出（最后100行）:\n%s", "".join(tail_lines))
        
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

            # 清理中间模型目录：final_vlm 已复制到命名目录，可以删除
            if os.path.isdir(final_vlm_output):
                try:
                    shutil.rmtree(final_vlm_output, ignore_errors=True)
                    logger.info("已清理中间模型目录: %s", final_vlm_output)
                except Exception as e:
                    logger.warning("清理中间模型目录失败（可忽略）: %s", e)
        else:
            # 如果 final_vlm_output 不存在或为空，记录日志
            if not os.path.isdir(final_vlm_output):
                logger.warning("final_vlm_output 目录不存在: %s", final_vlm_output)
            elif not os.listdir(final_vlm_output):
                logger.warning("final_vlm_output 目录为空: %s", final_vlm_output)
                # 即使为空也尝试清理
                try:
                    shutil.rmtree(final_vlm_output, ignore_errors=True)
                    logger.info("已清理空的中间模型目录: %s", final_vlm_output)
                except Exception as e:
                    logger.warning("清理空目录失败（可忽略）: %s", e)

        # 无论是否进入上面的 if 块，都尝试清理 final_vlm（如果存在且有命名目录）
        if os.path.isdir(final_vlm_output):
            # 检查是否有命名目录（表示任务已完成）
            has_named_dir = False
            for item in os.listdir(merge_dir):
                if item.startswith("_") or item in ["final_vlm", "output", "metadata.json", "progress.json", "bridge.log"]:
                    continue
                item_path = os.path.join(merge_dir, item)
                if os.path.isdir(item_path) and not os.path.islink(item_path):
                    # 检查是否是命名目录（包含时间戳格式或长度较长）
                    if "_202" in item or len(item) > 50:
                        has_named_dir = True
                        break
            if has_named_dir:
                try:
                    shutil.rmtree(final_vlm_output, ignore_errors=True)
                    logger.info("已清理中间模型目录（后置检查）: %s", final_vlm_output)
                except Exception as e:
                    logger.warning("后置清理中间模型目录失败（可忽略）: %s", e)

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
