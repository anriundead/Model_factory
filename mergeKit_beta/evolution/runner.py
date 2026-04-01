#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化融合 Runner：读取 merges/<task_id>/metadata.json，
默认调用仓内 `vendor/vlm_merge/run_vlm_search.py`；若设置 `VLM_SEARCH_DIR` 则改用该目录（便于外置调试或回滚）。
由 Worker 以单次子进程方式启动（默认 `python -m evolution.runner`）。
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = str(_PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config

MERGE_DIR = Config.MERGE_DIR
# 算法目录：非空 VLM_SEARCH_DIR 优先；否则使用仓内 vendor（不依赖外部 modelmerge_visual 树）
_EVOLUTION_DIR = Path(__file__).resolve().parent
_vlm_env = (os.environ.get("VLM_SEARCH_DIR") or "").strip()
VLM_SEARCH_DIR = _vlm_env or str(_EVOLUTION_DIR / "vendor" / "vlm_merge")
RUN_VLM_SEARCH_PY = os.path.join(VLM_SEARCH_DIR, "run_vlm_search.py")
PROMPT_MMLU = "eval/prompt_mmlu.yaml"

# cais/mmlu 的 BuilderConfig 名称与前端/分组中使用的简称不一致时，在此映射为 HF 实际 config 名
MMLU_SUBSET_TO_HF_CONFIG = {
    "high_school_government": "high_school_government_and_politics",
}

def _normalize_mmlu_subsets(hf_dataset: str, hf_subsets: list) -> list:
    """将前端/API 传入的子集名映射为 HuggingFace cais/mmlu 的 config 名，避免 BuilderConfig not found。"""
    if not hf_subsets or "mmlu" not in (hf_dataset or "").lower():
        return hf_subsets or []
    return [MMLU_SUBSET_TO_HF_CONFIG.get(s, s) for s in hf_subsets if s]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evolution.runner")


class TestsetYamlResolver:
    def __init__(self, testset_data_path: str):
        self.testset_data_path = testset_data_path

    def _load_entries(self) -> dict:
        path = self.testset_data_path or ""
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("testsets"), dict):
                return data.get("testsets") or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def resolve(self, hf_dataset, hf_subsets, testset_id):
        entries = self._load_entries()
        if testset_id and entries.get(testset_id):
            yaml_path = (entries.get(testset_id) or {}).get("yaml_template_path", "")
            return yaml_path if yaml_path and os.path.isfile(yaml_path) else ""
        if not hf_dataset:
            return ""
        subsets = [s for s in (hf_subsets or []) if s]
        candidates: list[tuple[int, float, str]] = []
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("hf_dataset") != hf_dataset:
                continue
            yaml_path = entry.get("yaml_template_path", "")
            if not yaml_path or not os.path.isfile(yaml_path):
                continue
            entry_subset = entry.get("hf_subset")
            score = 1
            if subsets and entry_subset in subsets:
                score = 2
            elif not subsets and (entry_subset is None or entry_subset == ""):
                score = 2
            created_at = float(entry.get("created_at", 0) or 0)
            candidates.append((score, created_at, yaml_path))
        if not candidates:
            return ""
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]


def _ensure_metadata_success(meta_path: str, log: logging.Logger, message: str = "任务完成") -> None:
    """将 metadata.json 的 status 置为 success，失败只打日志不抛异常。"""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["status"] = "success"
        meta.pop("error", None)
        meta["message"] = message
        task_id = meta.get("id") or os.path.basename(os.path.dirname(meta_path))
        task_dir = os.path.dirname(meta_path)
        try:
            from merge_manager import _write_metadata
            _write_metadata(task_id, task_dir, meta)
        except Exception:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        log.info("已更新 metadata.json status=success")
    except Exception as e:
        log.warning("更新 metadata.json 失败: %s", e)


def _do_success_path(
    *,
    merge_dir: str,
    final_vlm_output: str,
    meta_path: str,
    progress_path: str,
    task_id: str,
    hf_split_final: str,
    hf_split: str,
    logger: logging.Logger,
) -> None:
    """
    完全融合成功路径：复制 final_vlm 到命名目录、写 fusion_info/README/配方、创建 output 链接、
    清理中间目录，并更新 metadata.json 为 success。任一步骤失败会抛异常，由调用方决定是否仍写 metadata。
    """
    output_dir = os.path.join(merge_dir, "output")
    if not os.path.isdir(final_vlm_output) or not os.listdir(final_vlm_output):
        if not os.path.isdir(final_vlm_output):
            logger.warning("final_vlm_output 目录不存在: %s", final_vlm_output)
        else:
            logger.warning("final_vlm_output 目录为空: %s", final_vlm_output)
            try:
                shutil.rmtree(final_vlm_output, ignore_errors=True)
                logger.info("已清理空的中间模型目录: %s", final_vlm_output)
            except Exception as e:
                logger.warning("清理空目录失败（可忽略）: %s", e)
        _ensure_metadata_success(meta_path, logger)
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model_paths = meta.get("model_paths") or []

    def _safe_name(s):
        return re.sub(r"[^\w\-.]", "_", (os.path.basename(s) if s else "unknown").strip())[:64]

    name1 = _safe_name(model_paths[0]) if model_paths else "base"
    name2 = _safe_name(model_paths[1]) if len(model_paths) > 1 else "other"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    named_dir_name = f"{name1}_{name2}_{ts}"
    named_dir = os.path.join(merge_dir, named_dir_name)
    shutil.copytree(final_vlm_output, named_dir)
    logger.info("已复制最终模型到命名目录 %s", named_dir)

    completed_at = datetime.now().isoformat()
    best_genotype = None
    current_best_acc = None
    try:
        with open(progress_path, "r", encoding="utf-8") as pf:
            prog = json.load(pf)
        best_genotype = prog.get("best_genotype")
        current_best_acc = prog.get("current_best")
    except Exception:
        pass

    final_test_acc = None
    final_test_duration = None
    if hf_split_final and hf_split_final != hf_split:
        acc_path = os.path.join(merge_dir, "final_test_acc.json")
        if os.path.isfile(acc_path):
            try:
                with open(acc_path, "r", encoding="utf-8") as af:
                    acc_data = json.load(af)
                final_test_acc = acc_data.get("final_test_acc") or acc_data.get("accuracy")
                final_test_duration = acc_data.get("duration")
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
        "final_test_duration": final_test_duration,
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
        f"- **最终评测耗时**: {fusion_info.get('final_test_duration')}s" if fusion_info.get('final_test_duration') else "",
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

    recipes_dir = getattr(Config, "RECIPES_DIR", None) or os.path.join(PROJECT_ROOT, "recipes")
    os.makedirs(recipes_dir, exist_ok=True)
    recipe_path = os.path.join(recipes_dir, "%s.json" % task_id)
    with open(recipe_path, "w", encoding="utf-8") as f:
        json.dump(fusion_info, f, ensure_ascii=False, indent=2)
    logger.info("已保存配方到 %s", recipe_path)

    if os.path.lexists(output_dir):
        if os.path.islink(output_dir):
            os.unlink(output_dir)
            logger.info("已移除旧的 output 符号链接")
        elif os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            try:
                os.remove(output_dir)
            except OSError:
                pass
    try:
        os.symlink(named_dir_name, output_dir)
        logger.info("已创建 output -> %s 符号链接", named_dir_name)
    except OSError:
        shutil.copytree(named_dir, output_dir)
        logger.info("已复制命名目录到 output（无符号链接时回退）")

    try:
        shutil.rmtree(final_vlm_output, ignore_errors=True)
        logger.info("已清理中间模型目录: %s", final_vlm_output)
    except Exception as e:
        logger.warning("清理中间模型目录失败（可忽略）: %s", e)

    # 后置检查：若还有 final_vlm 且已有命名目录，再清一次
    if os.path.isdir(final_vlm_output):
        has_named_dir = any(
            os.path.isdir(os.path.join(merge_dir, item)) and not os.path.islink(os.path.join(merge_dir, item))
            for item in os.listdir(merge_dir)
            if item not in ("final_vlm", "output", "metadata.json", "progress.json", "bridge.log") and not item.startswith("_")
            and ("_202" in item or len(item) > 50)
        )
        if has_named_dir:
            try:
                shutil.rmtree(final_vlm_output, ignore_errors=True)
                logger.info("已清理中间模型目录（后置检查）: %s", final_vlm_output)
            except Exception as e:
                logger.warning("后置清理中间模型目录失败（可忽略）: %s", e)

    _ensure_metadata_success(meta_path, logger)


def main():
    ap = argparse.ArgumentParser(description="进化融合 Runner：按 task_id 读取参数并调用 run_vlm_search.py")
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

    hf_dataset = meta.get("hf_dataset") or "cais/mmlu"
    hf_subsets = meta.get("hf_subsets")
    if not hf_subsets or not isinstance(hf_subsets, list):
        hf_subsets = [meta.get("hf_subset") or "college_medicine"]
    # 将前端/API 子集名映射为 HF 实际 config 名（如 high_school_government -> high_school_government_and_politics）
    hf_subsets = _normalize_mmlu_subsets(hf_dataset, hf_subsets)
    hf_split = meta.get("hf_split") or "test"
    # 双 split：进化阶段用 hf_split（建议 val），最终评测用 hf_split_final（建议 test）
    hf_split_final = (meta.get("hf_split_final") or "").strip() or (None if hf_split == "test" else "test")
    pop_size = int(meta.get("pop_size", 20))
    n_iter = int(meta.get("n_iter", 15))
    max_samples = int(meta.get("max_samples", 64))
    dtype = meta.get("dtype") or "bfloat16"
    ray_num_gpus = int(meta.get("ray_num_gpus") or 1)
    eval_mode = meta.get("eval_mode") or "text"
    vlm_path = meta.get("vlm_path") or ""
    testset_id = (meta.get("testset_id") or "").strip()

    logger.info("=" * 80)
    logger.info("进化融合 Runner 启动")
    logger.info("=" * 80)
    logger.info("任务ID: %s", task_id)
    logger.info("模型路径: %s", model_paths)
    logger.info("VLM路径: %s", vlm_path or "(无)")
    logger.info("数据集: %s, 子集: %s, 分割: %s, 最终分割: %s", hf_dataset, hf_subsets, hf_split, hf_split_final or "(同训练)")
    logger.info("参数: pop_size=%s, n_iter=%s, max_samples=%s", pop_size, n_iter, max_samples)
    logger.info("精度: %s, GPU并行数: %s", dtype, ray_num_gpus)
    logger.info("输出目录: %s", final_vlm_output)
    logger.info("进度文件: %s", progress_path)
    logger.info("Runner 日志: %s", bridge_log_path)
    logger.info("-" * 80)

    default_prompt_yaml = "eval/prompt.yaml" if eval_mode == "vlm" else PROMPT_MMLU
    resolver = TestsetYamlResolver(getattr(Config, "TESTSET_DATA_PATH", "") or "")
    resolved_prompt_yaml = resolver.resolve(hf_dataset, hf_subsets, testset_id)
    if resolved_prompt_yaml:
        rlow = resolved_prompt_yaml.lower()
        if not rlow.endswith((".yaml", ".yml")):
            logger.warning("测试集登记的路径非 YAML（忽略，避免误用 README 等）: %s", resolved_prompt_yaml)
            resolved_prompt_yaml = ""
        elif not os.path.isfile(resolved_prompt_yaml):
            logger.warning("测试集 YAML 路径不存在（忽略）: %s", resolved_prompt_yaml)
            resolved_prompt_yaml = ""
    prompt_yaml = resolved_prompt_yaml or default_prompt_yaml
    if resolved_prompt_yaml:
        logger.info("使用测试集 YAML: %s", resolved_prompt_yaml)
    else:
        logger.info("使用默认 YAML: %s", default_prompt_yaml)

    # 数据集验证：在启动 run_vlm_search.py 前验证数据集是否可以成功加载
    logger.info("验证数据集加载...")
    try:
        # 导入数据集加载函数（与 run_vlm_search.py 使用相同的逻辑）
        sys.path.insert(0, VLM_SEARCH_DIR)
        from run_vlm_search import _load_mmlu_samples_one_split
        cache_dir = os.environ.get("HF_DATASETS_CACHE") or (getattr(Config, "HF_DATASETS_CACHE", None) or None)
        hf_subset_group = (meta.get("hf_subset_group") or "").strip()
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
    # 指定结果目录到 merge_dir 下，以便前端读取
    results_dir = os.path.join(merge_dir, "vlm_search_results")
    cmd.extend([
        "--prompt-yaml", prompt_yaml,
        "--pop-size", str(pop_size),
        "--n-iter", str(n_iter),
        "--max-samples", str(max_samples),
        "--dtype", dtype,
        "--device", "cuda",
        "--final-vlm-output", final_vlm_output,
        "--progress-file", progress_path,
        "--results-dir", results_dir,
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
        # 立即写入初始 progress.json，避免前端轮询时文件不存在而显示卡住
        total_steps = max(1, pop_size * n_iter)
        try:
            with open(progress_path, "w", encoding="utf-8") as pf:
                json.dump({
                    "status": "running",
                    "message": "子进程已启动，等待进化搜索首次进度…",
                    "percent": 0,
                    "current": 0,
                    "total": total_steps,
                    "current_step": 0,
                    "total_expected_steps": total_steps,
                    "step": 0,
                }, pf, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("写入初始 progress.json 失败: %s", e)
        out_lines = []
        import csv as csv_mod
        import re
        import time as time_module

        # ── 全局计数器与追踪状态 ──
        eval_count = 0            # 全局评估计数（只增不减，作为 current_step）
        max_acc_seen = 0.0        # 历史最优 acc
        current_gen = 0           # 当前代数
        prev_within_gen_step = 0  # 上一次 [eval] 里的代内 step，用于检测代切换
        gen_start_times = {}
        gen_durations = {}
        eval_times = []           # 最近评估耗时，用于 ETA
        eval_start_time = None    # 当前评估开始时间
        start_time = time_module.time()
        total_expected_steps = max(1, pop_size * n_iter)

        # ── 准备 CSV：每步追加，前端/3D 图可用 ──
        csv_dir = os.path.join(merge_dir, "vlm_search_results")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "vlm_search.csv")
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv_mod.writer(csv_file)
        csv_writer.writerow(["step", "objective_1", "genotype_1", "genotype_2", "generation", "best_acc"])
        csv_file.flush()
        csv_rows_written = 0

        # ── 子进程输出日志 ──
        subprocess_log_path = os.path.join(merge_dir, "subprocess_output.log")
        subprocess_log_file = open(subprocess_log_path, "w", encoding="utf-8")
        logger.info("子进程输出将保存到: %s", subprocess_log_path)

        def _flush_progress(prog_dict):
            """原子写 progress.json（先写临时再 rename）。"""
            tmp = progress_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(prog_dict, f, ensure_ascii=False, indent=2)
            os.replace(tmp, progress_path)

        try:
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line and proc.poll() is not None:
                    break
                if line:
                    out_lines.append(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    subprocess_log_file.write(line)
                    subprocess_log_file.flush()
                    if "[eval]" in line or "[main]" in line or "ERROR" in line.upper():
                        logger.info("子进程: %s", line.rstrip())

                # ── [eval-print] 标记评估开始，用于计算单步耗时 ──
                if "[eval-print]" in line and "step=" in line:
                    eval_start_time = time_module.time()

                # ── [eval] 标记评估完成，是关键数据来源 ──
                eval_match = re.search(
                    r'\[eval\]\s+step=(\d+)\s+acc=([\d.]+)(?:\s+best_acc=([\d.]+))?',
                    line,
                )
                if eval_match:
                    within_gen_step = int(eval_match.group(1))
                    acc_val = float(eval_match.group(2))
                    best_acc_val = float(eval_match.group(3)) if eval_match.group(3) else acc_val

                    # 检测代切换：代内 step 回退到 1 说明新一代开始
                    if within_gen_step <= prev_within_gen_step and eval_count > 0:
                        if current_gen > 0 and current_gen in gen_start_times:
                            gen_durations[current_gen] = time_module.time() - gen_start_times[current_gen]
                            logger.info("Generation %d 结束 (耗时: %.2fs)", current_gen, gen_durations[current_gen])
                        current_gen += 1
                        gen_start_times[current_gen] = time_module.time()
                        logger.info("Generation %d 开始", current_gen)
                    elif eval_count == 0:
                        current_gen = 1
                        gen_start_times[1] = time_module.time()
                        logger.info("Generation 1 开始")
                    prev_within_gen_step = within_gen_step

                    eval_count += 1
                    max_acc_seen = max(max_acc_seen, acc_val, best_acc_val)

                    # 单步耗时
                    if eval_start_time is not None:
                        eval_times.append(time_module.time() - eval_start_time)
                        eval_start_time = None

                    # 解析 genotype
                    geno_match = re.search(r'genotype=\[([\d.,\s\-eE+]+)\]', line)
                    genotype = []
                    if geno_match:
                        try:
                            genotype = [float(x.strip()) for x in geno_match.group(1).split(",")]
                        except (ValueError, TypeError):
                            pass

                    # ── 追加 CSV 行 ──
                    g1 = genotype[0] if len(genotype) > 0 else 0.0
                    g2 = genotype[1] if len(genotype) > 1 else 0.0
                    csv_writer.writerow([eval_count, acc_val, g1, g2, current_gen, max_acc_seen])
                    csv_file.flush()
                    csv_rows_written += 1

                    # ── 更新 progress.json ──
                    # 说明：外部搜索在某些配置下实际评估次数可能超过 pop_size*n_iter（预估值）。
                    # 为避免前端出现“32/18”这类反直觉展示，这里使用 display_total 做显示总步数。
                    display_total_steps = max(total_expected_steps, eval_count)
                    percent = (
                        min(99, max(0, round(100 * eval_count / display_total_steps)))
                        if display_total_steps else 0
                    )
                    step_message = "Step %d/%d (Gen %d, acc=%.4f, best=%.4f)" % (
                        eval_count, display_total_steps, current_gen, acc_val, max_acc_seen)

                    try:
                        with open(progress_path, "r", encoding="utf-8") as pf:
                            prog = json.load(pf)
                    except Exception:
                        prog = {}
                    prog["step"] = eval_count
                    prog["current_step"] = eval_count
                    prog["current"] = eval_count
                    prog["total"] = display_total_steps
                    prog["total_expected_steps"] = display_total_steps
                    prog["percent"] = percent
                    prog["message"] = step_message
                    prog["status"] = "running"
                    prog["current_best"] = max_acc_seen
                    prog["global_best"] = max_acc_seen
                    if genotype:
                        prog["best_genotype"] = genotype
                    if gen_durations:
                        prog["gen_durations"] = gen_durations
                    if eval_times:
                        recent = eval_times[-min(5, len(eval_times)):]
                        avg_eval = sum(recent) / len(recent)
                        prog["avg_step_time"] = avg_eval
                        remaining = max(0, display_total_steps - eval_count)
                        prog["eta_seconds"] = avg_eval * remaining
                        prog["estimated_completion"] = time_module.time() + prog["eta_seconds"]
                    if 1 in gen_durations:
                        prog["first_gen_time"] = gen_durations[1]
                    try:
                        _flush_progress(prog)
                    except Exception:
                        pass

                # ── 外部 ETA（如果 run_vlm_search 输出） ──
                if eval_count == 0:
                    eta_match = re.search(r'eta[=:]?\s*([\d.]+)\s*(s|sec|second)', line, re.IGNORECASE)
                    if eta_match:
                        try:
                            with open(progress_path, "r", encoding="utf-8") as pf:
                                prog = json.load(pf)
                            prog["eta_seconds"] = float(eta_match.group(1))
                            prog["estimated_completion"] = time_module.time() + prog["eta_seconds"]
                            _flush_progress(prog)
                        except Exception:
                            pass

        finally:
            csv_file.close()
            subprocess_log_file.close()
            logger.info("子进程输出已保存到: %s", subprocess_log_path)
            logger.info("CSV 已写入 %d 行到 %s", csv_rows_written, csv_path)

        proc.wait()

        # 最终一代结束
        if current_gen > 0 and current_gen in gen_start_times and current_gen not in gen_durations:
            gen_durations[current_gen] = time_module.time() - gen_start_times[current_gen]

        # 任务结束，强制更新最后一次进度
        try:
            if proc.returncode == 0:
                with open(progress_path, "r", encoding="utf-8") as pf:
                    prog = json.load(pf)
                final_step = max(eval_count, prog.get("current_step", 0))
                if total_expected_steps > 0 and final_step < total_expected_steps:
                    final_step = total_expected_steps
                prog["step"] = final_step
                prog["current_step"] = final_step
                prog["current"] = final_step
                prog["percent"] = 100
                prog["eta_seconds"] = 0
                prog["estimated_completion"] = time_module.time()
                prog["total_task_time"] = time_module.time() - start_time
                prog["current_best"] = max(max_acc_seen, prog.get("current_best", 0) or 0)
                prog["global_best"] = prog["current_best"]
                if gen_durations:
                    prog["gen_durations"] = gen_durations
                _flush_progress(prog)
        except Exception as e:
            logger.warning("更新最终进度失败: %s", e)

        if out_lines:
            logger.info("子进程输出统计: 总行数=%d, [eval]行数=%d, CSV行数=%d",
                         len(out_lines), eval_count, csv_rows_written)
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

        # 外部最终评测 (可选)：失败不阻断主流程，仅记录日志
        if proc.returncode == 0 and hf_split_final and hf_split_final != hf_split:
            logger.info("执行外部最终评测 (eval_final.py)...")
            eval_cmd = [
                python_cmd,
                os.path.join(VLM_SEARCH_DIR, "eval_final.py"),
                "--model-path", final_vlm_output,
                "--hf-dataset", hf_dataset,
                "--hf-split", hf_split_final,
                "--max-samples", str(max_samples),
                "--output-file", final_acc_file,
                "--prompt-yaml", prompt_yaml
            ]
            if hf_subsets:
                eval_cmd.extend(["--hf-subset", *hf_subsets])
            if hf_subset_group:
                eval_cmd.extend(["--hf-subset-group", hf_subset_group])
            try:
                logger.info("Final Eval Command: %s", " ".join(eval_cmd))
                result = subprocess.run(eval_cmd, cwd=VLM_SEARCH_DIR, env=env, timeout=3600)
                if result.returncode != 0:
                    logger.warning("最终评测退出码非 0: %s，继续完成收尾", result.returncode)
            except subprocess.TimeoutExpired as e:
                logger.warning("最终评测超时: %s，继续完成收尾", e)
            except Exception as e:
                logger.warning("最终评测异常（不阻断主流程）: %s", e)

        # 完全融合成功：复制到命名目录、写 fusion_info、更新 metadata；任一步骤失败仍尽量将任务标为成功
        try:
            _do_success_path(
                merge_dir=merge_dir,
                final_vlm_output=final_vlm_output,
                meta_path=meta_path,
                progress_path=progress_path,
                task_id=task_id,
                hf_split_final=hf_split_final,
                hf_split=hf_split,
                logger=logger,
            )
        except Exception as e:
            logger.exception("收尾步骤异常: %s", e)
            _ensure_metadata_success(meta_path, logger, message="任务完成（收尾步骤部分失败，模型在 final_vlm）")

        logger.info("=" * 80)
        logger.info("进化 Runner 结束（成功）")
        logger.info("=" * 80)
    except Exception as e:
        logger.exception("进化 Runner 异常: %s", e)
        try:
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({"status": "error", "message": str(e)}, f, ensure_ascii=False)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
