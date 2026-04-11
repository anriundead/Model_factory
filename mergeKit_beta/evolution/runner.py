#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化融合 Runner：读取 merges/<task_id>/metadata.json，
默认调用仓内 `vendor/vlm_merge/run_vlm_search.py`；若设置 `VLM_SEARCH_DIR` 则改用该目录（便于外置调试或回滚）。
由 Worker 以单次子进程方式启动（默认 `python -m evolution.runner`）。
成功后将输出命名为「父模型1_父模型2_结束时间戳」，并写入 fusion_info.json 与 README.md。
"""
import argparse
import csv as csv_std
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

def _lock_file(path: str, timeout_s: int, log: logging.Logger):
    """
    任务级 GPU 锁：避免同一容器内多个 GPU 重任务并发互相挤占。
    只约束本容器内任务；无法约束宿主机常驻进程（如 ollama），因此仍需显存阈值筛选。
    """
    import time
    import fcntl

    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    f = open(path, "w", encoding="utf-8")
    deadline = time.time() + max(0, int(timeout_s))
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            f.write(str(os.getpid()))
            f.flush()
            log.debug("GPU 锁持有 PID=%s path=%s", os.getpid(), path)
            return f
        except BlockingIOError:
            if time.time() >= deadline:
                f.close()
                raise TimeoutError(f"GPU 锁等待超时: {path}")
            time.sleep(0.5)


def _pick_tp2_pairs_or_fallback(log: logging.Logger) -> tuple[list[tuple[int, int]], str]:
    """
    选择可用的 NVLink 卡对；若不足两对，则降级为 1 对（TP=2 单 actor）；若都不可用则返回空。
    """
    from core.gpu_topology import env_int, parse_nvlink_pairs, select_free_pairs

    topo = (os.environ.get("MERGEKIT_EVOLUTION_TOPOLOGY") or os.environ.get("MERGEKIT_TOPOLOGY") or "01,23").strip()
    min_free_gb = env_int("MERGEKIT_EVOLUTION_MIN_FREE_GB", 12)
    min_free_mib = int(min_free_gb * 1024)
    try:
        pairs = parse_nvlink_pairs(topo)
    except Exception as e:
        log.warning("拓扑配置无效（%s），回退默认 01,23: %s", topo, e)
        pairs = [(0, 1), (2, 3)]

    ok = select_free_pairs(pairs=pairs, min_free_mib=min_free_mib)
    if len(ok) >= 2:
        return ok[:2], f"选用 2 对卡对（TP=2 x2 actor），min_free={min_free_gb}GiB topo={topo}"
    if len(ok) == 1:
        return ok, f"仅 1 对卡对满足空闲显存阈值，降级为 TP=2 x1 actor，min_free={min_free_gb}GiB topo={topo}"
    return [], f"无卡对满足空闲显存阈值，min_free={min_free_gb}GiB topo={topo}"


def _write_metadata_safe(task_id: str, task_dir: str, meta: dict, log: logging.Logger) -> None:
    """尽量用 merge_manager._write_metadata 写回；失败则直接覆盖 metadata.json。"""
    meta_path = os.path.join(task_dir, "metadata.json")
    try:
        from merge_manager import _write_metadata  # type: ignore

        _write_metadata(task_id, task_dir, meta)
        return
    except Exception as e:
        log.debug("merge_manager._write_metadata 不可用，改用直接写文件: %s", e)
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("写回 metadata.json 失败（可忽略，但将影响可复现性）: %s", e)


def _parse_cuda_visible_devices() -> list[int] | None:
    v = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not v:
        return None
    out: list[int] = []
    for tok in v.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except ValueError:
            return None
    return out or None


def _decide_tp_and_batch(
    *,
    meta: dict,
    log: logging.Logger,
) -> tuple[int, list[tuple[int, int]], int, dict]:
    """
    决策 TP（1/2）与初始 batch_size。

    约定：MERGEKIT_EVOLUTION_TP2=auto|1|0
    - auto（默认）：基于显存余量选择 TP=1（优先并行）；不满足则尝试 TP=2；仍不满足则 TP=1 + 降 batch。
    - 1：强制 TP=2
    - 0：强制 TP=1

    返回：(tp_size, chosen_pairs, batch_size_initial, decision_meta_patch)
    """
    from core.gpu_topology import env_int, query_gpus

    # 阈值：优先 MERGEKIT_EVOLUTION_MIN_FREE_GB，其次 MERGEKIT_EVAL_MIN_FREE_GB，最后默认 12GiB
    min_free_gb = env_int("MERGEKIT_EVOLUTION_MIN_FREE_GB", env_int("MERGEKIT_EVAL_MIN_FREE_GB", 12))
    vis = _parse_cuda_visible_devices()
    gpus = query_gpus()
    if vis is not None:
        gpus = [g for g in gpus if g.index in set(vis)]
    free_gib = [round(g.mem_free_mib / 1024.0, 2) for g in gpus]
    has_single_ok = any(g.mem_free_mib >= int(min_free_gb * 1024) for g in gpus)

    tp2_switch = (os.environ.get("MERGEKIT_EVOLUTION_TP2") or "auto").strip().lower()
    if tp2_switch in ("1", "true", "yes", "on"):
        tp2_mode = "forced_on"
    elif tp2_switch in ("0", "false", "no", "off"):
        tp2_mode = "forced_off"
    else:
        tp2_mode = "auto"

    # 默认 batch（允许 metadata 里指定；否则与 run_vlm_search 当前默认保持一致）
    batch_size = int(meta.get("batch_size") or 4)
    batch_size = max(1, batch_size)

    chosen_pairs: list[tuple[int, int]] = []
    tp_size = 1
    reason = "single_gpu_has_enough_vram" if has_single_ok else "low_free_vram"

    if tp2_mode == "forced_on":
        chosen_pairs, _ = _pick_tp2_pairs_or_fallback(log)
        tp_size = 2 if chosen_pairs else 1
        reason = "forced_tp2" if chosen_pairs else "forced_tp2_but_no_pairs"
    elif tp2_mode == "forced_off":
        tp_size = 1
        chosen_pairs = []
        reason = "forced_tp1"
        if not has_single_ok:
            # 按计划：单卡显存不足时，仅降 batch 继续任务
            batch_size = min(batch_size, 2)
            reason = "batch_reduced_due_to_low_free"
    else:
        # auto：优先 TP=1（最大化并行）；显存不足再尝试 TP=2
        if has_single_ok:
            tp_size = 1
            chosen_pairs = []
            reason = "single_gpu_has_enough_vram"
        else:
            chosen_pairs, _ = _pick_tp2_pairs_or_fallback(log)
            if chosen_pairs:
                tp_size = 2
                reason = "fallback_to_tp2_due_to_low_free"
            else:
                tp_size = 1
                chosen_pairs = []
                batch_size = min(batch_size, 2)
                reason = "batch_reduced_due_to_low_free"

    patch = {
        "evolution_tp2_mode": tp2_mode,
        "chosen_tp_size": int(tp_size),
        "tp2_pairs": ";".join(f"{a},{b}" for a, b in chosen_pairs) if chosen_pairs else "",
        "gpu_free_gib_snapshot": free_gib,
        "tp_decision_reason": reason,
        "eval_batch_size_initial": int(batch_size),
    }
    return tp_size, chosen_pairs, batch_size, patch


def _cap_ray_num_gpus_for_parallel_eval(
    *,
    ray_num_gpus: int,
    tp_size: int,
    log: logging.Logger,
) -> tuple[int, str | None, dict]:
    """
    TP=1 且多 worker 并行时：仅「空闲显存 >= min_free_gb」的卡才应承接 Ray worker。
    否则会出现「请求 4 并行但 0/1 号卡仅 ~10GiB 空闲」仍被调度，进而在 merge+eval 时 OOM。

    返回：(effective_ray_num_gpus, 子进程 CUDA_VISIBLE_DEVICES 或 None, 写入 metadata 的补丁 dict)
    """
    from core.gpu_topology import env_int, query_gpus

    patch: dict = {}
    if tp_size != 1 or ray_num_gpus <= 1:
        return ray_num_gpus, None, patch

    min_free_gb = env_int("MERGEKIT_EVOLUTION_MIN_FREE_GB", env_int("MERGEKIT_EVAL_MIN_FREE_GB", 12))
    min_mib = int(min_free_gb * 1024)
    vis = _parse_cuda_visible_devices()
    try:
        gpus = query_gpus()
    except Exception as e:
        log.warning("[ray] query_gpus 失败，跳过 ray 并行数裁剪: %s", e)
        return ray_num_gpus, None, patch
    if vis is not None:
        gpus = [g for g in gpus if g.index in set(vis)]
    if not gpus:
        return ray_num_gpus, None, patch

    eligible = sorted([g for g in gpus if g.mem_free_mib >= min_mib], key=lambda g: -g.mem_free_mib)
    if not eligible:
        log.warning(
            "[ray] 无 GPU 满足空闲 >= %s GiB，将 ray_num_gpus %s 降为 1",
            min_free_gb,
            ray_num_gpus,
        )
        patch["ray_num_gpus_effective"] = 1
        patch["ray_cap_reason"] = "no_gpu_meets_min_free"
        return 1, None, patch

    eff = min(ray_num_gpus, len(eligible))
    picked = eligible[:eff]
    subset_needed = len(eligible) < len(gpus) or eff < ray_num_gpus
    if not subset_needed:
        return eff, None, patch

    cvd = ",".join(str(g.index) for g in picked)
    log.warning(
        "[ray] 空闲>=%s GiB 的卡 %s/%s 张；ray_num_gpus %s -> %s；子进程 CUDA_VISIBLE_DEVICES=%s",
        min_free_gb,
        len(eligible),
        len(gpus),
        ray_num_gpus,
        eff,
        cvd,
    )
    patch["ray_num_gpus_effective"] = eff
    patch["evolution_cuda_visible_devices"] = cvd
    patch["ray_cap_reason"] = "vram_eligible_subset"
    return eff, cvd, patch


def _count_evolution_csv_data_rows(csv_path: str) -> int:
    """统计 evolution_stream CSV 的数据行数（不含表头）。"""
    if not os.path.isfile(csv_path):
        return 0
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            r = csv_std.reader(f)
            next(r, None)  # header
            return sum(1 for _ in r)
    except Exception:
        return 0


def _rebuild_evolution_csv_from_log(
    subprocess_log_path: str,
    csv_path: str,
    log: logging.Logger,
) -> int:
    """
    从 subprocess_output.log 解析 [eval] 行重写 evolution_stream.csv，
    与主循环中 generation 检测逻辑保持一致。
    """
    eval_pat = re.compile(
        r"\[eval\]\s+step=(\d+)\s+acc=([\d.]+)(?:\s+best_acc=([\d.]+))?"
    )
    geno_pat = re.compile(r"genotype=\[([\d.,\s\-eE+]+)\]")
    if not os.path.isfile(subprocess_log_path):
        log.warning("无法从日志重建 CSV：日志不存在 %s", subprocess_log_path)
        return 0
    try:
        with open(subprocess_log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        log.warning("读取子进程日志失败，无法重建 CSV: %s", e)
        return 0

    eval_count = 0
    max_acc_seen = 0.0
    current_gen = 0
    prev_within_gen_step = 0
    rows_out: list[list] = []

    for line in lines:
        m = eval_pat.search(line)
        if not m:
            continue
        within_gen_step = int(m.group(1))
        acc_val = float(m.group(2))
        best_acc_val = float(m.group(3)) if m.group(3) else acc_val

        if within_gen_step <= prev_within_gen_step and eval_count > 0:
            current_gen += 1
        elif eval_count == 0:
            current_gen = 1
        prev_within_gen_step = within_gen_step

        eval_count += 1
        max_acc_seen = max(max_acc_seen, acc_val, best_acc_val)

        genotype: list[float] = []
        gm = geno_pat.search(line)
        if gm:
            try:
                genotype = [float(x.strip()) for x in gm.group(1).split(",")]
            except (ValueError, TypeError):
                pass
        g1 = genotype[0] if len(genotype) > 0 else 0.0
        g2 = genotype[1] if len(genotype) > 1 else 0.0
        rows_out.append(
            [eval_count, acc_val, g1, g2, current_gen, max_acc_seen]
        )

    if not rows_out:
        log.warning("日志中未解析到 [eval] 行，跳过 CSV 重建")
        return 0

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv_std.writer(cf)
        w.writerow(
            ["step", "objective_1", "genotype_1", "genotype_2", "generation", "best_acc"]
        )
        for row in rows_out:
            w.writerow([str(v).replace("\x00", "") for v in row])
    log.info("已从子进程日志重建 evolution_stream.csv，共 %d 行", len(rows_out))
    return len(rows_out)


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
    max_evals = int(meta.get("max_evals") or 0)
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
    # max_evals（快测门禁）：为保证“硬上限”严格生效，强制串行评估（Ray 并行会导致 termination 超额调度）
    if max_evals > 0 and ray_num_gpus > 1:
        logger.info("[max_evals] max_evals=%s：为保证严格上限，强制 ray_num_gpus=1（串行）", max_evals)
        ray_num_gpus = 1
        meta["ray_num_gpus_effective"] = 1
        _write_metadata_safe(task_id, merge_dir, meta, logger)
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

    # 任务级 GPU 锁（默认启用）：避免同容器并发任务互相挤占显存
    lock_timeout_s = int(float(os.environ.get("MERGEKIT_GPU_LOCK_TIMEOUT_S", "30") or 30))
    lock_path = (os.environ.get("MERGEKIT_GPU_LOCK_PATH") or "/tmp/mergekit_gpu.lock").strip()
    lock_fp = None
    try:
        lock_fp = _lock_file(lock_path, lock_timeout_s, logger)
        logger.info("已获得 GPU 锁: %s", lock_path)
    except TimeoutError as e:
        # 不强制失败：按“保证完成”策略继续，但会在 TP2 选择时更可能触发降级
        logger.warning("GPU 锁获取超时，将继续执行并允许降级: %s", e)

    python_cmd = (getattr(Config, "MERGENETIC_PYTHON", None) or "").strip() or "python"

    # TP 策略：MERGEKIT_EVOLUTION_TP2=auto|1|0（默认 auto）
    tp_size, chosen_pairs, eval_batch_size, decision_patch = _decide_tp_and_batch(meta=meta, log=logger)
    meta.update(decision_patch)
    _write_metadata_safe(task_id, merge_dir, meta, logger)
    logger.info(
        "[tp] mode=%s tp_size=%s pairs=%s free_gib=%s batch_init=%s reason=%s",
        meta.get("evolution_tp2_mode"),
        tp_size,
        meta.get("tp2_pairs", ""),
        meta.get("gpu_free_gib_snapshot", []),
        eval_batch_size,
        meta.get("tp_decision_reason"),
    )
    if tp_size > 1 and chosen_pairs:
        visible = sorted({i for a, b in chosen_pairs for i in (a, b)})
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in visible)
        os.environ["MERGEKIT_EVOLUTION_TP2_PAIRS"] = ";".join(f"{a},{b}" for a, b in chosen_pairs)
        ray_num_gpus = 2 * len(chosen_pairs)

    # TP=1：避免在「未达 min_free_gb」的卡上与用户请求的 ray_num_gpus 叠床架屋导致 OOM（见 feb497dc 类任务）
    ray_cuda_visible_for_child: str | None = None
    if tp_size == 1 and ray_num_gpus > 1:
        new_n, cvd, cap_patch = _cap_ray_num_gpus_for_parallel_eval(
            ray_num_gpus=ray_num_gpus, tp_size=tp_size, log=logger
        )
        ray_num_gpus = new_n
        ray_cuda_visible_for_child = cvd
        if cap_patch:
            meta.update(cap_patch)
            _write_metadata_safe(task_id, merge_dir, meta, logger)

    cmd = [
        python_cmd,
        RUN_VLM_SEARCH_PY,
        "--ray-num-gpus", str(ray_num_gpus),
        "--tp-size", str(tp_size),
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
        "--max-evals", str(int(meta.get("max_evals") or 0)),
        "--max-samples", str(max_samples),
        "--dtype", dtype,
        "--device", "cuda",
        "--final-vlm-output", final_vlm_output,
        "--progress-file", progress_path,
        "--results-dir", results_dir,
        "--batch-size", str(eval_batch_size),
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

    # 子进程环境：继承当前进程 + Config 白名单合并（HF/MERGEKIT/可选 NCCL），避免与 merge_manager 路径混用全局污染
    env = os.environ.copy()
    try:
        patch = Config.evolution_subprocess_env_patch()
        env.update(patch)
    except Exception as e:
        logger.warning("evolution_subprocess_env_patch 失败，回退仅 HF 键: %s", e)
        if getattr(Config, "HF_ENDPOINT", None):
            env["HF_ENDPOINT"] = Config.HF_ENDPOINT
        if getattr(Config, "HF_DATASETS_CACHE", None):
            env["HF_DATASETS_CACHE"] = Config.HF_DATASETS_CACHE
    # 由 runner 独占写 merges/<task_id>/progress.json；子进程仅可写 progress_mergenetic_debug.json
    env["MERGEKIT_RUNNER_OWNS_PROGRESS"] = "1"
    if ray_cuda_visible_for_child:
        env["CUDA_VISIBLE_DEVICES"] = ray_cuda_visible_for_child
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
        merge_step_times: list[float] = []  # 从子进程 [eval] 行解析出的 merge=..s
        eval_step_times: list[float] = []   # 从子进程 [eval] 行解析出的 eval=..s
        eval_start_time = None    # 当前评估开始时间
        # 子进程读循环起点：超时与 total_task_time 用 monotonic，避免系统时间回拨/校时导致误判
        task_start_mono = time_module.monotonic()
        task_duration_limit_s = float(
            getattr(Config, "MERGEKIT_MAX_TASK_DURATION_S", 14400) or 14400
        )
        total_expected_steps = max(1, pop_size * n_iter)

        # ── 准备 CSV：每步追加，前端/3D 图可用 ──
        csv_dir = os.path.join(merge_dir, "vlm_search_results")
        os.makedirs(csv_dir, exist_ok=True)
        # 使用独立文件名，避免 mergenetic Searcher 结束时 to_csv(vlm_search.csv) 覆盖流式写入
        csv_path = os.path.join(csv_dir, "evolution_stream.csv")
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
                # 全局超时门禁：防止子进程卡死长期占用 GPU（墙钟用 monotonic，与 MERGEKIT_MAX_TASK_DURATION_S 对齐）
                elapsed_mono = time_module.monotonic() - task_start_mono
                if elapsed_mono > task_duration_limit_s:
                    logger.error(
                        "任务超时: elapsed=%.1fs > limit=%.0fs，将终止 run_vlm_search 子进程 (PID=%s)",
                        elapsed_mono,
                        task_duration_limit_s,
                        getattr(proc, "pid", None),
                    )
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    try:
                        with open(progress_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "status": "error",
                                    "message": "任务超时，已终止子进程",
                                    "timeout_elapsed_seconds": round(elapsed_mono, 2),
                                    "timeout_limit_seconds": int(task_duration_limit_s),
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                    except Exception:
                        pass
                    # 清理大目录（保留日志/元数据）
                    for dn in ("vlm_search_results", "final_vlm", "output"):
                        try:
                            shutil.rmtree(os.path.join(merge_dir, dn), ignore_errors=True)
                        except Exception:
                            pass
                    sys.exit(1)
                line = proc.stdout.readline() if proc.stdout else ""
                if not line and proc.poll() is not None:
                    break
                if line:
                    line = line.replace('\x00', '')
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
                    # 可选：解析 run_vlm_search.py 新增的分拆计时字段（merge/eval）
                    m_merge = re.search(r"merge=([\d.]+)s", line)
                    m_eval = re.search(r"eval=([\d.]+)s", line)
                    if m_merge:
                        try:
                            merge_step_times.append(float(m_merge.group(1)))
                        except Exception:
                            pass
                    if m_eval:
                        try:
                            eval_step_times.append(float(m_eval.group(1)))
                        except Exception:
                            pass

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
                    row_vals = [eval_count, acc_val, g1, g2, current_gen, max_acc_seen]
                    csv_writer.writerow([str(v).replace('\x00', '') for v in row_vals])
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
                    # Ray 并行下 [eval] 行内 step 为“worker 内部计数”，仅作展示用途，不能作为全局进度
                    prog["within_gen_step"] = within_gen_step
                    prog["eval_count_global"] = eval_count
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
                    if merge_step_times:
                        recent_m = merge_step_times[-min(5, len(merge_step_times)):]
                        prog["avg_merge_time"] = sum(recent_m) / len(recent_m)
                    if eval_step_times:
                        recent_e = eval_step_times[-min(5, len(eval_step_times)):]
                        prog["avg_eval_time"] = sum(recent_e) / len(recent_e)
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
        try:
            if lock_fp is not None:
                lock_fp.close()
        except Exception:
            pass
            logger.info("CSV 已写入 %d 行到 %s", csv_rows_written, csv_path)

        proc.wait()

        # 兜底：mergenetic 结束时可能写空 vlm_search.csv；若流式 evolution_stream 行数不足则从日志重建
        n_csv = _count_evolution_csv_data_rows(csv_path)
        if csv_rows_written > 0 and n_csv < csv_rows_written:
            logger.warning(
                "evolution_stream.csv 数据行 %d 少于预期 %d，从子进程日志重建",
                n_csv,
                csv_rows_written,
            )
            _rebuild_evolution_csv_from_log(subprocess_log_path, csv_path, logger)

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
                prog["total_task_time"] = time_module.monotonic() - task_start_mono
                if merge_step_times:
                    prog["total_merge_time"] = float(sum(merge_step_times))
                if eval_step_times:
                    prog["total_eval_time"] = float(sum(eval_step_times))
                if "total_task_time" in prog and ("total_merge_time" in prog or "total_eval_time" in prog):
                    prog["total_overhead_time"] = float(
                        (prog.get("total_task_time") or 0)
                        - (prog.get("total_merge_time") or 0)
                        - (prog.get("total_eval_time") or 0)
                    )
                prog["current_best"] = max(max_acc_seen, prog.get("current_best", 0) or 0)
                prog["global_best"] = prog["current_best"]
                if gen_durations:
                    prog["gen_durations"] = gen_durations
                _flush_progress(prog)
                # 同步一份 timings 到 metadata，作为单一可追溯来源
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta2 = json.load(mf)
                    meta2["timings"] = {
                        "total_task_time": prog.get("total_task_time"),
                        "total_merge_time": prog.get("total_merge_time"),
                        "total_eval_time": prog.get("total_eval_time"),
                        "total_overhead_time": prog.get("total_overhead_time"),
                        "avg_step_time": prog.get("avg_step_time"),
                        "avg_merge_time": prog.get("avg_merge_time"),
                        "avg_eval_time": prog.get("avg_eval_time"),
                        "step_count": eval_count,
                    }
                    _write_metadata_safe(task_id, merge_dir, meta2, logger)
                except Exception as e:
                    logger.debug("同步 timings 到 metadata 失败（可忽略）: %s", e)
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
        # 可选跳过：用于 smoke/快测，避免 final 评测拖慢墙钟
        _skip_final = bool(meta.get("skip_final_eval", False)) or (
            (os.environ.get("MERGEKIT_SKIP_FINAL_EVAL") or "").strip().lower() in ("1", "true", "yes", "on")
        )
        if _skip_final and proc.returncode == 0 and hf_split_final and hf_split_final != hf_split:
            logger.info("[final] skip_final_eval=true，跳过最终评测 (eval_final.py)")
            # 写占位文件，供 _do_success_path 读取并在 fusion_info/recipe 中可见
            try:
                with open(final_acc_file, "w", encoding="utf-8") as f:
                    json.dump({"final_test_acc": "skipped", "skipped": True}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning("写入 final_test_acc.json 占位失败（可忽略）: %s", e)
        if (not _skip_final) and proc.returncode == 0 and hf_split_final and hf_split_final != hf_split:
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

        # progress.json 终态与 metadata success 对齐，避免前端仍显示 running
        try:
            with open(progress_path, "r", encoding="utf-8") as pf:
                prog_done = json.load(pf)
            prog_done["status"] = "completed"
            prog_done["percent"] = 100
            prog_done["eta_seconds"] = 0
            prog_done["message"] = "任务完成"
            _flush_progress(prog_done)
        except Exception as e_done:
            logger.warning("写入 progress 终态 completed 失败: %s", e_done)

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
