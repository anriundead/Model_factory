"""
配置模块 - 集中管理所有配置项（mergeKit_beta，使用 mergenetic 融合）
"""
import os
from pathlib import Path


# 项目根目录（mergenetic 会使用当前工作目录，此处用于解析相对路径）
def _project_root() -> Path:
    return Path(__file__).resolve().parent


class Config:
    """应用配置类"""

    # ==================== 路径配置 ====================
    PROJECT_ROOT = str(_project_root())
    # 融合模型池路径：优先使用环境变量或绝对路径，否则相对项目上级 mergeKit
    MODEL_POOL_PATH = os.environ.get(
        "MERGEKIT_MODEL_POOL",
        os.path.join(os.path.dirname(PROJECT_ROOT), "..", "mergeKit", "models_pool"),
    )
    MODEL_POOL_PATH = os.path.abspath(MODEL_POOL_PATH)
    # 基座模型（本地可融合模型）的正式存放路径，界面「本地基座模型」列表由此读取
    LOCAL_MODELS_PATH = os.path.abspath(os.environ.get("LOCAL_MODELS_PATH", "/home/a/ServiceEndFiles/Models"))
    MERGE_DIR = os.path.join(PROJECT_ROOT, "merges")
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs", "merge")
    # 完全融合配方目录：保存可复现的配方（genotype + 参数），供直接融合
    RECIPES_DIR = os.path.join(PROJECT_ROOT, "recipes")

    # ==================== 优先级配置 ====================
    PRIORITY_MAP = {
        "vip": 0,
        "common": 10,
        "cutin": 20,
    }

    # ==================== 评估配置 ====================
    STANDARD_BENCHMARKS = ["hellaswag", "arc_easy", "boolq", "winogrande"]
    DEFAULT_DATASET = "hellaswag"
    DEFAULT_LIMIT = "0.5"

    # ==================== 环境变量配置 ====================
    NUMEXPR_MAX_THREADS = 64
    HF_ENDPOINT = "https://hf-mirror.com"
    # HuggingFace 数据集缓存目录（已下载的数据集优先从此读取，不设则使用 datasets 默认缓存）
    HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", "").strip() or None

    # ==================== 融合库配置 ====================
    MERGE_LIBRARY = "mergenetic"

    # ==================== Mergenetic 使用的 Python 环境（可选） ====================
    MERGENETIC_PYTHON = os.environ.get("MERGENETIC_PYTHON", "python")

    @classmethod
    def setup_environment(cls):
        """统一设置环境变量"""
        os.environ["HF_ENDPOINT"] = cls.HF_ENDPOINT
        os.environ["NUMEXPR_MAX_THREADS"] = str(cls.NUMEXPR_MAX_THREADS)
        os.makedirs(cls.MERGE_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        os.makedirs(cls.RECIPES_DIR, exist_ok=True)
        if not os.path.isdir(cls.MODEL_POOL_PATH):
            os.makedirs(cls.MODEL_POOL_PATH, exist_ok=True)
