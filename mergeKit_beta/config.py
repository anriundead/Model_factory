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
    # 测试集仓库：用户下载的 HF 数据集在此登记，测试集列表与评估页共用
    TESTSET_REPO = os.path.join(PROJECT_ROOT, "testset_repo")
    TESTSET_DATA_PATH = os.path.join(PROJECT_ROOT, "testset_repo", "data", "testsets.json")
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
    EFFICIENCY_BASE_SPS = 10
    # 内置基准测试专用：lm_eval 使用的 HF 数据集缓存目录，与系统默认缓存隔离，避免版本兼容问题
    # 可通过环境变量 MERGEKIT_EVAL_HF_CACHE 覆盖，不设则使用项目下 cache/eval_datasets
    # 说明：内置基准(hellaswag/arc_easy/boolq/winogrande)由 lm_eval 加载，lm_eval 依赖 HuggingFace 的 datasets 库；
    # 本项目不直接依赖 datasets，若未装 lm_eval 则不会用到。早期版本若无需 datasets 多为未走 lm_eval 或环境已带旧版依赖。
    _eval_cache = os.environ.get("MERGEKIT_EVAL_HF_CACHE", "").strip()
    EVAL_HF_DATASETS_CACHE = os.path.abspath(_eval_cache) if _eval_cache else os.path.join(PROJECT_ROOT, "cache", "eval_datasets")

    # ==================== 环境变量配置 ====================
    NUMEXPR_MAX_THREADS = 64
    HF_ENDPOINT = "https://hf-mirror.com"
    # HuggingFace 数据集缓存目录（已下载的数据集优先从此读取，不设则使用 datasets 默认缓存）
    _hf_cache_env = os.environ.get("HF_DATASETS_CACHE", "").strip()
    HF_DATASETS_CACHE = os.path.abspath(_hf_cache_env) if _hf_cache_env else os.path.join(PROJECT_ROOT, "cache", "datasets")

    # ==================== 融合库配置 ====================
    MERGE_LIBRARY = "mergenetic"

    # ==================== 数据库配置 ====================
    # 默认使用 SQLite，文件存储在项目根目录下
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or "sqlite:///" + os.path.join(PROJECT_ROOT, "app.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ==================== Mergenetic 使用的 Python 环境（可选） ====================
    # 优先使用环境变量，否则尝试 mergenetic conda 环境，最后回退到 python
    _default_mergenetic_python = "/home/a/miniconda3/envs/mergenetic/bin/python"
    if not os.path.isfile(_default_mergenetic_python):
        _default_mergenetic_python = "python"
    MERGENETIC_PYTHON = os.environ.get("MERGENETIC_PYTHON", _default_mergenetic_python)

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
        os.makedirs(cls.EVAL_HF_DATASETS_CACHE, exist_ok=True)
        if cls.HF_DATASETS_CACHE:
            os.makedirs(cls.HF_DATASETS_CACHE, exist_ok=True)
