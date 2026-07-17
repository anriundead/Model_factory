"""
配置模块 - 集中管理所有配置项（mergeKit_beta，使用 mergenetic 融合）
"""
import os
from pathlib import Path

from core.path_utils import (
    configured_local_models_path,
    configured_merge_dir,
    configured_model_pool_path,
    project_root,
)


# 项目根目录（mergenetic 会使用当前工作目录，此处用于解析相对路径）
def _project_root() -> Path:
    return project_root()


class Config:
    """应用配置类"""

    # ==================== 路径配置 ====================
    PROJECT_ROOT = str(_project_root())
    # 融合模型池路径：优先环境变量，否则 config/paths.json 相对路径
    MODEL_POOL_PATH = configured_model_pool_path()
    # 基座模型目录：优先环境变量，否则 config/paths.json 相对路径
    LOCAL_MODELS_PATH = configured_local_models_path()
    # 额外模型目录（可与主目录同级，如 Models-local_dir）；仅包含存在的目录
    _extra = os.path.join(os.path.dirname(LOCAL_MODELS_PATH), "Models-local_dir")
    LOCAL_MODELS_EXTRA_PATHS = [os.path.abspath(_extra)] if os.path.isdir(_extra) else []
    MERGE_DIR = configured_merge_dir()
    PUBLISHED_MODELS_PATH = os.path.abspath(
        os.environ.get("MERGEKIT_PUBLISHED_MODELS_PATH", "/data/PublishedModels")
    )
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

    # ==================== GPU 拓扑 / TP2 ====================
    # 评测（lm_eval 推理阶段）使用 NVLink 卡对 TP=2：pair01 或 pair23 或 auto
    MERGEKIT_EVAL_TOPOLOGY = (os.environ.get("MERGEKIT_EVAL_TOPOLOGY") or "off").strip().lower()
    # 评测阶段：获取卡对等待秒数；超时降级单卡
    MERGEKIT_EVAL_PAIR_TIMEOUT_S = int(float(os.environ.get("MERGEKIT_EVAL_PAIR_TIMEOUT_S", "30") or 30))
    # 评测阶段：每卡最小空闲显存（GiB），用于避开被常驻进程挤占的卡
    MERGEKIT_EVAL_MIN_FREE_GB = int(float(os.environ.get("MERGEKIT_EVAL_MIN_FREE_GB", "12") or 12))
    # 进化融合 TP2 开关（runner 侧默认开启；这里只做集中定义，供外部文档/一致性）
    MERGEKIT_EVOLUTION_TP2 = (os.environ.get("MERGEKIT_EVOLUTION_TP2") or "1").strip().lower() not in ("0", "false", "no", "off")
    # vLLM TP>1 时 c10d 绑定地址；空则 run_vlm_search 内默认 127.0.0.1
    MERGEKIT_VLLM_MASTER_ADDR = (os.environ.get("MERGEKIT_VLLM_MASTER_ADDR") or "").strip()
    # 进化子进程内 NCCL 走 socket 回退（与 merge_manager 评测路径语义类似；默认关，故障时开）
    _evo_nccl_shm = (os.environ.get("MERGEKIT_EVOLUTION_NCCL_SHM_DISABLE") or "").strip().lower()
    MERGEKIT_EVOLUTION_NCCL_SHM_DISABLE = _evo_nccl_shm in ("1", "true", "yes", "on")
    _evo_nccl_ib = (os.environ.get("MERGEKIT_EVOLUTION_NCCL_IB_DISABLE") or "").strip().lower()
    MERGEKIT_EVOLUTION_NCCL_IB_DISABLE = _evo_nccl_ib in ("1", "true", "yes", "on")
    # 单机 Docker：显式 1 时 ray.init 可用 127.0.0.1；多机请勿开
    _ray_loop = (os.environ.get("MERGEKIT_RAY_SINGLE_NODE_LOOPBACK") or "").strip().lower()
    MERGEKIT_RAY_SINGLE_NODE_LOOPBACK = _ray_loop in ("1", "true", "yes", "on")

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
    # Gateway state may move to its own PostgreSQL database without migrating the
    # model-factory SQLite registry. Empty keeps today's single-file deployment.
    MERGEKIT_MODEL_GATEWAY_DATABASE_URL = os.environ.get("MERGEKIT_MODEL_GATEWAY_DATABASE_URL", "").strip()
    SQLALCHEMY_BINDS = {
        "model_gateway": MERGEKIT_MODEL_GATEWAY_DATABASE_URL or SQLALCHEMY_DATABASE_URI,
    }

    # ==================== 运行时门禁/超时 ====================
    # 全局任务超时（秒）：从「子进程 Popen 后进入 stdout 读循环」起算（monotonic），默认 14400=4h。
    # 长跑进化请调大 MERGEKIT_MAX_TASK_DURATION_S；到点会 terminate 子进程并写 progress.json 含 elapsed/limit。
    MERGEKIT_MAX_TASK_DURATION_S = int(float(os.environ.get("MERGEKIT_MAX_TASK_DURATION_S", "14400") or 14400))

    # ==================== Mergenetic 使用的 Python 环境（可选） ====================
    # 优先使用环境变量，否则尝试 mergenetic conda 环境，最后回退到 python
    _default_mergenetic_python = "/home/a/miniconda3/envs/mergenetic/bin/python"
    if not os.path.isfile(_default_mergenetic_python):
        _default_mergenetic_python = "python"
    MERGENETIC_PYTHON = os.environ.get("MERGENETIC_PYTHON", _default_mergenetic_python)

    # ==================== 模型服务（vLLM Gateway） ====================
    MERGEKIT_MODEL_GATEWAY_ENABLED = (os.environ.get("MERGEKIT_MODEL_GATEWAY_ENABLED") or "1").strip().lower() not in ("0", "false", "no", "off")
    MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN = (os.environ.get("MERGEKIT_MODEL_GATEWAY_ADMIN_TOKEN") or "").strip()
    MERGEKIT_MODEL_GATEWAY_VLLM_BIN = os.environ.get("MERGEKIT_MODEL_GATEWAY_VLLM_BIN", "/opt/conda/envs/mergenetic/bin/vllm")
    MERGEKIT_MODEL_GATEWAY_PYTHON = os.environ.get("MERGEKIT_MODEL_GATEWAY_PYTHON", MERGENETIC_PYTHON)
    MERGEKIT_MODEL_GATEWAY_VLLM_USE_COMPAT_WRAPPER = (os.environ.get("MERGEKIT_MODEL_GATEWAY_VLLM_USE_COMPAT_WRAPPER") or "1").strip().lower() not in ("0", "false", "no", "off")
    MERGEKIT_MODEL_GATEWAY_PORT_START = int(float(os.environ.get("MERGEKIT_MODEL_GATEWAY_PORT_START", "18000") or 18000))
    MERGEKIT_MODEL_GATEWAY_PORT_END = int(float(os.environ.get("MERGEKIT_MODEL_GATEWAY_PORT_END", "18999") or 18999))
    MERGEKIT_MODEL_GATEWAY_SYNC_WAIT_SECONDS = float(os.environ.get("MERGEKIT_MODEL_GATEWAY_SYNC_WAIT_SECONDS", "60") or 60)
    MERGEKIT_MODEL_GATEWAY_LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "model_gateway")
    # Research sources are ephemeral and never share model-factory output paths.
    MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT = os.environ.get(
        "MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT", os.path.join(PROJECT_ROOT, "runtime", "model_gateway")
    )
    MERGEKIT_MODEL_GATEWAY_EMBEDDING_MODEL_PATH = os.environ.get(
        "MERGEKIT_MODEL_GATEWAY_EMBEDDING_MODEL_PATH",
        os.path.join(MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT, "embedding_models", "bge-m3"),
    )
    MERGEKIT_MODEL_GATEWAY_RESEARCH_RECONCILE_SECONDS = int(
        os.environ.get("MERGEKIT_MODEL_GATEWAY_RESEARCH_RECONCILE_SECONDS", "30") or 30
    )
    MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND = os.environ.get("MERGEKIT_MODEL_GATEWAY_QUEUE_BACKEND", "db").strip().lower()
    MERGEKIT_MODEL_GATEWAY_REDIS_URL = os.environ.get("MERGEKIT_MODEL_GATEWAY_REDIS_URL", "").strip()
    MERGEKIT_MODEL_GATEWAY_WORKER_TOKEN = os.environ.get("MERGEKIT_MODEL_GATEWAY_WORKER_TOKEN", "").strip()
    MERGEKIT_MODEL_GATEWAY_CLAMAV_HOST = os.environ.get("MERGEKIT_MODEL_GATEWAY_CLAMAV_HOST", "model-gateway-clamav").strip()
    MERGEKIT_MODEL_GATEWAY_CLAMAV_PORT = int(os.environ.get("MERGEKIT_MODEL_GATEWAY_CLAMAV_PORT", "3310") or 3310)
    MERGEKIT_MODEL_GATEWAY_FILE_RECONCILE_SECONDS = int(os.environ.get("MERGEKIT_MODEL_GATEWAY_FILE_RECONCILE_SECONDS", "30") or 30)
    # Invited-user pilot safeguards. Values are intentionally conservative because
    # parsing, retrieval and model serving share this host with Model Factory.
    MERGEKIT_MODEL_GATEWAY_CHAT_REQUESTS_PER_MINUTE = int(os.environ.get("MERGEKIT_MODEL_GATEWAY_CHAT_REQUESTS_PER_MINUTE", "20") or 20)
    MERGEKIT_MODEL_GATEWAY_RESEARCH_SUBMISSIONS_PER_HOUR = int(os.environ.get("MERGEKIT_MODEL_GATEWAY_RESEARCH_SUBMISSIONS_PER_HOUR", "4") or 4)
    MERGEKIT_MODEL_GATEWAY_MAX_ACTIVE_RESEARCH_JOBS = int(os.environ.get("MERGEKIT_MODEL_GATEWAY_MAX_ACTIVE_RESEARCH_JOBS", "1") or 1)
    MERGEKIT_MODEL_GATEWAY_IMPORT_BYTES_PER_DAY = int(os.environ.get("MERGEKIT_MODEL_GATEWAY_IMPORT_BYTES_PER_DAY", str(500 * 1024 * 1024)) or (500 * 1024 * 1024))
    MERGEKIT_MODEL_GATEWAY_CLEANUP_INTERVAL_SECONDS = int(os.environ.get("MERGEKIT_MODEL_GATEWAY_CLEANUP_INTERVAL_SECONDS", "60") or 60)
    MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_URL = os.environ.get("MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_URL", "").strip()
    MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_TOKEN = os.environ.get("MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_TOKEN", "").strip()
    MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_TIMEOUT_SECONDS = int(
        os.environ.get("MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_TIMEOUT_SECONDS", "45") or 45
    )
    MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_MAX_MIB = int(
        os.environ.get("MERGEKIT_MODEL_GATEWAY_LEGACY_PARSER_MAX_MIB", "50") or 50
    )
    MERGEKIT_MODEL_GATEWAY_MAX_SOURCES = int(os.environ.get("MERGEKIT_MODEL_GATEWAY_MAX_SOURCES", "10"))
    MERGEKIT_MODEL_GATEWAY_MAX_SOURCE_MIB = int(os.environ.get("MERGEKIT_MODEL_GATEWAY_MAX_SOURCE_MIB", "50"))
    MERGEKIT_MODEL_GATEWAY_SOURCE_TTL_HOURS = float(os.environ.get("MERGEKIT_MODEL_GATEWAY_SOURCE_TTL_HOURS", "24") or 24)

    # 进化融合：为真时使用 scripts/run_vlm_search_bridge.py 作为子进程入口；默认使用 python -m evolution.runner
    _evo_legacy = os.environ.get("MERGEKIT_EVOLUTION_LEGACY_BRIDGE", "").strip().lower()
    MERGEKIT_EVOLUTION_LEGACY_BRIDGE = _evo_legacy in ("1", "true", "yes")

    # lm_eval 子进程：merge_manager 默认注入 HF_ALLOW_CODE_EVAL=1（代码类任务名多样，仅靠名字匹配易漏）；
    # 禁止自动注入请设 MERGEKIT_FORBID_CODE_EVAL=1；若仍选代码类测试集会在启动前报错。
    # 高级：父进程设 HF_ALLOW_CODE_EVAL=0 可跳过注入（仅适合确认无 code_eval 的任务）。
    _forbid_ce = os.environ.get("MERGEKIT_FORBID_CODE_EVAL", "").strip().lower()
    MERGEKIT_FORBID_CODE_EVAL = _forbid_ce in ("1", "true", "yes")

    @classmethod
    def evolution_merge_entry_exists(cls) -> bool:
        root = cls.PROJECT_ROOT
        if cls.MERGEKIT_EVOLUTION_LEGACY_BRIDGE:
            return os.path.isfile(os.path.join(root, "scripts", "run_vlm_search_bridge.py"))
        init_py = os.path.join(root, "evolution", "__init__.py")
        runner_py = os.path.join(root, "evolution", "runner.py")
        return os.path.isfile(init_py) and os.path.isfile(runner_py)

    @classmethod
    def evolution_merge_popen_args(cls, task_id: str) -> tuple[list[str], str]:
        """返回 (argv, cwd)，供 merge_evolutionary Worker subprocess.Popen 使用。"""
        py = cls.MERGENETIC_PYTHON
        root = cls.PROJECT_ROOT
        tid = str(task_id).strip()
        if cls.MERGEKIT_EVOLUTION_LEGACY_BRIDGE:
            script = os.path.join(root, "scripts", "run_vlm_search_bridge.py")
            return [py, script, "--task-id", tid], root
        return [py, "-m", "evolution.runner", "--task-id", tid], root

    @classmethod
    def evolution_subprocess_env_patch(cls) -> dict[str, str]:
        """
        供 evolution.runner 启动 run_vlm_search 子进程时合并到 env（白名单，不污染 Flask 主进程）。
        仅覆盖/补充与 HF 缓存、MERGEKIT_* 开关、可选 NCCL 相关的键。
        """
        patch: dict[str, str] = {
            "HF_ENDPOINT": cls.HF_ENDPOINT,
        }
        if cls.HF_DATASETS_CACHE:
            patch["HF_DATASETS_CACHE"] = cls.HF_DATASETS_CACHE
        if cls.MERGEKIT_VLLM_MASTER_ADDR:
            patch["MERGEKIT_VLLM_MASTER_ADDR"] = cls.MERGEKIT_VLLM_MASTER_ADDR
        if cls.MERGEKIT_EVOLUTION_NCCL_SHM_DISABLE:
            patch["NCCL_SHM_DISABLE"] = "1"
        if cls.MERGEKIT_EVOLUTION_NCCL_IB_DISABLE:
            patch["NCCL_IB_DISABLE"] = "1"
        # 父进程中已设置的 MERGEKIT_/HF_/离线相关开关，显式传入子进程，避免继承灰区
        for key in (
            "MERGEKIT_VLLM_TP_SERIALIZE",
            "MERGEKIT_VLLM_ENABLE",
            "MERGEKIT_EVOLUTION_TP2",
            "MERGEKIT_EVOLUTION_TP2_PAIRS",
            "MERGEKIT_TOPOLOGY",
            "MERGEKIT_EVOLUTION_TOPOLOGY",
            "MERGEKIT_EVOLUTION_MIN_FREE_GB",
            "MERGEKIT_DATASETS_LOCAL_ONLY",
            "MERGEKIT_GPU_LOCK_TIMEOUT_S",
            "MERGEKIT_GPU_LOCK_PATH",
            "MERGEKIT_DEBUG_PRED_LIMIT",
            "MERGEKIT_RAY_NODE_IP_ADDRESS",
            "MERGEKIT_RAY_SINGLE_NODE_LOOPBACK",
            "MERGEKIT_VLLM_TP_SUBPROCESS",
            "MERGEKIT_VLLM_SUBPROCESS_TIMEOUT_S",
            "MERGEKIT_RAY_POOL_RUNTIME_ENV",
            "CUDA_VISIBLE_DEVICES",
            "HF_DATASETS_OFFLINE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_HUB_DISABLE_XET",
            "HF_TOKEN",
        ):
            val = os.environ.get(key)
            if val is not None and str(val) != "":
                patch[key] = str(val)
        return patch

    @classmethod
    def setup_environment(cls):
        """统一设置环境变量"""
        os.environ["HF_ENDPOINT"] = cls.HF_ENDPOINT
        os.environ["NUMEXPR_MAX_THREADS"] = str(cls.NUMEXPR_MAX_THREADS)
        os.makedirs(cls.MERGE_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        os.makedirs(cls.MERGEKIT_MODEL_GATEWAY_LOG_DIR, exist_ok=True)
        os.makedirs(cls.MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT, exist_ok=True)
        os.makedirs(cls.RECIPES_DIR, exist_ok=True)
        if not os.path.isdir(cls.MODEL_POOL_PATH):
            os.makedirs(cls.MODEL_POOL_PATH, exist_ok=True)
        os.makedirs(cls.EVAL_HF_DATASETS_CACHE, exist_ok=True)
        if cls.HF_DATASETS_CACHE:
            os.makedirs(cls.HF_DATASETS_CACHE, exist_ok=True)
