from dataclasses import dataclass, field
import queue
import threading

from config import Config


@dataclass
class AppState:
    config: object = Config
    logger: object = None
    merge_dir: str = field(default_factory=lambda: Config.MERGE_DIR)
    model_pool_path: str = field(default_factory=lambda: Config.MODEL_POOL_PATH)
    recipes_dir: str = field(default_factory=lambda: Config.RECIPES_DIR)
    tasks: dict = field(default_factory=dict)
    task_queue: queue.PriorityQueue = field(default_factory=queue.PriorityQueue)
    running_task_info: dict = field(default_factory=lambda: {"id": None, "priority": None, "process": None})
    scheduler_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def priority_map(self):
        return getattr(self.config, "PRIORITY_MAP", {})

    @property
    def testset_data_path(self):
        return getattr(self.config, "TESTSET_DATA_PATH", None) or ""

    @property
    def project_root(self):
        return getattr(self.config, "PROJECT_ROOT", None) or ""
