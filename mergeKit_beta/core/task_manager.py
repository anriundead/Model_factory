"""
任务管理模块 - Task类和TaskQueue类
"""
import time
import queue
import threading
from config import Config


class Task:
    """任务类 - 封装任务的所有信息"""
    
    def __init__(self, task_id, task_type, params, priority='common'):
        """
        初始化任务
        
        参数:
            task_id: 任务ID
            task_type: 任务类型 ('merge' 或 'eval_only')
            params: 任务参数字典
            priority: 优先级 ('vip', 'common', 'cutin')
        """
        self.id = task_id
        self.type = task_type
        self.params = params
        self.priority = priority
        self.status = 'queued'  # queued/running/completed/error/interrupted/stopped
        self.progress = 0
        self.message = ""
        self.created_at = params.get('created_at', time.time())
        self.process_info = None  # 进程信息（Popen对象）
        self.result = None  # 任务结果
        self.config_path = None  # 配置文件路径（融合任务）
        self.parent_task_id = None  # 父任务ID（递归融合，预留）
        self.control = {"aborted": False, "process": None}  # 任务控制字典
        self.original_data = params  # 原始数据（用于恢复）
        self.restarted = False  # 是否被重启过
    
    def to_dict(self):
        """转换为字典（用于API返回）"""
        return {
            'id': self.id,
            'type': self.type,
            'priority': self.priority,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'created_at': self.created_at,
            'result': self.result
        }
    
    def update_progress(self, progress, message):
        """更新任务进度"""
        if self.status not in ["interrupted", "stopped"]:
            self.progress = progress
            self.message = message


class TaskQueue:
    """任务队列类 - 封装队列操作和状态管理"""
    
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.tasks = {}  # task_id -> Task对象
        self.running_task = None  # 当前运行的任务（Task对象）
        self.lock = threading.Lock()
    
    def add_task(self, task: Task):
        """
        添加任务到队列（带优先级逻辑）
        
        参数:
            task: Task对象
        """
        with self.lock:
            # 处理优先级逻辑
            if task.priority == 'cutin':
                # Cutin逻辑：打断所有非Cutin任务
                if self.running_task and self.running_task.priority != 'cutin':
                    self._interrupt_task(self.running_task, "被Cutin任务打断")
            elif task.priority == 'vip':
                # VIP逻辑：只打断Common任务
                if self.running_task and self.running_task.priority == 'common':
                    self._interrupt_task(self.running_task, "被VIP任务打断")
            
            # 添加到队列
            priority_score = Config.PRIORITY_MAP.get(task.priority, 10)
            self.tasks[task.id] = task
            self.queue.put((priority_score, task.created_at, task.id, task))
    
    def get_next_task(self):
        """从队列获取下一个任务"""
        try:
            priority_score, created_at, task_id, task = self.queue.get(block=False)
            return self.tasks.get(task_id)
        except queue.Empty:
            return None
    
    def set_running_task(self, task: Task):
        """设置当前运行的任务"""
        with self.lock:
            self.running_task = task
    
    def clear_running_task(self):
        """清除当前运行的任务"""
        with self.lock:
            self.running_task = None
    
    def get_task(self, task_id):
        """获取任务对象"""
        return self.tasks.get(task_id)
    
    def _interrupt_task(self, task: Task, reason: str):
        """
        中断任务（内部方法）
        
        参数:
            task: 要中断的任务
            reason: 中断原因
        """
        from core.process_manager import ProcessManager
        
        print(f"!!! 正在打断任务 {task.id}，原因: {reason} !!!")
        
        # 终止进程
        if task.process_info:
            ProcessManager.kill_process_tree(task.process_info)
        
        # 设置控制标志
        task.control['aborted'] = True
        
        # 根据优先级决定任务状态
        priority_score = Config.PRIORITY_MAP.get(task.priority, 10)
        if priority_score == Config.PRIORITY_MAP['cutin'] or task.priority == 'cutin':
            task.status = 'interrupted'
            task.message = "任务被打断，等待恢复"
        else:
            # 重新加入队列
            task.status = 'queued'
            task.message = "正在排队 (自动恢复中)..."
            task.control = {"aborted": False, "process": None}
            task.restarted = True
            priority_score = Config.PRIORITY_MAP.get(task.priority, 10)
            self.queue.put((priority_score, task.created_at, task.id, task))
        
        # 清除运行状态
        self.running_task = None
