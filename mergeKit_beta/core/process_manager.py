"""
进程管理模块 - 统一进程创建和终止逻辑
"""
import os
import signal
import subprocess


class ProcessManager:
    """统一的进程管理类"""
    
    @staticmethod
    def create_process_group_kwargs() -> dict:
        """
        创建进程组参数（跨平台）
        
        返回:
            dict: 用于 subprocess.Popen 的参数
        """
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}
    
    @staticmethod
    def kill_process_tree(proc_or_pid):
        """
        终止进程树（支持Popen对象或PID）
        
        参数:
            proc_or_pid: subprocess.Popen 对象或进程ID (int)
        """
        if proc_or_pid is None:
            return
        
        # 获取PID和进程对象
        if isinstance(proc_or_pid, subprocess.Popen):
            pid = proc_or_pid.pid
            proc_obj = proc_or_pid  # 保存对象用于兜底
        else:
            pid = proc_or_pid
            proc_obj = None
        
        try:
            if os.name == "nt":
                # Windows: 使用 taskkill 终止进程树
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )
            else:
                # Linux: 使用 killpg 终止进程组
                os.killpg(pid, signal.SIGKILL)
        except Exception as e:
            # 兜底：如果有Popen对象，尝试直接kill
            if proc_obj:
                try:
                    proc_obj.kill()
                except Exception:
                    pass
            print(f"Kill process tree failed: {e}")
