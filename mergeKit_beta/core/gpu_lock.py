from __future__ import annotations

import os
import time


def lock_file(path: str, timeout_s: int) -> object:
    """
    简单的文件锁（POSIX flock）。
    返回文件句柄对象，调用方负责 close() 释放锁。
    """
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
            return f
        except BlockingIOError:
            if time.time() >= deadline:
                f.close()
                raise TimeoutError(f"GPU 锁等待超时: {path}")
            time.sleep(0.5)
