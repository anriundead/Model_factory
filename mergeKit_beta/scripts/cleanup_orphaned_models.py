#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理遗留的中间模型目录脚本
用于清理已完成任务但未清理的 final_vlm 目录
"""
import os
import shutil
import json
import sys
from pathlib import Path

MERGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "merges")

def cleanup_orphaned_final_vlm():
    """清理遗留的 final_vlm 目录"""
    if not os.path.isdir(MERGE_DIR):
        print(f"合并目录不存在: {MERGE_DIR}")
        return
    
    cleaned_count = 0
    cleaned_size = 0
    skipped_count = 0
    
    for task_id in os.listdir(MERGE_DIR):
        task_dir = os.path.join(MERGE_DIR, task_id)
        if not os.path.isdir(task_dir):
            continue
        
        final_vlm_dir = os.path.join(task_dir, "final_vlm")
        if not os.path.isdir(final_vlm_dir):
            continue
        
        # 检查任务状态
        metadata_path = os.path.join(task_dir, "metadata.json")
        status = "unknown"
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                status = meta.get("status", "unknown")
            except Exception:
                pass
        
        # 检查是否有命名目录（表示任务已完成）
        has_named_dir = False
        for item in os.listdir(task_dir):
            if item.startswith("_") or item in ["final_vlm", "output", "metadata.json", "progress.json", "bridge.log"]:
                continue
            item_path = os.path.join(task_dir, item)
            if os.path.isdir(item_path) and not os.path.islink(item_path):
                # 检查是否是命名目录（包含时间戳格式）
                if "_202" in item or len(item) > 50:
                    has_named_dir = True
                    break
        
        # 计算目录大小
        try:
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, dirnames, filenames in os.walk(final_vlm_dir)
                      for filename in filenames)
            size_gb = size / (1024 ** 3)
        except Exception:
            size_gb = 0
        
        # 决定是否清理
        should_clean = False
        reason = ""
        
        if status == "success" and has_named_dir:
            should_clean = True
            reason = "任务已完成且有命名目录"
        elif status == "success":
            should_clean = True
            reason = "任务已完成（可能有遗留）"
        elif status == "error":
            should_clean = True
            reason = "任务失败"
        elif has_named_dir:
            should_clean = True
            reason = "存在命名目录（可能已完成）"
        elif size_gb > 1.0:  # 大于1GB的目录，即使状态未知也清理（可能是遗留）
            should_clean = True
            reason = f"大目录（{size_gb:.2f}GB，可能是遗留）"
        
        if should_clean:
            try:
                shutil.rmtree(final_vlm_dir, ignore_errors=True)
                cleaned_count += 1
                cleaned_size += size_gb
                print(f"✓ 已清理: {task_id} ({size_gb:.2f}GB) - {reason}")
            except Exception as e:
                print(f"✗ 清理失败: {task_id} - {e}")
        else:
            skipped_count += 1
            print(f"○ 跳过: {task_id} ({size_gb:.2f}GB) - status={status}, has_named_dir={has_named_dir}")
    
    print(f"\n清理完成:")
    print(f"  已清理: {cleaned_count} 个目录, 释放空间: {cleaned_size:.2f}GB")
    print(f"  跳过: {skipped_count} 个目录")

if __name__ == "__main__":
    print("开始清理遗留的 final_vlm 目录...")
    print(f"合并目录: {MERGE_DIR}\n")
    cleanup_orphaned_final_vlm()
