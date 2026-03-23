#!/usr/bin/env python3
# 测试进化融合桥接脚本中 output 目录的安全移除逻辑：
# 当 output 是指向 named_dir 的符号链接时，只删除链接，不删除 named_dir。
import os
import shutil
import tempfile

def safe_remove_output_dir(output_dir):
    """与 run_vlm_search_bridge.py 中一致的安全移除逻辑。"""
    if os.path.lexists(output_dir):
        if os.path.islink(output_dir):
            os.unlink(output_dir)
        elif os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            try:
                os.remove(output_dir)
            except OSError:
                pass

def main():
    with tempfile.TemporaryDirectory() as tmp:
        merge_dir = os.path.join(tmp, "merges", "task1")
        os.makedirs(merge_dir, exist_ok=True)
        named_dir_name = "ModelA_ModelB_20260210"
        named_dir = os.path.join(merge_dir, named_dir_name)
        os.makedirs(named_dir, exist_ok=True)
        marker = os.path.join(named_dir, "fusion_info.json")
        with open(marker, "w") as f:
            f.write("{}")
        output_dir = os.path.join(merge_dir, "output")
        os.symlink(named_dir_name, output_dir)
        assert os.path.islink(output_dir), "output 应为符号链接"
        assert os.path.isdir(output_dir), "符号链接目标应为目录"
        safe_remove_output_dir(output_dir)
        assert not os.path.lexists(output_dir), "output 应已被移除"
        assert os.path.isdir(named_dir), "named_dir 必须仍然存在"
        assert os.path.isfile(marker), "named_dir 内文件必须仍在"
    print("OK: 安全移除逻辑测试通过（删除 output 链接未影响 named_dir）")

if __name__ == "__main__":
    main()
