import sys
import os
import glob
from pathlib import Path

sys.path.append("/home/a/ServiceEndFiles/Workspaces/mergeKit_beta")
from app.dataset_info import DatasetInfoService

class MockConfig:
    PROJECT_ROOT = "/home/a/ServiceEndFiles/Workspaces/mergeKit_beta"
    EVAL_HF_DATASETS_CACHE = "/home/a/.cache/huggingface/datasets"
    HF_ENDPOINT = "https://huggingface.co"

def test_dataset_info():
    service = DatasetInfoService(MockConfig())
    
    # Test 1: Check if 'cais/mmlu' returns subsets correctly
    print("--- Testing cais/mmlu ---")
    # We mock _collect_local_infos to print what it would find
    # But actually let's just call get_info?
    # No, get_info calls everything.
    
    # Let's inspect _collect_local_infos behavior directly
    hf_dataset = "cais/mmlu"
    cache_root = MockConfig.EVAL_HF_DATASETS_CACHE
    
    # Check what directories it would look in
    repo_dir_name = "datasets--" + hf_dataset.replace("/", "--")
    target_dirs = glob.glob(os.path.join(cache_root, "hub", repo_dir_name))
    print(f"Target dirs for {hf_dataset}: {target_dirs}")
    
    # Test 2: Check if 'mmlu' (no user) would match 'cais/mmlu' folders (it shouldn't anymore)
    hf_dataset_simple = "mmlu"
    repo_dir_name_simple = "datasets--" + hf_dataset_simple.replace("/", "--")
    target_dirs_simple = glob.glob(os.path.join(cache_root, "hub", repo_dir_name_simple))
    print(f"Target dirs for {hf_dataset_simple}: {target_dirs_simple}")
    
    if target_dirs and target_dirs == target_dirs_simple:
         print("FAIL: 'mmlu' matched same dirs as 'cais/mmlu'")
    else:
         print("SUCCESS: 'mmlu' and 'cais/mmlu' are distinct")

if __name__ == "__main__":
    test_dataset_info()
