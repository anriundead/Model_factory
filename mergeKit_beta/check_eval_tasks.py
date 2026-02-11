
import os
import json

MERGE_DIR = "/home/a/ServiceEndFiles/Workspaces/mergeKit_beta/merges"

print(f"Checking {MERGE_DIR} for running eval tasks...")

if not os.path.exists(MERGE_DIR):
    print("MERGE_DIR does not exist")
    exit()

for tid in os.listdir(MERGE_DIR):
    task_path = os.path.join(MERGE_DIR, tid)
    if not os.path.isdir(task_path):
        continue
    
    meta_path = os.path.join(task_path, "metadata.json")
    if not os.path.exists(meta_path):
        continue
        
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        task_type = meta.get("type")
        status = meta.get("status")
        
        if task_type == "eval_only":
            print(f"Task {tid}: type={task_type}, status={status}")
            if status == "running":
                print(f"FOUND STUCK TASK: {tid}")
    except Exception as e:
        print(f"Error reading {tid}: {e}")
