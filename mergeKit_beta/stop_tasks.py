import os
import json
import time

MERGE_DIR = "/home/a/ServiceEndFiles/Workspaces/mergeKit_beta/merges"

def stop_stuck_tasks():
    if not os.path.exists(MERGE_DIR):
        print(f"Directory {MERGE_DIR} does not exist.")
        return

    count = 0
    for task_id in os.listdir(MERGE_DIR):
        task_path = os.path.join(MERGE_DIR, task_id)
        if not os.path.isdir(task_path):
            continue

        # Check progress.json
        progress_path = os.path.join(task_path, "progress.json")
        updated = False
        if os.path.isfile(progress_path):
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                if not content:
                    # Empty file, ignore or init
                    continue
                    
                data = json.loads(content)
                
                if data.get("status") == "running":
                    # Check if process is actually running? 
                    # For now just trust it's stuck if user asked to run this script.
                    # Or check last modification time?
                    mtime = os.path.getmtime(progress_path)
                    if time.time() - mtime > 3600: # 1 hour timeout
                        print(f"Stopping stuck task {task_id} in progress.json (last update > 1h ago)")
                        data["status"] = "stopped"
                        data["message"] = "Task stopped by system maintenance (timeout/stuck)"
                        with open(progress_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        updated = True
                        count += 1
            except json.JSONDecodeError as e:
                print(f"Corrupted progress.json for task {task_id}: {e}")
                # Optional: Force stop if corrupted?
                # For now just log it.
            except Exception as e:
                print(f"Error reading/writing {progress_path}: {e}")

        # Check metadata.json
        meta_path = os.path.join(task_path, "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                if meta.get("status") == "running":
                    print(f"Stopping stuck task {task_id} in metadata.json")
                    meta["status"] = "stopped"
                    meta["error"] = "Task stopped by system maintenance (timeout/stuck)"
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    updated = True
            except Exception as e:
                print(f"Error reading/writing {meta_path}: {e}")

    print(f"Stopped {count} stuck tasks.")

if __name__ == "__main__":
    stop_stuck_tasks()
