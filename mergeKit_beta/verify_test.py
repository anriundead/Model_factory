import requests
import time
import json

BASE_URL = "http://localhost:5000"
MODEL_PATH = "/home/a/ServiceEndFiles/Models/Qwen2.5-0.5B-Instruct"
TESTSET_ID = "mmlu-default-restore"

def check_testsets():
    print("Checking testsets...")
    r = requests.get(f"{BASE_URL}/api/testset/list?refresh=1")
    data = r.json()
    testsets = data.get("testsets", [])
    found = False
    for t in testsets:
        if t.get("testset_id") == TESTSET_ID:
            print(f"FOUND target testset: {t['name']}")
            found = True
            break
    if not found:
        print("WARNING: Target testset not found in list!")
        print("Available:", [t.get("name") for t in testsets])
    return found

def submit_task():
    print(f"Submitting task for model {MODEL_PATH}...")
    payload = {
        "model_path": MODEL_PATH,
        "testset_id": TESTSET_ID,
        "limit": 5,  # Minimal scale
        "model_name": "Qwen2.5-0.5B-Test"
    }
    r = requests.post(f"{BASE_URL}/api/evaluate", json=payload)
    if r.status_code != 200:
        print(f"FAILED to submit: {r.text}")
        return None
    task_id = r.json().get("task_id")
    print(f"Task submitted. ID: {task_id}")
    return task_id

def monitor_task(task_id):
    print("Monitoring task...")
    start_time = time.time()
    while True:
        # Check specific status
        r = requests.get(f"{BASE_URL}/api/status/{task_id}")
        if r.status_code != 200:
            print(f"Error getting status: {r.status_code}")
            break
        status_data = r.json()
        status = status_data.get("status")
        progress = status_data.get("eval_progress", {})
        
        # Check history visibility (simulating frontend recovery)
        h_r = requests.get(f"{BASE_URL}/api/test_history")
        history_list = h_r.json().get("history", [])
        in_history = any(t.get("id") == task_id and t.get("status") in ["running", "queued"] for t in history_list)
        
        print(f"Status: {status} | Progress: {progress.get('percent', 0)}% | HistoryVisible: {in_history}")
        
        if status in ["success", "failed", "stopped", "error", "completed"]:
            print(f"Task finished with status: {status}")
            if status in ["success", "completed"]:
                # Print result snippet
                res_r = requests.get(f"{BASE_URL}/api/history/{task_id}")
                if res_r.status_code == 200:
                    res = res_r.json().get("data", {})
                    print("Metrics:", res.get("metrics"))
            break
        
        if time.time() - start_time > 300: # 5 min timeout
            print("Timeout waiting for task")
            break
            
        time.sleep(2)

if __name__ == "__main__":
    if check_testsets():
        tid = submit_task()
        if tid:
            monitor_task(tid)
