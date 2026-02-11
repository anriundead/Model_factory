
import os
import json

task_dir = "/home/a/ServiceEndFiles/Workspaces/mergeKit_beta/merges/4479a54c"
candidates = []
for root, _, files in os.walk(task_dir):
    for fn in files:
        if fn.endswith(".json") and fn not in ("metadata.json", "progress.json"):
            candidates.append(os.path.join(root, fn))

if not candidates:
    print("No result files found.")
else:
    latest = max(candidates, key=lambda p: os.path.getmtime(p))
    print(f"Found result file: {latest}")
    try:
        with open(latest, 'r') as f:
            data = json.load(f)
        print("Results:")
        print(json.dumps(data.get("results", {}), indent=2))
    except Exception as e:
        print(f"Error reading file: {e}")
