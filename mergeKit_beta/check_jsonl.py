import json
import os

file_path = "/home/a/ServiceEndFiles/Workspaces/mergeKit_beta/merges/4479a54c/__home__a__ServiceEndFiles__Models__qwen2.5-7b-math/samples_gsm8k_2026-02-08T17-24-32.356927.jsonl"
try:
    with open(file_path, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        print("Keys:", data.keys())
        if 'resps' in data:
            print("Response:", data['resps'])
        if 'filtered_resps' in data:
            print("Filtered Response:", data['filtered_resps'])
except Exception as e:
    print(e)
