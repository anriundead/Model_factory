
import json
import os

file_path = "/home/a/ServiceEndFiles/Workspaces/mergeKit_beta/merges/4479a54c/__home__a__ServiceEndFiles__Models__qwen2.5-7b-math/samples_gsm8k_2026-02-08T17-24-32.356927.jsonl"
try:
    with open(file_path, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        if 'resps' in data:
            # Print first 500 chars of generated text to avoid massive output
            resps = data['resps']
            print(f"Resps count: {len(resps)}")
            for i, r in enumerate(resps):
                 print(f"Resp {i} type: {type(r)}")
                 if isinstance(r, list):
                     for item in r:
                         print(f"  Item type: {type(item)}")
                         if isinstance(item, dict):
                             print(f"  Item keys: {item.keys()}")
                             if 'generated_text' in item:
                                 print(f"  Generated Text (first 200 chars): {item['generated_text'][:200]}")
                                 print(f"  Generated Text (last 200 chars): {item['generated_text'][-200:]}")
                         elif isinstance(item, str):
                             print(f"  Generated Text (first 200 chars): {item[:200]}")
                             print(f"  Generated Text (last 200 chars): {item[-200:]}")
        if 'filtered_resps' in data:
            print(f"Filtered Resps: {data['filtered_resps']}")
        
        # Also check doc answer to see what we are comparing against
        if 'doc' in data and 'answer' in data['doc']:
            print(f"Ground Truth: {data['doc']['answer']}")
            
except Exception as e:
    print(e)
