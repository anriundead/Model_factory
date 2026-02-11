import sys
import os
import json
import argparse
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from merge_manager import run_yaml_eval_stream
except ImportError:
    # Fallback if run directly from scripts folder (not the case here, but safe)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from merge_manager import run_yaml_eval_stream

def worker_callback(prog, msg):
    # Print progress in a structured format for parent to parse
    # Format: @PROG@prog|msg
    # Ensure msg doesn't contain newlines that break parsing
    safe_msg = msg.replace("\n", " ")
    print(f"@PROG@{prog}|{safe_msg}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--hf_dataset", default=None)
    parser.add_argument("--hf_subset", default=None)
    parser.add_argument("--hf_split", default=None)
    parser.add_argument("--prompt_yaml", default=None)
    parser.add_argument("--limit", default="0.5")
    
    args = parser.parse_args()
    
    try:
        # run_yaml_eval_stream signature:
        # (model_path, output_path, callback, start_prog, end_prog, task_control, ...)
        
        metrics = run_yaml_eval_stream(
            model_path=args.model_path,
            output_path=args.output_path,
            callback=worker_callback,
            start_prog=0,
            end_prog=100,
            task_control={}, # No control object passed, rely on process kill
            hf_dataset=args.hf_dataset if args.hf_dataset != "None" else None,
            hf_subset=args.hf_subset if args.hf_subset != "None" else None,
            hf_split=args.hf_split if args.hf_split != "None" else None,
            prompt_yaml_path=args.prompt_yaml if args.prompt_yaml != "None" else None,
            limit=args.limit
        )
        
        # Output result
        print("@RESULT@" + json.dumps(metrics), flush=True)
        
    except Exception as e:
        # Print error with traceback
        trace = traceback.format_exc()
        # Escape newlines for safer parsing if needed, but here we just print
        print(f"@ERROR@{str(e)}\n{trace}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
