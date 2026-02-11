
import sys
import os
sys.path.insert(0, os.getcwd())

def check():
    try:
        from app import create_app
        app = create_app()
        print("Create app success")
        # Check routes
        for rule in app.url_map.iter_rules():
            if "mmlu_subset_groups" in str(rule):
                print(f"Found rule: {rule}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check()
