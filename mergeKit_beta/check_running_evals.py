
import sys
import os

# Add project root to path
sys.path.append("/home/a/ServiceEndFiles/Workspaces/mergeKit_beta")

from app.state import AppState
from app.services import Services
from config import Config

Config.setup_environment()
state = AppState()
services = Services(state)

history = services.get_all_eval_history()
for h in history:
    if h.get("status") == "running":
        print(f"RUNNING TASK FOUND: {h['id']} type={h.get('type')}")
    elif h.get("status") not in ["success", "error", "stopped", "completed"]:
        print(f"NON-TERMINAL TASK: {h['id']} status={h.get('status')} type={h.get('type')}")
