import os
import json
from datetime import datetime

LOG_PATH = os.path.join("symbolicLogs", "symbolic_log.json")


def save_symbolic_command(command, source="pipeline"):
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "command": command,
        "source": source
    }

    os.makedirs("logs", exist_ok=True)

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"📝 Logged command: {command}")
