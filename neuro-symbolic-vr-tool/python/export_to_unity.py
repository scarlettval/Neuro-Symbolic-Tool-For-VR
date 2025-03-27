import os
import json

INPUT_PATH = os.path.join("SymbolicLogs", "symbolic_log.json")
OUTPUT_PATH = os.path.join("UnityProject", "symbolic_actions_unity.json")

def export_for_unity():
    if not os.path.exists(INPUT_PATH):
        print("❌ No symbolic log file found.")
        return

    with open(INPUT_PATH, "r") as f:
        logs = json.load(f)

    unity_actions = []

    for entry in logs:
        command = entry["command"]
        timestamp = entry["timestamp"]

        # Example: create(cube) → type: create, target: cube
        if "(" in command and ")" in command:
            action_type = command.split("(")[0]
            params = command.split("(")[1].split(")")[0]

            unity_actions.append({
                "action": action_type,
                "target": params,
                "timestamp": timestamp
            })

    os.makedirs("UnityProject", exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(unity_actions, f, indent=2)

    print(f"✅ Exported {len(unity_actions)} actions to Unity format:")
    print(f"📄 {OUTPUT_PATH}")

if __name__ == "__main__":
    export_for_unity()
