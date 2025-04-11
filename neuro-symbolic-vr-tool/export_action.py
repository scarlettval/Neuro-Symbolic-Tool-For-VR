import re
import json

# Simulated input from Prolog — replace this string if reading from output
symbolic_result = "move_object(small_red_cube, left)"

# Parse the action
pattern = r"(\w+)\(([^,]+),\s*([^)]+)\)"
match = re.match(pattern, symbolic_result)

if match:
    action_type, obj, direction = match.groups()

    export_data = {
        "action": action_type,
        "object": obj,
        "direction": direction
    }

    with open("exported/action.json", "w") as f:
        json.dump(export_data, f, indent=2)

    print("✅ Exported action.json:")
    print(json.dumps(export_data, indent=2))
else:
    print("❌ Failed to parse symbolic result.")
