import json
import re

def export_to_unity(symbolic_result_str, output_path="output/symbolic_action.json"):
    if not isinstance(symbolic_result_str, str):
        print(f"❌ Expected a string Prolog result, got {type(symbolic_result_str)}")
        return

    try:
        match = re.match(r"(\w+)\(([^)]+)\)", symbolic_result_str)
        if not match:
            print(f"❌ Invalid format: {symbolic_result_str}")
            return

        action = match.group(1)
        args = [arg.strip() for arg in match.group(2).split(",")]

        if action == "move" or action == "move_object":
            obj = args[0]
            dx, dy, dz = map(int, args[1:])

            direction_label = None
            if dx == 1: direction_label = "right"
            elif dx == -1: direction_label = "left"
            elif dy == 1: direction_label = "up"
            elif dy == -1: direction_label = "down"
            elif dz == 1: direction_label = "backward"
            elif dz == -1: direction_label = "forward"

            if direction_label is None:
                print(f"❌ Could not determine direction from ({dx}, {dy}, {dz})")
                return

            data = {
                "action": "move",
                "object": obj,
                "direction": [dx, dy, dz],
                "label": direction_label
            }

        elif action == "delete" or action == "delete_object":
            data = {
                "action": "delete",
                "object": args[0]
            }

        else:
            print(f"❌ Unsupported action: {action}")
            return

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
            print(f"✅ symbolic_action.json written: {json.dumps(data)}")

    except Exception as e:
        print(f"❌ Error exporting action: {e}")
