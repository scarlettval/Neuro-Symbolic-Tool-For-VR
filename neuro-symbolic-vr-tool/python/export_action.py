import json
import re

def export_to_unity(symbolic_result_str, output_path="output/symbolic_action.json"):
    if not isinstance(symbolic_result_str, str):
        print(f"❌ Expected a string Prolog result, got {type(symbolic_result_str)}")
        return

    try:
        # Regex to extract action and arguments
        match = re.match(r"(\w+)\(([^)]+)\)", symbolic_result_str)
        if not match:
            print(f"❌ Invalid format: {symbolic_result_str}")
            return

        action = match.group(1)
        args = [arg.strip() for arg in match.group(2).split(",")]

        if action == "move" or action == "move_object":
            obj = args[0]
            dx, dy, dz = map(int, args[1:])

            # Infer direction from delta (supports only single-axis moves)
            direction = None
            if dx == 1: direction = "right"
            elif dx == -1: direction = "left"
            elif dy == 1: direction = "up"
            elif dy == -1: direction = "down"
            elif dz == 1: direction = "backward"
            elif dz == -1: direction = "forward"

            if not direction:
                print(f"❌ Could not determine direction from ({dx}, {dy}, {dz})")
                return

            data = {
                "object": obj,
                "direction": direction
            }

        elif action == "delete" or action == "delete_object":
            data = {
                "object": args[0],
                "direction": "none"
            }

        else:
            print(f"❌ Unsupported action: {action}")
            return

        with open(output_path, "w") as f:
            json.dump(data, f)
            print(f"✅ JSON written to {output_path}: {data}")

    except Exception as e:
        print(f"❌ Error exporting action: {e}")
