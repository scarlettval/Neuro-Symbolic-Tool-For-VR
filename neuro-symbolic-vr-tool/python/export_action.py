import json

def export_to_unity(symbolic_result, output_path="output/symbolic_action.json"):
    if isinstance(symbolic_result, str):
        print(f"⚠️ Could not parse action properly: {symbolic_result}")
        return

    try:
        action = symbolic_result.functor
        args = list(symbolic_result.args)

        if action == "move_object":
            direction = args[1].value
            if direction not in ["left", "right", "up", "down"]:
                print(f"❌ Invalid direction: {direction}")
                return

            data = {
                "action": action,
                "object": args[0].value,
                "direction": direction
            }

        elif action == "create_object":
            data = {
                "action": action,
                "object": args[0].value,
                "properties": [arg.value for arg in args[1]]
            }

        elif action == "delete_object":
            data = {
                "action": action,
                "object": args[0].value
            }

        else:
            print(f"❌ Unsupported action: {action}")
            return

        with open(output_path, "w") as f:
            json.dump(data, f)
            print(f"✅ JSON written to {output_path}")

    except Exception as e:
        print(f"❌ Error exporting action: {e}")
