import os
import json
import datetime
from pyswip import Prolog
from voice_module import get_voice_command
from export_action import export_to_unity
from send_to_unity import send_action_to_unity



CLEVR_JSON_PATH = os.path.abspath("output/clevr_scene.json")
RULES_PATH = os.path.abspath("python/symbolic_module/rules.pl")
SNAPSHOT_PATH = os.path.abspath("output/vr_snapshot.png")
JSON_PATH = os.path.abspath("output/symbolic_action.json")


def run_symbolic_pipeline(command_text):
    prolog = Prolog()
    escaped_rules_path = RULES_PATH.replace("\\", "/")
    escaped_json_path = CLEVR_JSON_PATH.replace("\\", "/")

    print("[INFO] Consulting rules from:", escaped_rules_path)
    list(prolog.query(f"consult('{escaped_rules_path}')"))

    print("[INFO] Loading scene from:", escaped_json_path)
    list(prolog.query(f"load_scene('{escaped_json_path}')"))

    print("[INFO] Running interpret on:", command_text)
    result = list(prolog.query(f"interpret('{command_text}', Action)"))

    if not result:
        print("[WARNING] No valid symbolic action returned.")
        return

    action_term = result[0]["Action"]
    print("[RESULT] Symbolic Result:", action_term)

    try:
        action_str = str(action_term)
        if "(" in action_str and ")" in action_str:
            inner = action_str[action_str.index("(")+1:action_str.index(")")]
            parts = inner.split(",")
            if "move" in action_str and len(parts) == 4:
                obj, dx, dy, dz = [p.strip() for p in parts]
                print(f"[INFO] Executing: move_object({obj}, {dx}, {dy}, {dz})")
                list(prolog.query(f"move_object({obj}, {dx}, {dy}, {dz})"))
            else:
                print("[WARNING] Unsupported or malformed action.")
        else:
            print("[WARNING] Could not parse action properly.")

        export_to_unity(action_term, output_path=JSON_PATH)
        send_action_to_unity(json_path=JSON_PATH)

    except Exception as e:
        print("[ERROR] Error handling symbolic action:", e)


def main():
    print("[INFO] Listening for voice command...", datetime.datetime.now())
    voice_text = get_voice_command()

    if voice_text is None:
        print("[ERROR] No command detected.")
        return

    print("[INFO] Recognized:", voice_text)

    if "now" in voice_text.lower():
        print("[INFO] Trigger word 'now' detected. Capturing scene snapshot...")
        # Uncomment when ready to use inference
        # run_inference_and_save(SNAPSHOT_PATH, CLEVR_JSON_PATH)
        voice_text = voice_text.lower().replace("now", "").strip()

    if voice_text:
        run_symbolic_pipeline(voice_text)


if __name__ == "__main__":
    main()
