import os
import json
import time
import datetime
from pyswip import Prolog
from voice_module import get_voice_command
from export_action import export_to_unity
from send_to_unity import send_action_to_unity
from run_detection_and_generate_json import run_inference_and_save

BASE_DIR = os.path.abspath(".")
RULES_PATH = os.path.join(BASE_DIR, "python", "symbolic_module", "rules.pl")
CLEVR_JSON_PATH = os.path.join(BASE_DIR, "output", "clevr_scene.json")
SNAPSHOT_PATH = os.path.abspath("RealUnityFolder/3DEnvironment/Assets/Snapshots/vr_snapshot.png")
JSON_PATH = os.path.join(BASE_DIR, "output", "symbolic_action.json")

# === Prolog Setup ===
def run_symbolic_pipeline(command_text):
    prolog = Prolog()
    print(f"[INFO] Consulting rules from: {RULES_PATH}")
    list(prolog.query(f"consult('{RULES_PATH.replace(os.sep, '/')}')"))

    print(f"[INFO] Loading scene from: {CLEVR_JSON_PATH}")
    list(prolog.query(f"load_scene('{CLEVR_JSON_PATH.replace(os.sep, '/')}')"))

    print(f"[INFO] Running interpret on: {command_text}")
    result = list(prolog.query(f"interpret('{command_text}', Action)"))

    if not result:
        print("[WARN] No valid symbolic action returned.")
        return

    action_term = result[0]["Action"]
    print(f"[RESULT] Symbolic Result: {action_term}")

    try:
        # Attempt to execute the symbolic action
        if "move" in str(action_term):
            inner = str(action_term)[str(action_term).index("(")+1:-1]
            obj, dx, dy, dz = [v.strip() for v in inner.split(",")]
            print(f"[INFO] Executing: move_object({obj}, {dx}, {dy}, {dz})")
            list(prolog.query(f"move_object({obj}, {dx}, {dy}, {dz})"))

        export_to_unity(action_term, output_path=JSON_PATH)
        send_action_to_unity(json_path=JSON_PATH)

    except Exception as e:
        print(f"[ERROR] Failed during symbolic pipeline: {e}")

# === Voice + Trigger Handler ===
def main():
    print("[INFO] Listening for voice command...", datetime.datetime.now())
    voice_text = get_voice_command()

    if voice_text is None:
        print("[ERROR] No command detected.")
        return

    print(f"[INFO] Recognized: {voice_text}")

    # Handle "now" trigger
    if "now" in voice_text.lower():
        print("[INFO] Trigger word 'now' detected. Capturing scene snapshot...")
        try:
            # Send Unity a screenshot command
            screenshot_payload = {"screenshot": True}
            with open(JSON_PATH, "w") as f:
                json.dump(screenshot_payload, f)
            send_action_to_unity(json_path=JSON_PATH)
        except Exception as e:
            print(f"[ERROR] Failed to notify Unity to take screenshot: {e}")
            return

        # Wait for Unity to save screenshot
        for _ in range(20):
            if os.path.exists(SNAPSHOT_PATH):
                print(f"[INFO] Found screenshot: {SNAPSHOT_PATH}")
                break
            print("[INFO] Waiting for Unity to write snapshot...")
            time.sleep(0.5)
        else:
            print("[ERROR] Screenshot was never created.")
            return

        # Run detection + generate new clevr_scene.json
        run_inference_and_save(SNAPSHOT_PATH, CLEVR_JSON_PATH)

        # Remove 'now' from command and continue
        voice_text = voice_text.lower().replace("now", "").strip()

    # Run full symbolic processing
    if voice_text:
        run_symbolic_pipeline(voice_text)

if __name__ == "__main__":
    main()
