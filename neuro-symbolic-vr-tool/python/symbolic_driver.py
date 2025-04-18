
import os
import time
from pyswip import Prolog
from export_action import export_to_unity
from send_to_unity import send_action_to_unity
from voice_module import get_voice_command
from run_detection_and_generate_json import run_inference_and_save

RULES_PATH = os.path.abspath("python/symbolic_module/rules.pl").replace("\\", "/")
SCENE_PATH = os.path.abspath("output/clevr_scene.json").replace("\\", "/")
JSON_PATH = "output/symbolic_action.json"

def run_symbolic_pipeline(command):
    if "now" in command:
        print("📸 Trigger word 'now' detected. Waiting for Unity screenshot...")
        time.sleep(2)  # Give Unity time to save the screenshot
        if not run_inference_and_save():
            print("❌ Failed to generate clevr_scene.json")
            return
    else:
        print("⚠️ 'now' not in command. Using existing scene.")

    print(f"🧠 Consulting rules from: {RULES_PATH}")
    prolog = Prolog()
    prolog.consult(RULES_PATH)

    print(f"📄 Loading scene from: {SCENE_PATH}")
    list(prolog.query(f"load_scene('{SCENE_PATH}')"))

    print(f"🔁 Running interpret on: '{command}'")
    result = list(prolog.query(f"interpret('{command}', Action)"))

    if not result:
        print("❌ No symbolic action found.")
        return

    action = result[0]["Action"]
    print(f"🎯 Symbolic Result: {action}")

    try:
        export_to_unity(action, output_path=JSON_PATH)
        send_action_to_unity(json_path=JSON_PATH)
    except Exception as e:
        print(f"⚠️ Could not send action to Unity: {e}")

def main():
    print("🎤 Listening for voice command...")
    voice_text = get_voice_command()
    if voice_text:
        print(f"🗣️ Recognized: {voice_text}")
        run_symbolic_pipeline(voice_text.lower())
    else:
        print("❌ No input detected.")

if __name__ == "__main__":
    main()
