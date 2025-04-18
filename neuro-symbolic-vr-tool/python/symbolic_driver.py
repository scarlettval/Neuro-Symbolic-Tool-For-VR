""import os
import json
import time
from pyswip import Prolog
from voice_module import get_voice_command
from export_action import export_to_unity
from send_to_unity import send_action_to_unity, send_custom_command

SCENE_JSON = os.path.abspath("output/clevr_scene.json")
RULES_PATH = os.path.abspath("python/symbolic_module/rules.pl")
JSON_PATH = os.path.abspath("output/symbolic_action.json")

def consult_rules(prolog):
    try:
        print(f"\U0001f9e0 Consulting rules from: {RULES_PATH}")
        prolog.consult(RULES_PATH)
        return True
    except Exception as e:
        print(f"❌ Failed to consult rules: {e}")
        return False

def load_scene(prolog):
    try:
        print(f"\U0001f4c4 Loading scene from: {SCENE_JSON}")
        list(prolog.query(f"load_scene('{SCENE_JSON}')"))
        return True
    except Exception as e:
        print(f"❌ Failed to load scene: {e}")
        return False

def interpret_command(prolog, command):
    try:
        query = f"interpret('{command}', Action)"
        print(f"\U0001f501 Running interpret on: '{command}'")
        result = list(prolog.query(query))
        return result[0]["Action"] if result else None
    except Exception as e:
        print(f"❌ Could not interpret command: {e}")
        return None

def run_action(prolog, action):
    try:
        if isinstance(action, str):
            print(f"⚠️ Got string instead of Prolog term: {action}")
            return action

        functor = action.name
        args = [str(a) for a in action.args]
        goal = f"{functor}({', '.join(args)})"
        print(f"✅ Action executed in Prolog: {goal}")
        list(prolog.query(goal))
        return action
    except Exception as e:
        print(f"❌ Could not run action in Prolog: {e}")
        return None

def trigger_screenshot():
    print("\U0001f4f8 Sending screenshot trigger to Unity...")
    send_custom_command({"action": "take_screenshot"})

def run_symbolic_pipeline(voice_text):
    if not voice_text:
        print("❌ No voice input detected.")
        return

    voice_text = voice_text.lower().strip()

    if voice_text.endswith("now"):
        trigger_screenshot()
        return

    prolog = Prolog()
    if not consult_rules(prolog):
        return
    if not load_scene(prolog):
        return

    action = interpret_command(prolog, voice_text)
    if not action:
        print("⚠️ No valid symbolic action returned.")
        return

    print(f"\U0001f3af Symbolic Result: {action}")
    executed = run_action(prolog, action)
    if not executed:
        print(f"⚠️ Could not parse action properly: {action}")
        return

    export_to_unity(executed, output_path=JSON_PATH)
    send_action_to_unity(json_path=JSON_PATH)

def main():
    print(">>\n🎤 Listening for voice command...")
    voice_text = get_voice_command()
    print(f"🗣️ Recognized: {voice_text}")
    run_symbolic_pipeline(voice_text)

if __name__ == "__main__":
    main()
