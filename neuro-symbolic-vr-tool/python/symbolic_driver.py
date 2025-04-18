import os
from pyswip import Prolog
from voice_module import get_voice_command
from export_action import export_to_unity
from send_to_unity import send_action_to_unity

RULES_PATH = os.path.abspath("python/symbolic_module/rules.pl").replace("\\", "/")
SCENE_JSON = os.path.abspath("output/clevr_scene.json").replace("\\", "/")
JSON_PATH = "output/symbolic_action.json"

def consult_and_load_scene(p):
    print("🧠 Consulting rules from:", RULES_PATH)
    list(p.query(f"consult('{RULES_PATH}')"))

    print("📄 Loading scene from:", SCENE_JSON)
    list(p.query(f"rules:load_scene('{SCENE_JSON}')"))

def run_symbolic_pipeline(command_str):
    p = Prolog()
    consult_and_load_scene(p)

    print(f"🔁 Running interpret on: '{command_str}'")
    result = list(p.query(f"rules:interpret('{command_str}', Action)"))
    filtered = [r for r in result if r["Action"] != "unknown_command"]
    if not filtered:
        print("⚠️ No valid symbolic action returned.")
        return

    raw_action = filtered[0]["Action"]
    print("🎯 Symbolic Result:", raw_action)

    # Remap symbolic to actual predicate
    if raw_action.startswith("move("):
        action_query = raw_action.replace("move", "move_object", 1)
    elif raw_action.startswith("delete("):
        action_query = raw_action.replace("delete", "delete_object", 1)
    else:
        action_query = raw_action

    try:
        list(p.query(f"rules:{action_query}"))
        print("✅ Action executed in Prolog:", action_query)

        # Export and Unity
        export_to_unity(raw_action, output_path=JSON_PATH)
        send_action_to_unity(json_path=JSON_PATH)

    except Exception as e:
        print("❌ Could not run action in Prolog:", e)

def main():
    try:
        print("🎤 Listening for voice command...")
        voice_text = get_voice_command()
    except Exception as e:
        print(f"⚠️ Microphone not available. Switching to typed input.\n{e}")
        voice_text = input("⌨️ Type your command instead: ")

    if voice_text:
        print("🗣️ Recognized:", voice_text)
        run_symbolic_pipeline(voice_text)
    else:
        print("❌ Voice recognition failed or no input provided.")

if __name__ == "__main__":
    main()
