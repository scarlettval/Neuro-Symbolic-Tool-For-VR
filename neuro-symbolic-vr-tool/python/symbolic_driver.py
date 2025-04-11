import json
import os
from pyswip import Prolog
from voice_module import get_voice_command, parse_command_to_symbolic

def scene_to_prolog_facts(scene_file):
    with open(scene_file, "r") as f:
        data = json.load(f)

    facts = []
    for obj in data:
        label = obj["label"].replace(" ", "_")
        if label:
            facts.append(f"object({label})")
    return facts

def run_prolog_logic(scene_file, command):
    prolog = Prolog()

    # ✅ Absolute path to rules.pl with escaped slashes
    rules_path = os.path.abspath("python/symbolic_module/rules.pl").replace("\\", "\\\\")
    print(f"🧠 Consulting Prolog rules from: {rules_path}")

    try:
        list(prolog.query(f"consult('{rules_path}')"))
    except Exception as e:
        print(f"❌ Failed to consult rules.pl: {e}")
        return "consult_error"

    # ✅ Load valid scene facts
    facts = scene_to_prolog_facts(scene_file)
    for fact in facts:
        try:
            prolog.assertz(fact)
        except Exception as e:
            print(f"⚠️ Skipped fact: {fact} | Reason: {e}")

    query = f"{command}(Action)."
    try:
        result = list(prolog.query(query))
        return result[0]["Action"] if result else "no_action"
    except Exception as e:
        print(f"❌ Prolog query failed: {e}")
        return "query_error"

if __name__ == "__main__":
    scene_file = os.path.join("exported", "received_s_scene.json")

    if not os.path.exists(scene_file):
        print("❌ scene.json not found.")
        exit()

    voice_text = get_voice_command()
    symbolic_command = parse_command_to_symbolic(voice_text)

    if symbolic_command:
        symbolic_command = symbolic_command.replace("the_the", "the").replace("_to_to_", "_to_")
        print(f"🔁 Querying: {symbolic_command}(Action).")
        result = run_prolog_logic(scene_file, symbolic_command)
        print("🎯 Symbolic Result:", result)
    else:
        print("❌ Could not parse voice command into symbolic query.")
