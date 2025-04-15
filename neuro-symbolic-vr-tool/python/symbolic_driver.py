import os
from pyswip import Prolog
from datetime import datetime
from voice_module import get_voice_command
from export_action import export_to_unity
from send_to_unity import send_action_to_unity

# -------------------------------
# Configuration
# -------------------------------
RULES_PATH = os.path.abspath("python/symbolic_module/rules.pl")
JSON_PATH = os.path.join("output", "symbolic_action.json")

# -------------------------------
# Main Symbolic Reasoning Flow
# -------------------------------
def run_prolog_interpretation(voice_text):
    if not voice_text:
        print("❌ No voice input received.")
        return None

    prolog = Prolog()
    sanitized_text = voice_text.replace("'", "\\'")
    query = f"interpret('{sanitized_text}', Action)."

    try:
        escaped_path = RULES_PATH.replace("\\", "\\\\")
        print(f"🧠 Consulting Prolog rules from: {escaped_path}")
        list(prolog.query(f"consult('{escaped_path}')"))

        loaded = list(prolog.query("current_predicate(interpret/2)"))
        if not loaded:
            print("❌ interpret/2 is NOT loaded")
            return None
        print("✅ interpret/2 is confirmed to be loaded.")

    except Exception as e:
        print(f"❌ Failed during consult: {e}")
        return None

    try:
        print(f"🔁 Running Prolog Query: {query}")
        result = list(prolog.query(query))
        if result:
            value = result[0]["Action"]

            # If Prolog returned a string instead of a compound term
            if isinstance(value, str):
                print(f"⚠️ Got a string instead of compound: {value}")
                reparsed = list(prolog.query(f"Action = {value}."))
                if reparsed:
                    reparsed_value = reparsed[0]["Action"]
                    if hasattr(reparsed_value, "functor"):
                        return reparsed_value
                    else:
                        print("❌ reparsed_value is still not a compound term.")
                        return None
                else:
                    print("❌ Failed to reparse symbolic result.")
                    return None

            return value
        else:
            print("⚠️ interpret/2 returned false (no match)")
            return None
    except Exception as e:
        print(f"❌ Prolog query failed: {e}")
        return None

# -------------------------------
# Main Entry Point
# -------------------------------
def main():
    try:
        print(f"🎤 Listening for voice command... ({datetime.now()})")
        voice_text = get_voice_command()
    except Exception as e:
        print(f"⚠️ Microphone not available. Switching to typed input.\n{e}")
        voice_text = input("⌨️ Type your command instead: ")

    if voice_text:
        print(f"🗣️ Recognized: {voice_text}")
        symbolic_action = run_prolog_interpretation(voice_text)

        if symbolic_action and hasattr(symbolic_action, "functor"):
            print(f"🎯 Symbolic Result: {symbolic_action}")
            export_to_unity(symbolic_action, output_path=JSON_PATH)
            send_action_to_unity(json_path=JSON_PATH)
        else:
            print("⚠️ No valid symbolic action returned or could not be parsed.")
    else:
        print("❌ Voice recognition failed or no input provided.")

if __name__ == "__main__":
    main()
