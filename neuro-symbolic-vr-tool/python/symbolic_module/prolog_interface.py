from pyswip import Prolog
import os

def send_to_prolog(command):
    print("Sending to symbolic reasoner:")
    print(command)

    prolog = Prolog()

    # Full absolute path with forward slashes
    rules_path = os.path.abspath(os.path.join("symbolic_module", "rules.pl"))
    rules_path = rules_path.replace("\\", "/")

    # Manually run the consult query using proper quoting
    consult_query = f"consult('{rules_path}')"
    print(f"Running Prolog: {consult_query}")

    try:
        list(prolog.query(consult_query))
        result = list(prolog.query(f"action({command})"))

        if result:
            print("✅ Prolog succeeded.")
        else:
            print("⚠️ Prolog ran but returned no output.")
    except Exception as e:
        print("❌ Error from Prolog:")
        print(e)
