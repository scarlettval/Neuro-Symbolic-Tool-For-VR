from pyswip import Prolog
import os

def send_to_prolog(command):
    print("Sending to symbolic reasoner:")
    print(command)

    prolog = Prolog()

    # Get absolute, forward-slashed path to rules.pl
    rules_path = os.path.abspath(os.path.join("symbolic_module", "rules.pl"))
    rules_path = rules_path.replace("\\", "/")  # Make sure it's forward slashes

    try:
        # Tell Prolog to consult this path
        prolog.consult(f"{rules_path}")
        result = list(prolog.query(f"action({command})"))

        if result:
            print("✅ Prolog succeeded.")
        else:
            print("⚠️ Prolog ran but returned no output.")
    except Exception as e:
        print("❌ Error from Prolog:")
        print(e)
