from pyswip import Prolog
import os

def query_prolog():
    prolog = Prolog()

    # Get the absolute path and replace ALL backslashes with forward slashes
    prolog_file = os.path.abspath("prolog/symbolic_rules.pl").replace("\\", "/")

    # Ensure Prolog receives a correctly formatted path
    prolog_command = f"consult('{prolog_file}')"

    print(f"🔹 Using Prolog file path: {prolog_file}")

    try:
        # Load the Prolog file
        prolog.consult(prolog_file)
        print("✅ Prolog file loaded successfully.")

        # Test query
        results = list(prolog.query("ancestor(X, alice)."))

        for result in results:
            print(f"Ancestor of Alice: {result['X']}")

    except Exception as e:
        print(f"❌ Error loading Prolog file: {e}")

if __name__ == "__main__":
    query_prolog()
