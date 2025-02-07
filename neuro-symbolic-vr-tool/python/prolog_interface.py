import os
from pyswip import Prolog

# ✅ Get the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ Move to the project root (neuro-symbolic-vr-tool)
project_root = os.path.join(script_dir, "..")
os.chdir(project_root)  # Change working directory to the project root

# ✅ Set SWI-Prolog path explicitly
os.environ["SWI_PROLOG_PATH"] = "C:/Program Files/swipl/bin/swipl.exe"

prolog = Prolog()

# ✅ Use a Prolog-safe path
prolog_file = os.path.abspath("prolog/symbolic_rules.pl").replace("\\", "/")

print(f"🔹 Using Prolog file path: {prolog_file}")

try:
    # ✅ Pass the absolute file path to Prolog
    prolog.consult(prolog_file)
    print("✅ Prolog file loaded successfully!")

    # ✅ Run a test query (modify this based on your Prolog rules)
    result = list(prolog.query("some_fact(X)."))  # Replace with an actual fact/rule
    print("🧠 Query Result:", result)

except Exception as e:
    print(f"❌ Error loading Prolog file: {e}")
