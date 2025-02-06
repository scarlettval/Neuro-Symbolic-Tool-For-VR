from pyswip import Prolog

prolog = Prolog()

# Ensure the correct Prolog file path with DOUBLE BACKSLASHES
prolog_file = "C:\\Users\\mjhen\\OneDrive\\Documents\\Spring 2025\\EGN 4952C\\Symbolic-Tool-For-Virtual-Reality\\neuro-symbolic-vr-tool\\prolog\\symbolic_rules.pl"

print(f"🔹 Using Prolog file path: {prolog_file}")

try:
    # Set the working directory explicitly
    prolog_directory = "c:\\Users\\mjhen\\OneDrive\\Documents\\Spring 2025\\EGN 4952C\\Symbolic-Tool-For-Virtual-Reality\\neuro-symbolic-vr-tool\\prolog"
    prolog.query(f"working_directory(_, '{prolog_directory.replace('\\', '\\\\')}').")

    # Load the Prolog file (ensure proper escaping)
    prolog.consult(prolog_file.replace("\\", "\\\\"))  # Double escaping for Prolog

    print("✅ Prolog file loaded successfully!")

except Exception as e:
    print(f"❌ Error loading Prolog file: {e}")
