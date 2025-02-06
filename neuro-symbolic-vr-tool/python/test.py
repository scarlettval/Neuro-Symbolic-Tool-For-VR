import os

# Ensure we only join the path once
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the correct base directory
prolog_file = os.path.join(base_dir, "../prolog/symbolic_rules.pl")  # Go up one level to "prolog"
prolog_file = os.path.abspath(prolog_file).replace("\\", "/")  # Convert to absolute path with forward slashes

print(f"🔹 Debug: {prolog_file}")
