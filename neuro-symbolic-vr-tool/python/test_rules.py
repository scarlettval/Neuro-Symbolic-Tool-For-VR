import os
from pyswip import Prolog

RULES_PATH = os.path.abspath("python/symbolic_module/rules.pl").replace("\\", "/")
SCENE_JSON = os.path.abspath("output/clevr_scene.json").replace("\\", "/")

def main():
    print("🧠 Consulting Prolog rules from:", RULES_PATH)
    p = Prolog()
    try:
        list(p.query(f"consult('{RULES_PATH}')"))
        print("✅ rules.pl consulted successfully.")
    except Exception as e:
        print("❌ Failed to consult rules.pl:", e)
        return

    print("📄 Loading scene manually from JSON:", SCENE_JSON)
    try:
        result = list(p.query(f"rules:load_scene('{SCENE_JSON}')"))
        print("✅ Scene loaded successfully:", result)
    except Exception as e:
        print("❌ Failed to load scene:", e)
        return

    print("\n📎 Calling list_objects from Prolog:")
    try:
        list(p.query("rules:list_objects"))
    except Exception as e:
        print("❌ list_objects failed:", e)

    print("\n📦 Prolog Objects in Memory BEFORE interpret:")
    try:
        for r in p.query("rules:object(ID, Color, Shape, Material, Size, Pos)"):
            print(f"  - {r}")
    except Exception as e:
        print("❌ Could not query rules:object/6 facts:", e)

    cmd = "move the small red cube to the left"
    print(f"\n🔁 Querying interpret for: {cmd}")
    try:
        result = list(p.query(f"rules:interpret('{cmd}', Action)"))
        print("🎯 interpret result:", result)
    except Exception as e:
        print("❌ interpret/2 query failed:", e)

    print("\n📦 Prolog Objects in Memory AFTER interpret:")
    try:
        for r in p.query("rules:object(ID, Color, Shape, Material, Size, Pos)"):
            print(f"  - {r}")
    except Exception as e:
        print("❌ Could not query rules:object/6 facts:", e)

    print("\n🔁 Trying rules:move_object(small_red_cube, -1, 0, 0)...")
    try:
        result = list(p.query("rules:move_object(small_red_cube, -1, 0, 0)"))
        print("✅ move_object result:", result)
    except Exception as e:
        print("❌ move_object/4 failed:", e)

if __name__ == "__main__":
    main()
