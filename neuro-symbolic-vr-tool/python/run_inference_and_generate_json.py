import glob
import logging


log_file = os.path.join(os.path.dirname(OUTPUT_JSON), "scene_pipeline.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Save latest scene JSON (overwrite clevr_scene.json)
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(objects, f, indent=2)
print(f"✅ Scene JSON saved to: {OUTPUT_JSON}")

# Save timestamped backup for debugging
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
debug_path = os.path.join(os.path.dirname(OUTPUT_JSON), f"clevr_scene_{timestamp}.json")
with open(debug_path, "w") as debug_f:
    json.dump(objects, debug_f, indent=2)
print(f"🧾 Debug scene snapshot saved to: {debug_path}")

# 🔄 Keep only the 20 most recent debug snapshots
snapshots = sorted(
    glob.glob(os.path.join(os.path.dirname(OUTPUT_JSON), "clevr_scene_*.json")),
    key=os.path.getmtime,
    reverse=True
)

for old_file in snapshots[20:]:
    os.remove(old_file)
    print(f"🗑️ Deleted old debug snapshot: {old_file}")
