from inference_detectron import run_inference, setup_cfg, load_class_labels
from detectron2.engine import DefaultPredictor
import os

# Image file to use
image_path = "received_s.png"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found: {image_path}")

# Set up Detectron2
cfg = setup_cfg()
predictor = DefaultPredictor(cfg)
thing_classes = load_class_labels()

# Run inference and export scene.json
run_inference(image_path, predictor, thing_classes)
print(f"✅ scene.json saved to exported/{os.path.splitext(os.path.basename(image_path))[0]}_scene.json")
