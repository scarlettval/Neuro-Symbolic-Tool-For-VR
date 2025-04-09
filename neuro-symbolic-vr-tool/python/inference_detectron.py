import cv2
import os
import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

# --- Config ---
model_weights = "../output/model_final.pth"
image_path = "../data/CLEVR_v1.0/images/test/CLEVR_test_000000.png"
output_json_path = "../data/demo_output/scene.json"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Set this to your number of CLEVR classes
cfg.MODEL.WEIGHTS = model_weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

im = cv2.imread(image_path)
outputs = predictor(im)

instances = outputs["instances"].to("cpu")

# --- Generate symbolic JSON ---
symbolic_objects = []

for box, label in zip(instances.pred_boxes, instances.pred_classes):
    x1, y1, x2, y2 = box.tensor[0].tolist()
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2

    label_str = f"class_{label.item()}"  # Replace with your class-to-name map if needed
    symbolic_objects.append({
        "type": label_str,
        "color": "unknown",  # Fill in if your model predicts color
        "position": [round(x, 2), round(y, 2)]
    })

with open(output_json_path, "w") as f:
    json.dump({"objects": symbolic_objects}, f, indent=2)

print(f"🔁 Inference complete. Output saved to: {output_json_path}")
