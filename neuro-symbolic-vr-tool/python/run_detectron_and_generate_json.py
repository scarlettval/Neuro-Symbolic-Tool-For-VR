# python/run_detection_and_generate_json.py

import torch
import json
import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes

INPUT_IMAGE = "Assets/python/vr_screenshot.png"
OUTPUT_JSON = "output/clevr_scene.json"
MODEL_PATH = "output/clevr_model_final.pth"

def get_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 24  # Adjust to match CLEVR classes
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

def run_inference_and_save():
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Screenshot not found: {INPUT_IMAGE}")
        return False

    predictor = get_predictor()
    image = cv2.imread(INPUT_IMAGE)
    outputs = predictor(image)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) if cfg.DATASETS.TRAIN else None
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else Boxes([])
    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else []

    objects = []
    for i in range(len(classes)):
        class_id = classes[i]
        label = metadata.thing_classes[class_id] if metadata else f"obj_{class_id}"
        tokens = label.split()  # Expected: "small red cube"
        if len(tokens) == 3:
            size, color, shape = tokens
            objects.append({
                "label": label,
                "size": size,
                "color": color,
                "shape": shape,
                "position": [0, 0, 0]  # Optional: later use bounding box center or custom logic
            })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(objects, f, indent=2)
    print(f"✅ Scene JSON saved to: {OUTPUT_JSON}")

    # ✅ ALSO save a timestamped backup for debugging
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = os.path.join(os.path.dirname(OUTPUT_JSON), f"clevr_scene_{timestamp}.json")
    with open(debug_path, "w") as debug_f:
        json.dump(objects, debug_f, indent=2)
    print(f"🧾 Debug scene snapshot saved to: {debug_path}")