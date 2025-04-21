# python/run_detection_and_generate_json.py

import os
import json
import torch
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes
import numpy as np

# Paths
MODEL_PATH = os.path.abspath("output/clevr_model_final.pth")
LABELS_PATH = os.path.abspath("output/clevr_labels.json")

def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(load_class_labels())
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

def load_class_labels():
    with open(LABELS_PATH, "r") as f:
        return json.load(f)

def run_inference_and_save(image_path, output_json_path):
    print(f"📸 Running object detection on: {image_path}")
    predictor = setup_predictor()
    image = np.array(Image.open(image_path).convert("RGB"))

    outputs = predictor(image)
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()

    class_labels = load_class_labels()
    result = []

    for box, cls in zip(boxes, classes):
        label = class_labels[str(cls)]
        obj = {
            "label": label,
            "size": label.split()[0],
            "color": label.split()[1],
            "shape": label.split()[2],
            "position": estimate_position_from_box(box)
        }
        result.append(obj)

    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=2)
        print(f"✅ Scene saved to {output_json_path}")

def estimate_position_from_box(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return [round(cx / 100, 2), 0, round(cy / 100, 2)]  # Simulated (x, y, z)

