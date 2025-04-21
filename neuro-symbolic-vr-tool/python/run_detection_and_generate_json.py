import os
import json
import torch
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

CLEVR_LABELS_PATH = os.path.abspath("output/clevr_labels.json")
MODEL_PATH = os.path.abspath("output/clevr_model_final.pth")

def load_labels():
    with open(CLEVR_LABELS_PATH, "r") as f:
        return json.load(f)

def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(load_labels())
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)

def run_inference_and_save(image_path, output_json_path):
    print(f"[INFO] Running inference on {image_path}")
    predictor = setup_predictor()
    image = cv2.imread(image_path)
    outputs = predictor(image)

    instances = outputs["instances"]
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    labels = load_labels()
    objects = []

    for i, (box, cls_idx) in enumerate(zip(boxes, classes)):
        label = labels[cls_idx]
        tokens = label.split()  # e.g., "small red cube"
        if len(tokens) != 3:
            size, color, shape = "unknown", "unknown", "unknown"
        else:
            size, color, shape = tokens

        x_center = int((box[0] + box[2]) / 2)
        y_center = int((box[1] + box[3]) / 2)

        obj = {
            "label": label.replace(" ", "_"),
            "position": [x_center, y_center, 0],
            "size": size,
            "color": color,
            "shape": shape
        }
        objects.append(obj)

    scene = {"objects": objects}
    with open(output_json_path, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"[✅] CLEVR scene written to {output_json_path}")
