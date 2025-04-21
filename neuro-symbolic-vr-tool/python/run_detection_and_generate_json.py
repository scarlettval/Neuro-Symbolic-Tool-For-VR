import os
import torch
import cv2
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

LABELS_PATH = os.path.abspath("output/clevr_labels.json")
MODEL_PATH = os.path.abspath("output/clevr_model_final.pth")

def load_labels():
    with open(LABELS_PATH, "r") as f:
        return json.load(f)

def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(load_labels())
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

def run_inference_and_save(image_path, json_output_path):
    print(f"[INFO] Running inference on {image_path}")
    image = cv2.imread(image_path)
    predictor = setup_predictor()
    outputs = predictor(image)

    labels = load_labels()
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.numpy()
    pred_boxes = instances.pred_boxes.tensor.numpy()

    result = []
    for i, cls_idx in enumerate(pred_classes):
        label = labels[str(cls_idx)]
        parts = label.split()
        if len(parts) == 3:
            size, color, shape = parts
        else:
            size, color, shape = "medium", "gray", "cube"

        center = pred_boxes[i][:2].tolist()
        obj = {
            "label": label,
            "size": size,
            "color": color,
            "shape": shape,
            "position": [int(center[0]), 0, int(center[1])]
        }
        result.append(obj)

    with open(json_output_path, "w") as f:
        json.dump(result, f, indent=2)
        print(f"✅ clevr_scene.json written to: {json_output_path}")
