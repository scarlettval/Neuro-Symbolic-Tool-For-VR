import os
import json
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
import cv2

CLEVR_CLASSES = [
    "small red cube", "large red cube", "small blue cube", "large blue cube",
    "small green cube", "large green cube", "small red sphere", "large red sphere",
    "small blue sphere", "large blue sphere", "small green sphere", "large green sphere",
    "small red cylinder", "large red cylinder", "small blue cylinder", "large blue cylinder",
    "small green cylinder", "large green cylinder"
]

# This assumes the trained model is already available.
MODEL_PATH = os.path.abspath("output/clevr_model_final.pth")


def run_inference_and_save(image_path, json_output_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLEVR_CLASSES)
    predictor = DefaultPredictor(cfg)

    im = cv2.imread(image_path)
    outputs = predictor(im)

    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()

    objects = []
    for i in range(len(classes)):
        class_idx = classes[i]
        label = CLEVR_CLASSES[class_idx]
        size, color, shape = label.split(" ")
        center_x = int((boxes[i][0] + boxes[i][2]) / 2)
        center_z = int((boxes[i][1] + boxes[i][3]) / 2)
        obj = {
            "label": label,
            "size": size,
            "color": color,
            "shape": shape,
            "position": [center_x // 100, 0, center_z // 100]
        }
        objects.append(obj)

    with open(json_output_path, "w") as f:
        json.dump(objects, f, indent=2)

    print(f"✅ Scene JSON written to {json_output_path}")
