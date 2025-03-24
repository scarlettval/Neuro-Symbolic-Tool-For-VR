import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import os

# COCO labels (subset you'll likely use)
COCO_LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat'
]

class MaskRCNNDetector:
    def __init__(self, confidence_threshold=0.7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights).to(self.device).eval()
        self.confidence_threshold = confidence_threshold

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return F.to_tensor(image).to(self.device), image

    def predict(self, image_path):
        tensor_image, _ = self.load_image(image_path)

        with torch.no_grad():
            predictions = self.model([tensor_image])[0]

        results = []
        for i, score in enumerate(predictions["scores"]):
            if score < self.confidence_threshold:
                continue

            label_id = predictions["labels"][i].item()
            label_name = COCO_LABELS[label_id] if label_id < len(COCO_LABELS) else str(label_id)
            box = predictions["boxes"][i].cpu().numpy().astype(int).tolist()

            results.append({
                "label": label_name,
                "score": round(score.item(), 2),
                "box": box
            })

        return results
