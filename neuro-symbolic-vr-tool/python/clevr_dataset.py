import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class CLEVRDataset(Dataset):
    def __init__(self, annFile=None, annotation_file=None, image_dir=None, root=None, transforms=None):
        if annFile is not None:
            annotation_file = annFile
        if annotation_file is None:
            raise ValueError("You must provide 'annotation_file' or 'annFile'")
        if root is not None and image_dir is None:
            image_dir = root
        if image_dir is None:
            raise ValueError("You must provide 'image_dir' or 'root'")

        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        if 'images' not in self.coco or 'annotations' not in self.coco:
            raise ValueError("Annotation file must contain 'images' and 'annotations' keys.")
        
        self.image_dir = image_dir
        self.transforms = transforms if transforms is not None else T.ToTensor()
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco['images']}
        self.image_id_to_annotations = {}
        for ann in self.coco['annotations']:
            image_id = ann.get('image_id')
            if image_id is None:
                continue
            self.image_id_to_annotations.setdefault(image_id, []).append(ann)
        self.image_ids = list(self.image_id_to_filename.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        filename = self.image_id_to_filename.get(image_id)
        if not filename:
            raise ValueError(f"No filename found for image_id {image_id}")
        
        image_path = os.path.join(self.image_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Error opening image file {image_path}: {e}")

        anns = self.image_id_to_annotations.get(image_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            bbox = ann.get('bbox')
            if bbox is None or len(bbox) != 4:
                continue
            x_center, y_center, width, height = bbox
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann.get('category_id', 0))
            areas.append(width * height)
            iscrowd.append(0)

        if len(boxes) == 0:
            raise ValueError(f"No valid boxes for image_id {image_id} ({image_path})")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id_tensor = torch.tensor([image_id])

        width, height = image.size
        masks = torch.zeros((len(boxes), height, width), dtype=torch.uint8)
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box.int()
            masks[i, y0:y1, x0:x1] = 1

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id_tensor,
            "area": areas,
            "iscrowd": iscrowd,
            "masks": masks
        }

        image = self.transforms(image)
        return image, target
