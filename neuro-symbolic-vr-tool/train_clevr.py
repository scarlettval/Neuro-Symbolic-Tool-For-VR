
import sys
import os
sys.path.insert(0, os.path.abspath("python"))

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
from engine import train_one_epoch
from clevr_dataset import CLEVRDataset

sys.path.insert(0, os.path.abspath("python"))  # Use insert(0) to give it priority

def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CLEVRDataset(
        root="data/CLEVR_v1.0/images/train",
        annFile="data/CLEVR_v1.0/annotations_coco_train.json",
        transforms=T.ToTensor()
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes=4)  # 3 shapes + background
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(5):
        train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=10)

    os.makedirs("trained_models", exist_ok=True)
    torch.save(model.state_dict(), "trained_models/maskrcnn_clevr.pth")
    print("✅ Model saved to trained_models/maskrcnn_clevr.pth")

if __name__ == "__main__":
    main()
