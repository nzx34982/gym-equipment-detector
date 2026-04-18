from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.dataset_voc import GymDetectionDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


project_root = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = GymDetectionDataset(
    images_dir=str(project_root / "images_flat"),
    annotations_dir=str(project_root / "annotations_voc"),
    split_file=str(project_root / "splits" / "train.txt")
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn
)

images, targets = next(iter(train_loader))
images = [img.to(device) for img in images]
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# 6 个类别 + 1 个背景类
model = get_model(num_classes=7)
model.to(device)
model.train()

loss_dict = model(images, targets)

print("device:", device)
print("loss_dict:", loss_dict)
print("total_loss:", sum(loss for loss in loss_dict.values()).item())