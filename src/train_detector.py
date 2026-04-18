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


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for step, (images, targets) in enumerate(loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        print(f"Epoch [{epoch}] Step [{step}/{len(loader)}] Loss: {losses.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dataset = GymDetectionDataset(
        images_dir=str(project_root / "images_flat"),
        annotations_dir=str(project_root / "annotations_voc"),
        split_file=str(project_root / "splits" / "train.txt")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = get_model(num_classes=7)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    num_epochs = 10
    loss_history = []

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        loss_history.append(avg_loss)

    save_dir = project_root / "checkpoints"
    save_dir.mkdir(exist_ok=True)

    save_path = save_dir / "gym_detector_v1.pth"
    torch.save(model.state_dict(), save_path)

    print("训练完成")
    print("loss_history:", loss_history)
    print("模型已保存到:", save_path)


if __name__ == "__main__":
    main()