from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import collate_fn, seed_everything
from .dataset_voc import CLASS_NAME_TO_ID, GymDetectionDataset, NUM_CLASSES
from .modeling import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train_one_epoch(model, loader, optimizer, device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
    for step, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]
        losses = sum(model(images, targets).values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
        print(
            f"Epoch {epoch} step {step}/{len(loader)} "
            f"loss={losses.item():.4f}"
        )
    return total_loss / len(loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the gym-equipment detector.")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "images_flat")
    parser.add_argument(
        "--annotations-dir", type=Path, default=PROJECT_ROOT / "annotations_voc"
    )
    parser.add_argument(
        "--split-file", type=Path, default=PROJECT_ROOT / "splits" / "train.txt"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "gym_detector_v1.pth"
    )
    parser.add_argument(
        "--history-file", type=Path, default=PROJECT_ROOT / "outputs" / "train_history.json"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not initialize the detector from TorchVision weights.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset = GymDetectionDataset(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        split_file=args.split_file,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    model = build_model(num_classes=NUM_CLASSES, pretrained=not args.no_pretrained)
    model.to(device)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )

    history: list[dict[str, float | int]] = []
    print(f"Device: {device}; training samples: {len(dataset)}")
    for epoch in range(1, args.epochs + 1):
        average_loss = train_one_epoch(model, loader, optimizer, device, epoch)
        history.append({"epoch": epoch, "average_loss": average_loss})
        print(f"Epoch {epoch} average_loss={average_loss:.4f}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_name_to_id": CLASS_NAME_TO_ID,
            "epochs": args.epochs,
            "seed": args.seed,
            "training_samples": len(dataset),
        },
        args.checkpoint,
    )
    args.history_file.parent.mkdir(parents=True, exist_ok=True)
    args.history_file.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Training history: {args.history_file}")


if __name__ == "__main__":
    main()
