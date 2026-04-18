from pathlib import Path
from torch.utils.data import DataLoader

from src.dataset_voc import GymDetectionDataset


def collate_fn(batch):
    return tuple(zip(*batch))


project_root = Path(__file__).resolve().parent.parent

train_dataset = GymDetectionDataset(
    images_dir=str(project_root / "images_flat"),
    annotations_dir=str(project_root / "annotations_voc"),
    split_file=str(project_root / "splits" / "train.txt")
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

images, targets = next(iter(train_loader))

print("batch 图片数:", len(images))
print("第1张图片 shape:", images[0].shape)
print("第1张图片 boxes shape:", targets[0]["boxes"].shape)
print("第1张图片 labels:", targets[0]["labels"])
print("第2张图片 shape:", images[1].shape)
print("第2张图片 boxes shape:", targets[1]["boxes"].shape)
print("第2张图片 labels:", targets[1]["labels"])