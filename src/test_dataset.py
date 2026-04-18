from pathlib import Path
from src.dataset_voc import GymDetectionDataset

project_root = Path(__file__).resolve().parent.parent

dataset = GymDetectionDataset(
    images_dir=str(project_root / "images_flat"),
    annotations_dir=str(project_root / "annotations_voc"),
    split_file=str(project_root / "splits" / "train.txt")
)

print("训练集样本数:", len(dataset))

image, target = dataset[0]
print("image shape:", image.shape)
print("boxes:", target["boxes"])
print("labels:", target["labels"])