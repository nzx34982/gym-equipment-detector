from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = (
    "treadmill",
    "elliptical",
    "stair_climber",
    "big_scissor_machine",
    "single_arm_row_machine",
    "tbar_row_machine",
)
CLASS_NAME_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES, start=1)}
ID_TO_CLASS_NAME = {class_id: name for name, class_id in CLASS_NAME_TO_ID.items()}
NUM_CLASSES = len(CLASS_NAMES) + 1  # Six foreground classes plus background.


class GymDetectionDataset(Dataset):
    """A minimal Pascal VOC-style dataset for gym-equipment detection."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        split_file: str | Path,
        transforms: Callable | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transforms = transforms

        split_path = Path(split_file)
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file does not exist: {split_path}")
        self.ids = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.ids:
            raise ValueError(f"Split file contains no image identifiers: {split_path}")

    def __len__(self) -> int:
        return len(self.ids)

    def _image_path(self, image_id: str) -> Path:
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = self.images_dir / f"{image_id}{suffix}"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No JPG/JPEG/PNG image found for identifier '{image_id}' in "
            f"{self.images_dir}"
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.ids[idx]
        image_path = self._image_path(image_id)
        annotation_path = self.annotations_dir / f"{image_id}.xml"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation does not exist: {annotation_path}")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        root = ET.parse(annotation_path).getroot()

        boxes: list[list[float]] = []
        labels: list[int] = []
        for obj in root.findall("object"):
            name_node = obj.find("name")
            if name_node is None or not name_node.text:
                raise ValueError(f"Object without class name in {annotation_path}")
            class_name = name_node.text.strip()
            if class_name not in CLASS_NAME_TO_ID:
                raise ValueError(
                    f"Unknown class '{class_name}' in {annotation_path}; "
                    f"expected one of {CLASS_NAMES}"
                )

            box_node = obj.find("bndbox")
            if box_node is None:
                raise ValueError(f"Object without bndbox in {annotation_path}")
            try:
                xmin = float(box_node.findtext("xmin", ""))
                ymin = float(box_node.findtext("ymin", ""))
                xmax = float(box_node.findtext("xmax", ""))
                ymax = float(box_node.findtext("ymax", ""))
            except ValueError as exc:
                raise ValueError(f"Invalid bndbox values in {annotation_path}") from exc

            xmin = max(0.0, min(xmin, width - 1.0))
            ymin = max(0.0, min(ymin, height - 1.0))
            xmax = max(0.0, min(xmax, width - 1.0))
            ymax = max(0.0, min(ymax, height - 1.0))
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_NAME_TO_ID[class_name])

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        )
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
        }

        image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
        image_tensor = image_tensor.float() / 255.0
        if self.transforms is not None:
            image_tensor = self.transforms(image_tensor)
        return image_tensor, target
