from pathlib import Path

import torch
from PIL import Image

from src.dataset_voc import CLASS_NAME_TO_ID, GymDetectionDataset


def write_annotation(path: Path, class_name: str, box: tuple[int, int, int, int]) -> None:
    xmin, ymin, xmax, ymax = box
    path.write_text(
        f"""<annotation>
  <object>
    <name>{class_name}</name>
    <bndbox>
      <xmin>{xmin}</xmin><ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax><ymax>{ymax}</ymax>
    </bndbox>
  </object>
</annotation>
""",
        encoding="utf-8",
    )


def test_dataset_parses_voc_sample(tmp_path: Path) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (20, 10), color="white").save(images / "sample.jpg")
    write_annotation(annotations / "sample.xml", "treadmill", (1, 2, 15, 9))
    split = tmp_path / "split.txt"
    split.write_text("sample\n", encoding="utf-8")

    dataset = GymDetectionDataset(images, annotations, split)
    image, target = dataset[0]

    assert image.shape == (3, 10, 20)
    assert image.dtype == torch.float32
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [CLASS_NAME_TO_ID["treadmill"]]
    assert target["area"].tolist() == [98.0]


def test_dataset_returns_stable_empty_box_shape(tmp_path: Path) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (10, 10), color="white").save(images / "empty.png")
    (annotations / "empty.xml").write_text(
        "<annotation></annotation>", encoding="utf-8"
    )
    split = tmp_path / "split.txt"
    split.write_text("empty\n", encoding="utf-8")

    _, target = GymDetectionDataset(images, annotations, split)[0]
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert target["area"].shape == (0,)
