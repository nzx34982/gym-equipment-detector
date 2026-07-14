from pathlib import Path

import pytest

from check_files import find_missing_images
from flatten_images import flatten_images
from make_splits import create_splits


def test_flatten_images_copies_nested_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "class_a").mkdir(parents=True)
    (source / "class_b").mkdir(parents=True)
    (source / "class_a" / "a.jpg").write_bytes(b"a")
    (source / "class_b" / "b.png").write_bytes(b"b")

    destination = tmp_path / "flat"
    assert flatten_images(source, destination) == 2
    assert sorted(path.name for path in destination.iterdir()) == ["a.jpg", "b.png"]


def test_flatten_images_rejects_duplicate_basenames(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "class_a").mkdir(parents=True)
    (source / "class_b").mkdir(parents=True)
    (source / "class_a" / "same.jpg").write_bytes(b"a")
    (source / "class_b" / "same.jpg").write_bytes(b"b")

    with pytest.raises(ValueError, match="Duplicate image basenames"):
        flatten_images(source, tmp_path / "flat")


def test_create_splits_is_deterministic(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for index in range(10):
        (annotations / f"sample_{index}.xml").write_text("<annotation />")

    first_train, first_val = create_splits(
        annotations, tmp_path / "first", train_ratio=0.8, seed=42
    )
    second_train, second_val = create_splits(
        annotations, tmp_path / "second", train_ratio=0.8, seed=42
    )

    assert first_train == second_train
    assert first_val == second_val
    assert len(first_train) == 8
    assert len(first_val) == 2


def test_find_missing_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "present.jpg").write_bytes(b"image")
    split = tmp_path / "split.txt"
    split.write_text("present\nmissing\n", encoding="utf-8")

    assert find_missing_images(images, split) == ["missing"]
