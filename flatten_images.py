from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def flatten_images(source_dir: Path, destination_dir: Path) -> int:
    """Copy nested images into one directory without silently overwriting files."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source image directory does not exist: {source_dir}")

    image_paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    duplicate_names = sorted(
        name
        for name in {path.name for path in image_paths}
        if sum(path.name == name for path in image_paths) > 1
    )
    if duplicate_names:
        preview = ", ".join(duplicate_names[:5])
        raise ValueError(f"Duplicate image basenames detected: {preview}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        shutil.copy2(image_path, destination_dir / image_path.name)
    return len(image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten nested class folders into a single image directory."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "images_jpg",
        help="Nested source image directory.",
    )
    parser.add_argument(
        "--destination-dir",
        type=Path,
        default=PROJECT_ROOT / "images_flat",
        help="Destination directory for flattened images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copied = flatten_images(args.source_dir, args.destination_dir)
    print(f"Copied images: {copied}")
    print(f"Destination: {args.destination_dir.resolve()}")


if __name__ == "__main__":
    main()
