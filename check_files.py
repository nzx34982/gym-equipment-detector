from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def find_missing_images(images_dir: Path, split_file: Path) -> list[str]:
    """Return split identifiers that have no matching image file."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")

    image_stems = {
        path.stem
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    split_ids = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [image_id for image_id in split_ids if image_id not in image_stems]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether every split identifier has a matching image."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=PROJECT_ROOT / "images_flat",
        help="Directory containing flattened JPG/PNG images.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=PROJECT_ROOT / "splits" / "train.txt",
        help="Text file containing one image identifier per line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = find_missing_images(args.images_dir, args.split_file)
    print(f"Missing images: {len(missing)}")
    for image_id in missing:
        print(image_id)


if __name__ == "__main__":
    main()
