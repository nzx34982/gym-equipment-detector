from __future__ import annotations

import argparse
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def create_splits(
    annotations_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Create deterministic train/validation splits from VOC XML filenames."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"Annotation directory does not exist: {annotations_dir}"
        )

    names = sorted(path.stem for path in annotations_dir.glob("*.xml"))
    if len(names) < 2:
        raise ValueError("At least two XML annotations are required to create splits")

    random.Random(seed).shuffle(names)
    train_count = max(1, min(len(names) - 1, int(len(names) * train_ratio)))
    train_names = names[:train_count]
    val_names = names[train_count:]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.txt").write_text(
        "\n".join(train_names) + "\n", encoding="utf-8"
    )
    (output_dir / "val.txt").write_text(
        "\n".join(val_names) + "\n", encoding="utf-8"
    )
    return train_names, val_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation split files."
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=PROJECT_ROOT / "annotations_voc",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "splits"
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_names, val_names = create_splits(
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(f"Total samples: {len(train_names) + len(val_names)}")
    print(f"Training samples: {len(train_names)}")
    print(f"Validation samples: {len(val_names)}")


if __name__ == "__main__":
    main()
