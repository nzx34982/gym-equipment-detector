from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .modeling import load_model_for_inference
from .predict_one import predict_and_draw


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a split file.")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "images_flat")
    parser.add_argument(
        "--split-file", type=Path, default=PROJECT_ROOT / "splits" / "val.txt"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "gym_detector_v1.pth"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "val_predictions"
    )
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--max-boxes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_ids = [
        line.strip()
        for line in args.split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not image_ids:
        raise ValueError(f"Split contains no image identifiers: {args.split_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, str | float | int]] = []
    for image_id in image_ids:
        image_path = args.images_dir / f"{image_id}.jpg"
        output_path = args.output_dir / f"{image_id}_pred.jpg"
        summary = predict_and_draw(
            model=model,
            image_path=image_path,
            output_path=output_path,
            device=device,
            score_threshold=args.score_threshold,
            max_boxes=args.max_boxes,
        )
        report_rows.append({"image_id": image_id, **summary})
        print(
            f"{image_id}: kept={summary['kept_boxes']} "
            f"best_score={summary['best_score']:.4f}"
        )

    report_path = args.output_dir / "summary.csv"
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Summary: {report_path}")


if __name__ == "__main__":
    main()
