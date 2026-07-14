from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .dataset_voc import ID_TO_CLASS_NAME
from .modeling import load_model_for_inference


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def predict_and_draw(
    model,
    image_path: Path,
    output_path: Path,
    device: torch.device,
    score_threshold: float = 0.3,
    max_boxes: int | None = None,
) -> dict[str, float | int]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    image_tensor = image_tensor.float().div(255).to(device)

    with torch.no_grad():
        output = model([image_tensor])[0]
    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    draw = ImageDraw.Draw(image)
    kept = 0
    best_score = 0.0
    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        if max_boxes is not None and kept >= max_boxes:
            break
        xmin, ymin, xmax, ymax = box
        xmin = max(0.0, min(float(xmin), width - 1.0))
        ymin = max(0.0, min(float(ymin), height - 1.0))
        xmax = max(0.0, min(float(xmax), width - 1.0))
        ymax = max(0.0, min(float(ymax), height - 1.0))
        if xmax <= xmin or ymax <= ymin:
            continue

        class_name = ID_TO_CLASS_NAME.get(int(label), "unknown")
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
        draw.text((xmin, max(ymin - 20, 0)), f"{class_name} {score:.2f}", fill="red")
        kept += 1
        best_score = max(best_score, float(score))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {"kept_boxes": kept, "best_score": best_score}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "gym_detector_v1.pth"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "prediction.jpg")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--max-boxes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(args.checkpoint, device)
    summary = predict_and_draw(
        model=model,
        image_path=args.image,
        output_path=args.output,
        device=device,
        score_threshold=args.score_threshold,
        max_boxes=args.max_boxes,
    )
    print(f"Device: {device}")
    print(f"Kept boxes: {summary['kept_boxes']}")
    print(f"Best score: {summary['best_score']:.4f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
