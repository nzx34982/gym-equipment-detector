from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


CLASS_NAME_TO_ID = {
    "treadmill": 1,
    "elliptical": 2,
    "stair_climber": 3,
    "big_scissor_machine": 4,
    "single_arm_row_machine": 5,
    "tbar_row_machine": 6,
}

ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}


def get_model(num_classes):
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def predict_and_draw(model, image_path, save_path, device, score_threshold=0.4, max_boxes=3):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(device)

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
        if kept >= max_boxes:
            break

        xmin, ymin, xmax, ymax = box

        xmin = max(0, min(float(xmin), w - 1))
        ymin = max(0, min(float(ymin), h - 1))
        xmax = max(0, min(float(xmax), w - 1))
        ymax = max(0, min(float(ymax), h - 1))

        if xmax <= xmin or ymax <= ymin:
            continue

        class_name = ID_TO_CLASS_NAME.get(int(label), "unknown")
        text = f"{class_name} {score:.2f}"

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
        draw.text((xmin, max(ymin - 20, 0)), text, fill="red")

        kept += 1
        if score > best_score:
            best_score = float(score)

    image.save(save_path)
    return kept, best_score

def main():
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=7)
    ckpt_path = project_root / "checkpoints" / "gym_detector_v1.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    val_txt = project_root / "splits" / "val.txt"
    output_dir = project_root / "outputs" / "val_preds"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(val_txt, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    print("device:", device)
    print("验证集图片数量:", len(image_ids))

    for image_id in image_ids:
        image_path = project_root / "images_flat" / f"{image_id}.jpg"
        save_path = output_dir / f"{image_id}_pred.jpg"

        kept, best_score = predict_and_draw(
            model=model,
            image_path=image_path,
            save_path=save_path,
            device=device,
            score_threshold=0.3
        )

        print(f"{image_id}: 保留框数={kept}, 最高分={best_score:.4f}")

    print("验证集预测结果已保存到:", output_dir)


if __name__ == "__main__":
    main()