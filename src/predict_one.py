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


def main():
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取验证集第一张图片名
    val_txt = project_root / "splits" / "train.txt"
    with open(val_txt, "r", encoding="utf-8") as f:
        image_id = f.readline().strip()

    image_path = project_root / "images_flat" / f"{image_id}.jpg"
    save_dir = project_root / "outputs"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"{image_id}_pred.jpg"

    # 加载模型
    model = get_model(num_classes=7)
    ckpt_path = project_root / "checkpoints" / "gym_detector_v1.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # 读取图片
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(device)

    # 推理
    with torch.no_grad():
        output = model([image_tensor])[0]

    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    # 画框
    draw = ImageDraw.Draw(image)
    score_threshold = 0.2

    count = 0
    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        xmin, ymin, xmax, ymax = box
        class_name = ID_TO_CLASS_NAME.get(int(label), "unknown")
        text = f"{class_name} {score:.2f}"

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
        draw.text((xmin, max(ymin - 20, 0)), text, fill="red")
        count += 1

    image.save(save_path)
    print("前10个原始预测：")
    for i in range(min(10, len(scores))):
        cls_id = int(labels[i])
        cls_name = ID_TO_CLASS_NAME.get(cls_id, "unknown")
        print(f"{i}: class={cls_name}, score={scores[i]:.4f}, box={boxes[i]}")

    print("device:", device)
    print("image_id:", image_id)
    print("预测框数量(阈值过滤后):", count)
    print("结果已保存到:", save_path)


if __name__ == "__main__":
    main()