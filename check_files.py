from pathlib import Path

project_root = Path(r"D:\gym_detector")
images_dir = project_root / "images_jpg"
split_file = project_root / "splits" / "train.txt"

image_stems = {p.stem for p in images_dir.iterdir() if p.is_file()}

missing = []
with open(split_file, "r", encoding="utf-8") as f:
    for line in f:
        name = line.strip()
        if name and name not in image_stems:
            missing.append(name)

print("缺失图片数量:", len(missing))
for name in missing:
    print(name)