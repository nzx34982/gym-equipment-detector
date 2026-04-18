from pathlib import Path
import shutil

project_root = Path(r"D:\gym_detector")
src_dir = project_root / "images_jpg"
dst_dir = project_root / "images_flat"
dst_dir.mkdir(exist_ok=True)

count = 0
for p in src_dir.rglob("*"):
    if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        target = dst_dir / p.name
        shutil.copy2(p, target)
        count += 1

print("已复制图片数量:", count)
print("目标目录:", dst_dir)