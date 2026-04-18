import os
import random

random.seed(42)

xml_dir = "annotations_voc"
split_dir = "splits"
os.makedirs(split_dir, exist_ok=True)

names = []
for file in os.listdir(xml_dir):
    if file.endswith(".xml"):
        names.append(os.path.splitext(file)[0])

names.sort()
random.shuffle(names)

n = len(names)
train_num = int(n * 0.8)

train_names = names[:train_num]
val_names = names[train_num:]

with open(os.path.join(split_dir, "train.txt"), "w", encoding="utf-8") as f:
    for name in train_names:
        f.write(name + "\n")

with open(os.path.join(split_dir, "val.txt"), "w", encoding="utf-8") as f:
    for name in val_names:
        f.write(name + "\n")

print("总样本数:", n)
print("训练集:", len(train_names))
print("验证集:", len(val_names))