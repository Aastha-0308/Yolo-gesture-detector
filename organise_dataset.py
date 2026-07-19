import os
import shutil
import random
from glob import glob

BASE_DIR = "gesture_data"
OUTPUT_DIR = "dataset"
CLASSES = ["thumbs_up", "peace", "open_palm", "fist", "ok_sign"]
VAL_SPLIT = 0.2  

random.seed(42)  

# Create output folder structure
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

for gesture in CLASSES:
    folder = os.path.join(BASE_DIR, gesture)
    image_files = glob(os.path.join(folder, "*.jpg"))

    random.shuffle(image_files)
    val_count = int(len(image_files) * VAL_SPLIT)
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]

    for split, files in [("train", train_files), ("val", val_files)]:
        for img_path in files:
            txt_path = img_path.replace(".jpg", ".txt")
            if not os.path.exists(txt_path):
                continue  # skip images without labels

            img_name = os.path.basename(img_path)
            txt_name = os.path.basename(txt_path)

            shutil.copy(img_path, os.path.join(OUTPUT_DIR, "images", split, img_name))
            shutil.copy(txt_path, os.path.join(OUTPUT_DIR, "labels", split, txt_name))

    print(f"{gesture}: {len(train_files)} train, {len(val_files)} val")

print("\nDataset organized!")