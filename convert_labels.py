import json
import os
from glob import glob

BASE_DIR = "gesture_data"
CLASSES = ["thumbs_up", "peace", "open_palm", "fist", "ok_sign"]

def convert_json_to_yolo(json_path, img_width, img_height):
    with open(json_path, "r") as f:
        data = json.load(f)

    yolo_lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASSES:
            continue
        class_id = CLASSES.index(label)

        points = shape["points"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Convert to YOLO format: normalized center x, y, width, height
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        box_width = (x_max - x_min) / img_width
        box_height = (y_max - y_min) / img_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    return yolo_lines

# Process all JSON files in each gesture folder
for gesture in CLASSES:
    folder = os.path.join(BASE_DIR, gesture)
    json_files = glob(os.path.join(folder, "*.json"))

    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)

        img_width = data["imageWidth"]
        img_height = data["imageHeight"]

        yolo_lines = convert_json_to_yolo(json_path, img_width, img_height)

        # Save as .txt with same name as image
        txt_path = json_path.replace(".json", ".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"Converted {len(json_files)} labels for '{gesture}'")

print("\nAll conversions done!")