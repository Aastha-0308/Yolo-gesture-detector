from ultralytics import YOLO

model = YOLO("yolov8n.pt")

#custom gesture dataset
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="gesture_detector"
)