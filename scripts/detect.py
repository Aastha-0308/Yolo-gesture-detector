from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break


    results = model(frame, verbose=False, conf=0.5)
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"{label}: {conf:.2f}")


    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()