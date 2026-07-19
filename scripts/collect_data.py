import cv2
import os
import time

GESTURES = ["thumbs_up", "peace", "open_palm", "fist", "ok_sign"]
BASE_DIR = "gesture_data"

# Each stage = (instruction shown to user, number of images to capture)
STAGES = [
    ("Hold hand CLOSE to camera", 25),
    ("Hold hand at MEDIUM distance", 25),
    ("Hold hand FAR from camera", 20),
    ("Move hand around — different angles/tilts", 30),
    ("Try different position in frame (left/right/top/bottom)", 20),
]

for gesture in GESTURES:
    os.makedirs(os.path.join(BASE_DIR, gesture), exist_ok=True)

cap = cv2.VideoCapture(0)

def wait_for_ready(gesture, stage_instruction):
    while True:
        success, frame = cap.read()
        if not success:
            continue
        cv2.putText(frame, f"Gesture: {gesture.upper()}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, stage_instruction, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 's' to start, 'q' to quit", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            return True
        if key == ord('q'):
            return False

for gesture in GESTURES:
    print(f"\n=== Starting: {gesture.upper()} ===")
    total_count = 0

    for stage_instruction, num_images in STAGES:
        if not wait_for_ready(gesture, stage_instruction):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        captured = 0
        while captured < num_images:
            success, frame = cap.read()
            if not success:
                continue

            cv2.putText(frame, stage_instruction, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"{captured}/{num_images}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Data Collection", frame)

            clean_frame = frame.copy()
            filepath = os.path.join(BASE_DIR, gesture, f"{gesture}_{total_count}.jpg")
            cv2.imwrite(filepath, clean_frame)
            captured += 1
            total_count += 1

            time.sleep(0.15)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"  Stage done: {stage_instruction} ({captured} images)")

    print(f"Finished '{gesture}': {total_count} total images")

cap.release()
cv2.destroyAllWindows()
print("\nAll gestures captured with variety!")