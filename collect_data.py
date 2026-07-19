import cv2
import os
import time

# Gestures we're collecting
GESTURES = ["thumbs_up", "peace", "open_palm", "fist", "ok_sign"]
IMAGES_PER_GESTURE = 80

# Create folders for each gesture
BASE_DIR = "gesture_data"
for gesture in GESTURES:
    os.makedirs(os.path.join(BASE_DIR, gesture), exist_ok=True)

cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    print(f"\n=== Get ready for: {gesture.upper()} ===")
    print("Press 's' to START capturing, 'q' to quit early")

    # Wait for user to be ready
    while True:
        success, frame = cap.read()
        cv2.putText(frame, f"Ready for '{gesture}'? Press 's' to start",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Capture images
    count = 0
    while count < IMAGES_PER_GESTURE:
        success, frame = cap.read()
        if not success:
            continue

        cv2.putText(frame, f"Capturing '{gesture}': {count}/{IMAGES_PER_GESTURE}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Data Collection", frame)

        # Save the RAW frame (without text overlay)
        clean_frame = frame.copy()
        filepath = os.path.join(BASE_DIR, gesture, f"{gesture}_{count}.jpg")
        cv2.imwrite(filepath, clean_frame)
        count += 1

        time.sleep(0.15)  # small delay so images vary slightly (move hand between shots)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Done with {gesture}! Captured {count} images.")

cap.release()
cv2.destroyAllWindows()
print("\nAll done! Check the 'gesture_data' folder.")