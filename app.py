import streamlit as st
import cv2
from ultralytics import YOLO

st.set_page_config(
    page_title="Gesture Detector",
    page_icon="🖐️",
    layout="wide"
)


st.markdown("""
    <style>
    .stat-card {
        background-color: #1e1e2e;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #7c5cff;
    }
    .stat-label {
        font-size: 14px;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

GESTURE_EMOJIS = {
    "thumbs_up": "👍",
    "peace": "✌️",
    "open_palm": "✋",
    "fist": "👊",
    "ok_sign": "👌",
}

# --- Header ---
st.title("🖐️ Real-Time Hand Gesture Detector")
st.caption("Custom fine-tuned YOLOv8 model — detects 5 hand gestures live from your webcam")


@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

col_video, col_stats = st.columns([3, 1])

with col_video:
    run = st.checkbox("▶️ Start Detection")
    frame_placeholder = st.empty()

with col_stats:
    st.subheader("📊 Live Detections")
    stats_placeholder = st.empty()

if not run:
    with col_video:
        frame_placeholder.info("Check the box above to start your webcam.")
    with col_stats:
        stats_placeholder.markdown("Waiting for detection to start...")

cap = cv2.VideoCapture(0)

while run:
    success, frame = cap.read()
    if not success:
        st.error("Failed to access webcam")
        break

    results = model(frame, verbose=False, conf=0.3)

    # Count detected objects by class ->
    counts = {}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        counts[label] = counts.get(label, 0) + 1

    # Render stat cards ->
    if counts:
        cards_html = ""
        for label, count in counts.items():
            emoji = GESTURE_EMOJIS.get(label, "🖐️")
            cards_html += f"""
                <div class="stat-card">
                    <div class="stat-label">{emoji} {label.replace('_', ' ').title()}</div>
                    <div class="stat-value">{count}</div>
                </div>
            """
        stats_placeholder.markdown(cards_html, unsafe_allow_html=True)
    else:
        stats_placeholder.markdown("No gestures detected")

    # Render video frame ->
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(annotated_frame, channels="RGB")

cap.release()