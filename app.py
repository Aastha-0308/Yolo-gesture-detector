import streamlit as st
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

st.set_page_config(page_title="Gesture Detector", page_icon="🖐️", layout="wide")

st.markdown("""
    <style>
    video {
        max-width: 100% !important;
        border-radius: 12px;
        border: 2px solid #7c5cff;
    }
    .panel {
        background-color: #2a2a3d;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: #f0f0f5;
        font-size: 14px;
        line-height: 1.7;
    }
    .panel b { color: #ffffff; }
    .gesture-row {
        padding: 6px 0;
        border-bottom: 1px solid #3a3a4d;
        font-size: 14px;
    }
    .gesture-row:last-child { border-bottom: none; }
    div.block-container { padding-top: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

GESTURE_LABELS = {
    "thumbs_up": "👍 Thumbs Up",
    "peace": "✌️ Peace",
    "open_palm": "✋ Open Palm",
    "fist": "👊 Fist",
    "ok_sign": "👌 OK Sign",
}

st.title("🖐️ Real-Time Hand Gesture Detector")
st.caption("Custom fine-tuned YOLOv8 model — detects 5 hand gestures live from your webcam")

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model(img, verbose=False, conf=0.3)
    annotated = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

col_video, col_side = st.columns([2, 1])

with col_video:
    webrtc_streamer(
        key="gesture-detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with col_side:
    st.markdown("""
        <div class="panel">
        <b>How to use</b><br>
        1. Click START on the video<br>
        2. Allow camera access
        </div>
    """, unsafe_allow_html=True)

    rows = "".join(f'<div class="gesture-row">{v}</div>' for v in GESTURE_LABELS.values())
    st.markdown(f'<div class="panel"><b>Gestures it detects</b>{rows}</div>', unsafe_allow_html=True)