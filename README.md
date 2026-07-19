# 🖐️ Real-Time Hand Gesture Detector

A custom-trained YOLOv8 model that detects 5 hand gestures live from a webcam, built end-to-end — from data collection to a deployed web app.

![Demo](demo.gif)


## What it does

Detects 5 hand gestures in real time via webcam:

| Gesture | Emoji |
|---|---|
| Thumbs Up | 👍 |
| Peace Sign | ✌️ |
| Open Palm | ✋ |
| Fist | 👊 |
| OK Sign | 👌 |

The app shows a live annotated video feed plus a real-time count of detected gestures in the sidebar.

## Why I built this

I started with a pretrained YOLOv8 model (trained on COCO's 80 general object classes) and quickly ran into its limits — it has no concept of "hand gestures," since that's not a class it was trained on. Rather than settle for that, I collected my own dataset, labeled it, and fine-tuned YOLOv8 to recognize gestures it had never seen before. This project covers the full ML pipeline: **data collection → labeling → training → evaluation → deployment.**

## Tech Stack

- **YOLOv8** (Ultralytics) — object detection model, fine-tuned on custom data
- **OpenCV** — webcam capture and frame processing
- **Streamlit** — web app interface
- **LabelMe** — local, offline image annotation (kept training photos private, not uploaded to any third-party service)

## How it was built

1. **Data collection** (`collect_data.py`) — captured ~400 webcam images across 5 gesture classes
2. **Labeling** (LabelMe) — manually drew bounding boxes and assigned gesture labels, run 100% locally for privacy
3. **Format conversion** (`convert_labels.py`) — converted LabelMe's JSON output into YOLO's normalized `.txt` label format
4. **Dataset organization** (`organise_dataset.py`) — split data into train/val sets (80/20)
5. **Fine-tuning** (`train.py`) — trained YOLOv8n (nano) starting from pretrained COCO weights, 50 epochs, CPU-only
6. **Deployment** (`app.py`) — wrapped the fine-tuned model in a Streamlit interface with live detection and gesture counting

## Results

| Metric | Score |
|---|---|
| mAP50 (all classes) | 0.995 |
| mAP50-95 (all classes) | 0.739 |

All 5 gesture classes performed consistently well, with mAP50 scores of 0.994–0.997 individually.

## Known limitations & what I'd improve next

- **Thumbs Up** initially showed flickering detection (confidence hovering near the threshold) — likely due to limited variety in training angles/distances for that class. Lowering the confidence threshold and testing in better lighting improved this.
- **Peace Sign** occasionally gets confused with **OK Sign** and **Open Palm** at certain angles, since these gestures share visually similar hand silhouettes from some viewpoints.
- **Root cause:** my original dataset lacked variety in hand distance, angle, and position — most training images were captured from a fairly consistent position. Neural networks generalize based on the *diversity* of training examples, not just quantity.
- **Next iteration:** recollect data using a staged capture script that deliberately varies distance, angle, and frame position per gesture (already built, see `collect_data.py`), targeting ~100-120 images per class instead of 80.

## Running it locally

```bash
git clone https://github.com/YOUR_USERNAME/yolo-gesture-detector.git
cd yolo-gesture-detector
pip install -r requirements.txt
streamlit run app.py
```

Check the "Start Detection" box to begin webcam-based gesture recognition.

## Project structure

```
├── app.py                  # Streamlit web app (main entry point)
├── train.py                 # Fine-tuning script
├── collect_data.py          # Staged webcam data collection with variety prompts
├── convert_labels.py        # LabelMe JSON → YOLO format converter
├── organise_dataset.py      # Train/val dataset splitter
├── data.yaml                 # YOLO dataset config
├── models/
│   └── best.pt               # Fine-tuned model weights
└── requirements.txt
```

## Future roadmap

This project is Phase 1 of a longer-term idea: expanding from basic gesture detection toward full sign language recognition (starting with static alphabet signs, eventually moving to motion-based word/sentence recognition), with text-to-speech output.

## Note on training data

Raw training images are excluded from this repository for privacy (see `.gitignore`). The trained model weights (`models/best.pt`) are included and fully functional — they're just numerical parameters, not reconstructible into the original photos.