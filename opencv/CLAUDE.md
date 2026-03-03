# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Applications

```bash
# Practical exercises: grayscale, edge detection (Canny), face detection (Haar Cascade), color detection
python app.py

# Side-by-side comparison of OpenCV Haar Cascade vs YOLOv8 face detection on lots_of_people.jpg
python opencv_yolo_comparison.py
```

## Architecture

### app.py
Sequential pipeline that processes `Mohith.jpeg` through 5 stages, each opening a blocking `cv2.imshow` window (press any key to advance):
1. Raw image display
2. Grayscale conversion
3. Canny edge detection
4. Haar Cascade face detection with bounding boxes
5. HSV-based red color detection with masking

### opencv_yolo_comparison.py
Side-by-side matplotlib comparison of two face detection approaches on `lots_of_people.jpg`:
- **OpenCV path:** `detect_faces_opencv()` — Haar Cascade on grayscale image
- **YOLOv8 path:** `detect_faces_yolo()` → always delegates to `detect_faces_yolo_fallback()`, which loads `yolov8n.pt` (auto-downloaded on first run), detects persons (COCO class 0), then estimates face regions from the top portion of each person bounding box

The YOLOv8 fallback approach does not directly detect faces — it estimates face bounding boxes from person detections, so results reflect person count rather than true face detection.

## Dependencies

- `opencv-python==4.9.0.80`
- `numpy==1.26.4`
- `matplotlib` — used only in `opencv_yolo_comparison.py` for side-by-side display
- `ultralytics` — YOLOv8; auto-downloads `yolov8n.pt` (~6MB) to working directory on first run

## Media Assets

- `Mohith.jpeg`, `image_boys.png` — used by `app.py`
- `lots_of_people.jpg` — used by `opencv_yolo_comparison.py`
- `video.mov`, `new-video.mp4` — available for video processing experiments
