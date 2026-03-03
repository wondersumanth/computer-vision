"""
Filter frames where any plant is visible in the image.
Uses YOLOv11 (potted-plant class) first, then falls back to HSV green-region
analysis. A frame passes as long as a plant/green region is detected above
the minimum coverage threshold — no centering or edge constraints applied.

Passing frames are copied to yolo_shortlisted/.
"""

import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR  = "extracted_frames"
OUTPUT_DIR = "yolo_shortlisted"

# Minimum fraction of the frame that must be green for a plant to count.
GREEN_MIN_RATIO = 0.05      # 5 % — very permissive

# YOLOv11 confidence threshold
CONF_THRESHOLD = 0.20
PLANT_CLASS_ID = 58         # COCO: potted plant

# HSV range for green vegetation
LOWER_GREEN = np.array([25,  30,  30])
UPPER_GREEN = np.array([90, 255, 255])
# ─────────────────────────────────────────────────────────────────────────────


def has_plant_green(img):
    """Return (True, ratio) if green coverage meets the minimum threshold."""
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask   = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    h, w   = img.shape[:2]
    ratio  = int(mask.sum() / 255) / (h * w)
    return ratio >= GREEN_MIN_RATIO, ratio


def evaluate_frame(img, model):
    """
    Returns (passed: bool, method: str, details: str).
    """
    h, w = img.shape[:2]

    # ── Step 1: YOLOv11 potted-plant detection ────────────────────────────────
    results = model(img, verbose=False, conf=CONF_THRESHOLD)
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            if int(box.cls[0]) != PLANT_CLASS_ID:
                continue
            conf = float(box.conf[0])
            return True, "YOLO", f"potted plant detected (conf {conf:.2f})"

    # ── Step 2: Green-region fallback ─────────────────────────────────────────
    found, ratio = has_plant_green(img)
    if found:
        return True, "GREEN", f"green coverage {ratio:.1%}"
    return False, "GREEN", f"insufficient green ({ratio:.1%})"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading YOLOv11 model (yolo11n.pt) …")
    model = YOLO("yolo11n.pt")
    print("Model ready.\n")

    frames = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    total = len(frames)
    print(f"Scanning {total} frames in '{INPUT_DIR}/'")
    print(f"Output  → '{OUTPUT_DIR}/'")
    print("-" * 65)

    kept = 0
    for i, fname in enumerate(frames, 1):
        fpath = os.path.join(INPUT_DIR, fname)
        img   = cv2.imread(fpath)
        if img is None:
            continue

        passed, method, details = evaluate_frame(img, model)

        if passed:
            shutil.copy2(fpath, os.path.join(OUTPUT_DIR, fname))
            kept += 1
            print(f"  ✓ [{method:5s}] {fname}  —  {details}")

        if i % 100 == 0:
            print(f"  … {i}/{total} processed | {kept} kept so far")

    print("-" * 65)
    print(f"Done.  {kept} / {total} frames saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
