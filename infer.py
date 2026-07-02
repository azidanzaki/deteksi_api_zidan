import torch

# Monkey-patch torch.load untuk kompatibilitas checkpoint lama
_original_torch_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs.pop("weights_only", None)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_load

from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

# ==========================
# Load model
# ==========================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
model = YOLO(str(MODEL_PATH))

# ==========================
# Cache agar hasil video stabil
# ==========================
_last_detection = []
_lost_frame = 0
MAX_LOST_FRAME = 5


# ==========================
# Detect Image
# ==========================
def detect_fire_image(image_path: str):

    results = model.predict(
        source=image_path,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            det = {
                "bbox": box.xyxy.tolist()[0],
                "confidence": float(box.conf),
                "class_id": int(box.cls),
                "label": model.names[int(box.cls)]
            }

            # Filter smoke yang terlalu kecil
            if det["class_id"] == 1:
                x1, y1, x2, y2 = det["bbox"]
                area = (x2 - x1) * (y2 - y1)

                if area < 300:
                    continue

            detections.append(det)

    return detections


# ==========================
# Draw Bounding Box
# ==========================
def annotate_frame(frame: np.ndarray, detections):

    for det in detections:

        x1, y1, x2, y2 = map(int, det["bbox"])

        label = f"{det['label'].capitalize()} {det['confidence']:.2f}"

        color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame


# ==========================
# Detect Webcam Frame
# ==========================
def detect_fire_frame(frame: np.ndarray):

    global _last_detection
    global _lost_frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(
        source=rgb,
        conf=0.25,
        iou=0.45,
        imgsz=960,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            det = {
                "bbox": box.xyxy.tolist()[0],
                "confidence": float(box.conf),
                "class_id": int(box.cls),
                "label": model.names[int(box.cls)]
            }

            # Filter smoke yang terlalu kecil
            if det["class_id"] == 1:

                x1, y1, x2, y2 = det["bbox"]
                area = (x2 - x1) * (y2 - y1)

                if area < 300:
                    continue

            detections.append(det)

    # ==========================
    # Stabilizer (anti flicker)
    # ==========================
    if len(detections) > 0:

        _last_detection = detections
        _lost_frame = 0

    else:

        if _lost_frame < MAX_LOST_FRAME:
            detections = _last_detection
            _lost_frame += 1

    return detections