import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response
from infer import detect_fire_image, detect_fire_frame
 
import cv2
import base64
import numpy as np
from pathlib import Path

app = Flask(__name__)
import cv2
# Ensure the model path is correctly resolved (infer.py already loads the model)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload')
def upload_page():
    return render_template('upload_image.html')

@app.route('/camera')
def camera_page():
    return render_template('life_detection.html')
latest_detection = {
    "object_label": "-",
    "total_bbox_area": 0,
    "percent": 0,
    "category": "-",
}
def get_category(percent):
    if percent < 0.5:
        return "Sangat Kecil"
    elif percent < 2:
        return "Kecil"
    elif percent < 5:
        return "Sedang"
    else:
        return "Besar"


@app.route("/detect_frame", methods=["POST"])
def detect_frame():

    global latest_detection

    data = request.json

    image_data = data["image"]

    image_data = image_data.split(",")[1]

    image_bytes = base64.b64decode(image_data)

    npimg = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    detections = detect_fire_frame(frame)

    img_h, img_w = frame.shape[:2]
    total_area = img_h * img_w

    fire_area = 0
    smoke_area = 0

    for det in detections:

        x1, y1, x2, y2 = map(int, det["bbox"])

        area = (x2-x1)*(y2-y1)

        if det["class_id"] == 0:
            fire_area += area
            label = "Fire"
        else:
            smoke_area += area
            label = "Smoke"

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.putText(
            frame,
            f"{label} {det['confidence']:.2f}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

    fire_percent = fire_area/total_area*100 if total_area else 0
    smoke_percent = smoke_area/total_area*100 if total_area else 0

    latest_detection = {

        "object_label":
            "Fire, Smoke" if fire_area and smoke_area else
            "Fire" if fire_area else
            "Smoke" if smoke_area else
            "Tidak Ada",

        "fire":{
            "area":fire_area,
            "percent":round(fire_percent,2),
            "category":get_category(fire_percent)
        },

        "smoke":{
            "area":smoke_area,
            "percent":round(smoke_percent,2),
            "category":get_category(smoke_percent)
        }

    }

    _, buffer = cv2.imencode(".jpg", frame)

    encoded = base64.b64encode(buffer).decode()

    return jsonify({

        "image":"data:image/jpeg;base64,"+encoded,

        "result":latest_detection

    })
    
# def gen_frames():
#     """Yield JPEG frames from webcam with fire detection overlay."""
#     global latest_detection

#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         raise RuntimeError("Cannot open webcam")

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             detections = detect_fire_frame(frame)

#             # Ukuran frame
#             img_h, img_w = frame.shape[:2]
#             total_area = img_w * img_h

#             # Total luas bounding box
#             total_bbox_area = 0

#             # Gambar bounding box
#             for det in detections:
#                 x1, y1, x2, y2 = map(int, det["bbox"])
#                 conf = det["confidence"]
#                 class_id = det["class_id"]

#                 label = "Smoke" if class_id == 1 else "Fire"

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(
#                     frame,
#                     f"{label} {conf:.2f}",
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 0, 255),
#                     2
#                 )

#                 total_bbox_area += (x2 - x1) * (y2 - y1)

#             # ==========================
#             # Hitung Fire & Smoke terpisah
#             # ==========================

#             fire_area = 0
#             smoke_area = 0

#             for det in detections:

#                 x1, y1, x2, y2 = map(int, det["bbox"])

#                 width = max(0, x2 - x1)
#                 height = max(0, y2 - y1)

#                 area = width * height

#                 if det["class_id"] == 0:
#                     fire_area += area
#                 elif det["class_id"] == 1:
#                     smoke_area += area

#             fire_percent = (fire_area / total_area) * 100 if total_area else 0
#             smoke_percent = (smoke_area / total_area) * 100 if total_area else 0

#             fire_category = get_category(fire_percent)
#             smoke_category = get_category(smoke_percent)

#             # Tentukan objek yang terdeteksi
#             labels = []

#             if any(det["class_id"] == 0 for det in detections):
#                 labels.append("Fire")

#             if any(det["class_id"] == 1 for det in detections):
#                 labels.append("Smoke")

#             if not labels:
#                 labels.append("Tidak Ada")

#             object_label = ", ".join(labels)

#             # Simpan hasil terbaru agar bisa dibaca AJAX
#             latest_detection = {

#                 "object_label": object_label,

#                 "fire": {
#                     "area": int(fire_area),
#                     "percent": round(fire_percent, 2),
#                     "category": fire_category
#                 },

#                 "smoke": {
#                     "area": int(smoke_area),
#                     "percent": round(smoke_percent, 2),
#                     "category": smoke_category
#                 }

#             }

#             # Encode frame
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if not ret:
#                 continue

#             frame_bytes = buffer.tobytes()

#             yield (
#                 b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' +
#                 frame_bytes +
#                 b'\r\n'
#             )

#     finally:
#         cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_data')
def detection_data():
    return jsonify(latest_detection)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    img_file = request.files['image']
    # Save to a temporary location
    tmp_path = Path('tmp')
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / img_file.filename
    img_file.save(file_path)

    # Load image with OpenCV
    img = cv2.imread(str(file_path))
    if img is None:
        return jsonify({"error": "Failed to read image"}), 400

    # Run detection
    detections = detect_fire_image(str(file_path))

    # Annotate image with bounding boxes and appropriate label
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        class_id = det["class_id"]
        label = "Smoke" if class_id == 1 else "Fire"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save annotated image to static folder for display
    result_dir = Path('static/results')
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"annotated_{img_file.filename}"
    cv2.imwrite(str(result_path), img)

    # Compute summary statistics
    img_h, img_w = img.shape[:2]
    total_area = img_w * img_h
    fire_area = 0
    smoke_area = 0

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        area = (x2 - x1) * (y2 - y1)

        if det["class_id"] == 0:
            fire_area += area
        elif det["class_id"] == 1:
            smoke_area += area

    fire_percent = (fire_area / total_area) * 100 if total_area else 0
    smoke_percent = (smoke_area / total_area) * 100 if total_area else 0

    fire_category = get_category(fire_percent)
    smoke_category = get_category(smoke_percent)
    
    # Tentukan semua objek yang terdeteksi
    labels = []

    if any(det["class_id"] == 0 for det in detections):
        labels.append("Fire")

    if any(det["class_id"] == 1 for det in detections):
        labels.append("Smoke")

    if not labels:
        labels.append("Tidak Ada")

    object_label = ", ".join(labels)

    # Build URL for the annotated image
    from flask import url_for
    image_url = url_for('static', filename=f'results/annotated_{img_file.filename}')
    # Hapus string HTML f-string lama, ganti dengan return JSON ini:
    response_data = {

        "image_url": image_url,

        "object_label": object_label,

        "fire": {
            "area": int(fire_area),
            "percent": round(fire_percent,2),
            "category": fire_category
        },

        "smoke": {
            "area": int(smoke_area),
            "percent": round(smoke_percent,2),
            "category": smoke_category
        }
    }

    # Clean up the original uploaded file
    try:
        os.remove(file_path)
    except OSError:
        pass

    return jsonify(response_data)  # Mengembalikan JSON ke JavaScript

if __name__ == '__main__':
    # Use 0.0.0.0 to be reachable from other devices if needed
    app.run(host='0.0.0.0', port=5000, debug=True)
