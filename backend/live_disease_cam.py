import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# -----------------------------------------
# 1. LOAD MODELS
# -----------------------------------------

# Load YOUR trained YOLO chicken detector
yolo_model = YOLO(r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\chicken_detector.pt")

# FORCE YOLO TO DETECT ONLY CHICKEN (CLASS 0)
yolo_model.overrides['classes'] = [0]

# Load YOUR Xception disease classifier
disease_model = load_model(r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\xception_disease.h5")

# Load disease class names
with open(r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\disease_classes.txt", "r") as f:
    disease_labels = [line.strip() for line in f.readlines()]

# -----------------------------------------
# 2. START WEBCAM
# -----------------------------------------
cap = cv2.VideoCapture(0)

print("✔ Webcam started")
print("✔ Running live disease detection...")

# -----------------------------------------
# 3. REAL-TIME LOOP
# -----------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read frame")
        break

    results = yolo_model(frame, verbose=False)[0]

    # -----------------------------------------
    # PROCESS EACH DETECTION
    # -----------------------------------------
    for box in results.boxes:

        # STRICT CONFIDENCE FILTER
        conf = float(box.conf[0])
        if conf < 0.75:   # HIGHER = fewer false detections
            continue

        # CLASS FILTER — ONLY CHICKEN (class 0)
        cls = int(box.cls[0])
        if cls != 0:
            continue

        # EXTRACT BOX COORDINATES
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # REJECT VERY SMALL OBJECTS (false positives)
        w = x2 - x1
        h = y2 - y1
        if w < 80 or h < 80:
            continue

        # CROP CHICKEN
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue

        # -----------------------------------------
        # PREPROCESS FOR XCEPTION MODEL
        # -----------------------------------------
        crop_resized = cv2.resize(crop, (299, 299))
        crop_norm = crop_resized.astype("float32") / 255.0
        crop_norm = np.expand_dims(crop_norm, axis=0)

        # -----------------------------------------
        # PREDICT DISEASE
        # -----------------------------------------
        pred = disease_model.predict(crop_norm, verbose=0)
        idx = int(np.argmax(pred))
        disease_name = disease_labels[idx]
        disease_conf = float(pred[0][idx])

        label = f"{disease_name} ({disease_conf:.2f})"

        # DRAW RESULTS
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # DISPLAY WINDOW
    cv2.imshow("Live Disease Detection", frame)

    # QUIT WITH Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✔ Live detection stopped")
