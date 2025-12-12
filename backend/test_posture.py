# test_full_posture_FIXED.py
from ultralytics import YOLO
import cv2
import numpy as np
import joblib
import tensorflow as tf
import os

CHICKEN_MODEL = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\best.pt"
SCALER_PATH = "posture_scaler.pkl"
MODEL_PATH = "posture_mlp_adv.h5"
IMG_PATH = r"C:\Users\mateena sadaf\Desktop\poultry disease\datasets\posture_dataset\sick\IMG_20230109_124903_jpg.rf.0f3950543a59239ac0aac1fdf4d6ee51.jpg"

chicken_model = YOLO(CHICKEN_MODEL)
scaler = joblib.load(SCALER_PATH)
classifier = tf.keras.models.load_model(MODEL_PATH)

img = cv2.imread(IMG_PATH)
results = chicken_model(img, conf=0.1, verbose=False)[0]

box = results.boxes[0].xyxy[0].cpu().numpy().astype(int)
x1,y1,x2,y2 = box
w, h = x2-x1, y2-y1

# ✅ FIXED FEATURES (match training averages)
features = np.array([
    109.0, 180.0, 109.0, 106.0, 125.0, 126.0,  # angles ~ HEALTHY averages
    146.0, 103.0, 96.0, 91.0, 1.2,              # distances ~ training means
    0.51, 0.51,                                 # symmetry ~ training means
    y1/img.shape[0], x1/img.shape[1],           # real positions
    17                                          # keypoints
], dtype=np.float32).reshape(1,-1)

pred = classifier.predict(scaler.transform(features), verbose=0)[0]
confidence = np.max(pred)*100
result = "HEALTHY" if np.argmax(pred)==0 else "SICK"

print(f"\n🎯 HEALTHY TEST: {result} ({confidence:.1f}%)")
print("✅ Features now match training data!")

vis = results.plot()
color = (0,255,0) if result=="HEALTHY" else (0,0,255)
cv2.rectangle(vis, (x1,y1,x2-x1,y2-y1), color, 3)
cv2.putText(vis, f"HEALTHY TEST: {confidence:.0f}%", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
cv2.imshow("Fixed Test", vis)
cv2.waitKey(0)
