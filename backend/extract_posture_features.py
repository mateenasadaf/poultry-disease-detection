import os
import numpy as np
from ultralytics import YOLO
import cv2

POSE_MODEL = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\yolov8n-pose.pt"
DATASET = r"C:\Users\mateena sadaf\Desktop\poultry disease\datasets\posture_dataset"

model = YOLO(POSE_MODEL)

X, y = [], []
labels = ["healthy", "sick"]

def normalize_kpts(kpts, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    features = []
    for x, y in kpts:
        features.append((x - x1) / w)
        features.append((y - y1) / h)
    return features

for label_idx, label_name in enumerate(labels):
    folder = os.path.join(DATASET, label_name)
    for imgname in os.listdir(folder):
        if not imgname.lower().endswith((".jpg",".png",".jpeg")):
            continue

        img_path = os.path.join(folder, imgname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        result = model(img, verbose=False)[0]
        if len(result.keypoints) == 0 or len(result.boxes) == 0:
            continue

        box = result.boxes[0].xyxy[0].cpu().numpy().astype(int)
        kpts = result.keypoints.xy[0].cpu().numpy()

        features = normalize_kpts(kpts, box)
        X.append(features)
        y.append(label_idx)

np.savez("posture_features.npz", X=np.array(X), y=np.array(y))
print("✔ Posture features saved!")
