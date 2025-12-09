import os
import numpy as np
import cv2
from ultralytics import YOLO

# Pose model
POSE_MODEL = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\yolov8n-pose.pt"

# Your dataset
DATASET = r"C:\Users\mateena sadaf\Desktop\poultry disease\datasets\posture_dataset"

model = YOLO(POSE_MODEL)

X = []
y = []

# Two classes
labels = ["healthy", "sick"]


def normalize_keypoints(keypoints, box):
    """Normalize keypoints relative to the bounding box."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1

    features = []
    for (x, y) in keypoints:
        nx = (x - x1) / w
        ny = (y - y1) / h
        features.extend([nx, ny])

    return features


for label_index, label_name in enumerate(labels):

    folder = os.path.join(DATASET, label_name)

    for filename in os.listdir(folder):

        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        result = model(img, verbose=False)[0]

        # If no keypoints or no detection
        if len(result.keypoints) == 0 or len(result.boxes) == 0:
            continue

        # Get bounding box
        box = result.boxes[0].xyxy[0].cpu().numpy().astype(int)

        # Get keypoints
        kpts = result.keypoints.xy[0].cpu().numpy()

        features = normalize_keypoints(kpts, box)

        X.append(features)
        y.append(label_index)


# Save everything
np.savez("posture_features.npz", X=np.array(X), y=np.array(y))
print("✔ Posture features extracted and saved!")
