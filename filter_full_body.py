import os
import cv2
from ultralytics import YOLO

# ---------------------------
# PATHS
# ---------------------------
SOURCE_BASE = r"C:\Users\mateena sadaf\Desktop\poultry disease\posture detection"
OUTPUT_BASE = r"C:\Users\mateena sadaf\Desktop\poultry disease\posture detection_filtered"

MODEL = YOLO("yolov8n.pt")  # built-in bird detector

os.makedirs(OUTPUT_BASE, exist_ok=True)

def keep_full_body(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Rule 1 - bounding box must cover enough area
    if h < img_h * 0.5:
        return False  # too small → partial bird

    # Rule 2 - width must be decent
    if w < img_w * 0.2:
        return False

    # Rule 3 - bird must be centered roughly
    if x1 > img_w*0.4 or x2 < img_w*0.4:
        pass  # optional

    return True


def process_split(split):
    src = os.path.join(SOURCE_BASE, split)
    dst = os.path.join(OUTPUT_BASE, split)
    os.makedirs(dst, exist_ok=True)

    print(f"\nScanning {split}...")

    for imgname in os.listdir(src):
        if not imgname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(src, imgname)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        results = MODEL(img, verbose=False)[0]

        if len(results.boxes) == 0:
            continue

        # Find "bird" class (class id = 14 in COCO)
        keep = False
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 14:  # bird
                xyxy = box.xyxy[0].cpu().numpy()
                if keep_full_body(xyxy, w, h):
                    keep = True
                    break

        if keep:
            cv2.imwrite(os.path.join(dst, imgname), img)

    print(f"Saved filtered images → {dst}")


for split in ["train", "test", "valid"]:
    process_split(split)

print("\n✔ FULL-BODY FILTERING DONE!")
