import os
import cv2
from ultralytics import YOLO
import shutil

# YOLOv8n COCO model
model = YOLO("yolov8n.pt")

SOURCE = r"C:\Users\mateena sadaf\Desktop\poultry disease\datasets\disease_dataset"
DEST = r"C:\Users\mateena sadaf\Desktop\poultry disease\datasets\chicken_yolo_dataset"

img_dest = os.path.join(DEST, "images")
lbl_dest = os.path.join(DEST, "labels")

os.makedirs(img_dest, exist_ok=True)
os.makedirs(lbl_dest, exist_ok=True)

def save_label(filename, box, img_w, img_h):
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h

    label_path = os.path.join(lbl_dest, filename.replace(".jpg",".txt"))
    with open(label_path, "w") as f:
        f.write(f"0 {xc} {yc} {w} {h}")

for split in ["train", "valid", "test"]:
    split_folder = os.path.join(SOURCE, split)

    for disease in os.listdir(split_folder):
        cls_folder = os.path.join(split_folder, disease)
        if not os.path.isdir(cls_folder):
            continue

        for imgname in os.listdir(cls_folder):
            if not imgname.lower().endswith((".jpg",".png",".jpeg")):
                continue

            img_path = os.path.join(cls_folder, imgname)
            img = cv2.imread(img_path)
            if img is None: continue

            h, w = img.shape[:2]

            results = model(img, verbose=False)[0]

            for box in results.boxes:
                cls_id = int(box.cls[0])

                # COCO: Class 14 = Bird
                if cls_id != 14:
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)

                # copy image
                newname = imgname.replace(" ", "_")
                shutil.copy(img_path, os.path.join(img_dest, newname))

                # save label
                save_label(newname, xyxy, w, h)

                print("Labeled:", newname)
                break

print("✔ Auto labeling completed!")
