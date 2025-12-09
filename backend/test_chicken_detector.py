from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\chicken_detector.pt")

img = cv2.imread(r"C:\Users\mateena sadaf\Desktop\poultry disease\datasets\disease_dataset\test\coryza\003_jpg.rf.36a6cf6b7bb5a19176f3e1f767aa660c.jpg")  # change path
results = model(img)[0]
results.show()
