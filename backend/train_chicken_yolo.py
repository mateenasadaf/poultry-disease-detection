from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=r"C:\Users\mateena sadaf\Desktop\poultry disease\datasets\chicken_yolo_dataset\data.yaml",
    imgsz=640,
    epochs=50,
    batch=8
)

model.export(format="pt")
