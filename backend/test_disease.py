# test_disease_classifier.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# ----------------------------
# PATHS (change if needed)
# ----------------------------
MODEL_PATH = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\xception_disease.h5"
LABELS_PATH = r"C:\Users\mateena sadaf\Desktop\poultry disease\backend\disease_classes.txt"

# ----------------------------
# LOAD MODEL & LABELS
# ----------------------------
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    disease_labels = [line.strip() for line in f.readlines()]

# Xception preprocess
preprocess = tf.keras.applications.xception.preprocess_input

# ----------------------------
# FUNCTION TO PREDICT SINGLE IMAGE
# ----------------------------
def predict_disease(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found:", image_path)
        return
    
    # Show image
    cv2.imshow("Input Image", img)
    cv2.waitKey(10)

    # Resize + preprocess
    img_resized = cv2.resize(img, (299, 299))
    img_np = img_resized.astype("float32")
    img_np = preprocess(img_np)
    img_np = np.expand_dims(img_np, axis=0)

    # Predict
    pred = model.predict(img_np, verbose=0)
    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx])
    disease_name = disease_labels[idx]

    print("\n----------------------------")
    print("📌 IMAGE:", image_path)
    print("🐔 Predicted Disease:", disease_name)
    print("📊 Confidence:", round(confidence, 4))
    print("----------------------------")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ----------------------------
# RUN TEST
# ----------------------------
if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR TEST IMAGE
    test_image = r"C:\Users\mateena sadaf\Desktop\poultry disease\datasets\disease_dataset\test\ncd\ncd.302.jpg"

    predict_disease(test_image)
