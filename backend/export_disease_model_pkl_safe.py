# export_disease_model_pkl_safe.py
import joblib
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "xception_disease.h5"
CLASS_PATH = "disease_classes.txt"
OUTPUT_PKL = "disease_classifier2.pkl"   # this will overwrite existing

# load keras model
model = load_model(MODEL_PATH)

# load classes
with open(CLASS_PATH, "r") as f:
    labels = [l.strip() for l in f.readlines()]

# minimal wrapper (picklable). Keeps only necessary things:
class DiseaseClassifierWrapper:
    def __init__(self, model, labels):
        self.model = model   # Keras model object (not the best for cross-env, but works)
        self.labels = labels

    def predict(self, img_array):
        """
        Input: img_array shape (299,299,3) float32 scaled 0-1
        Output: (label, confidence)
        """
        arr = np.expand_dims(img_array, axis=0)
        preds = self.model.predict(arr, verbose=0)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        return self.labels[idx], conf

# create wrapper and save
wrapper = DiseaseClassifierWrapper(model, labels)
joblib.dump(wrapper, OUTPUT_PKL)
print("Saved new PKL to:", OUTPUT_PKL)
