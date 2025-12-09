# train_posture_model_adv.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers, callbacks
import joblib

DATA_NPZ = "posture_adv_features.npz"
SCALER_OUT = "posture_scaler.pkl"
MODEL_OUT = "posture_mlp_adv.h5"

data = np.load(DATA_NPZ)
X = data["X"]
y = data["y"]

# simple imputation: replace NaN with column median
col_medians = np.nanmedian(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_medians, inds[1])

# train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# save scaler
joblib.dump(scaler, SCALER_OUT)

# build MLP
n_in = X_train_s.shape[1]
model = models.Sequential([
    layers.Input(shape=(n_in,)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(2, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=100, batch_size=16, callbacks=[es])

model.save(MODEL_OUT)
print(f"✔ Saved scaler: {SCALER_OUT}")
print(f"✔ Saved model: {MODEL_OUT}")
