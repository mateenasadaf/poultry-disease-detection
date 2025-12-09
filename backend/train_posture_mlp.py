import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

data = np.load("posture_features.npz")
X = data["X"]
y = data["y"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = models.Sequential([
    layers.Input((X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=25, validation_data=(X_val, y_val))

model.save("posture_health_mlp.h5")
print("✔ Posture model saved!")
