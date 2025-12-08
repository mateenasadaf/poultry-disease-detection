import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models


BASE_DIR = r"C:\Users\mateena sadaf\Desktop\poultry disease\datasets\disease_dataset"
IMG_SIZE = (299, 299)
BATCH_SIZE = 16

train_dir = os.path.join(BASE_DIR, "train")
val_dir   = os.path.join(BASE_DIR, "valid")
test_dir  = os.path.join(BASE_DIR, "test")

train_ds = image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

val_ds = image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

test_ds = image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_ds.class_names
num_classes = len(class_names)

preprocess = tf.keras.applications.xception.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
val_ds   = val_ds.map(lambda x, y: (preprocess(x), y))
test_ds  = test_ds.map(lambda x, y: (preprocess(x), y))

base = Xception(weights="imagenet", include_top=False,
                input_shape=(299, 299, 3))
base.trainable = False

inputs = layers.Input(shape=(299, 299, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds, epochs=8, validation_data=val_ds)

model.save("xception_disease.h5")

with open("disease_classes.txt", "w") as f:
    for c in class_names:
        f.write(c + "\n")

print("✔ Model saved!")

