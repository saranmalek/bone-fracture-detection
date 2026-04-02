import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models
import json
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -------------------------
# Configuration
# -------------------------

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20

DATASET_PATH = "data/processed"
MODEL_PATH = "models/fracture_model.h5"
HISTORY_PATH = "models/training_history.json"

# -------------------------
# Data Generator
# -------------------------
# Create generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

# Validation data
val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# -------------------------
# Model (Transfer Learning)
# -------------------------

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=base_model.input, outputs=output)

# -------------------------
# Compile
# -------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Train
# -------------------------

checkpoint = ModelCheckpoint(
    "models/best_model.h5",
    monitor="val_accuracy",
    save_best_only=True
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# -------------------------
# Save Model
# -------------------------

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)

print("Model saved to:", MODEL_PATH)

# -------------------------
# Save Training History
# -------------------------

with open(HISTORY_PATH, "w") as f:
    json.dump(history.history, f)

print("Training history saved.")