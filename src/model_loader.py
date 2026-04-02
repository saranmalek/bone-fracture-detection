import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model

MODEL_PATH = "models/fracture_model.h5"

def load_model():
    model = keras_load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )
    return model
