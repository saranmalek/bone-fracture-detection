import tensorflow as tf

MODEL_PATH = "models/fracture_model.h5"

def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model
