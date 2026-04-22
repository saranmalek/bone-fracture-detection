import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "models/fracture_model.h5"

def load_model():
    try:
        # Remove safe_mode - it's deprecated in newer Keras versions
        model = keras_load_model(
            MODEL_PATH,
            compile=False
        )
        return model
    except Exception as e:
        print(f"Warning: Model loading issue: {e}")
        # Fallback with custom_objects handling
        model = keras_load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=None
        )
        return model
