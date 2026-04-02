from src.preprocess import preprocess_image
from src.model_loader import load_model
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
import numpy as np

model = load_model()

LAST_CONV_LAYER = "out_relu"  # Works for MobileNetV2


def predict(image_path):

    img = preprocess_image(image_path)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        result = "Normal"
        confidence = prediction
    else:
        result = "Fractured"
        confidence = 1 - prediction

    # Generate heatmap
    heatmap = make_gradcam_heatmap(img, model, LAST_CONV_LAYER)

    heatmap_path = overlay_heatmap(heatmap, image_path)

    return result, float(confidence), heatmap_path