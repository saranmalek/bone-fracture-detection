import os
from flask import Blueprint, request, jsonify
from src.predictor import predict

routes = Blueprint("routes", __name__)

UPLOAD_FOLDER = "static/uploads"

@routes.route("/predict", methods=["POST"])
def predict_xray():

    file = request.files["file"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(path)

    result, confidence, heatmap = predict(path)

    return jsonify({
        "prediction": result,
        "confidence": round(confidence*100,2),
        "heatmap": heatmap
    })