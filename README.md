
# Bone Fracture Detection using Deep Learning

## Project Overview
This project is an AI-powered Bone Fracture Detection System that analyzes X-ray images and predicts whether a bone is fractured or normal.

The system uses Deep Learning with TensorFlow and MobileNetV2 to classify X-ray images. It also uses Grad-CAM visualization to highlight the region where the fracture might be located.

The application includes:
- A trained deep learning model
- A Flask web application
- Image preprocessing
- Grad-CAM heatmap visualization
- A web interface to upload X-rays

Users can upload an X-ray image through the website and the AI model will analyze it and return:
- Prediction (Fractured / Normal)
- Confidence percentage
- Heatmap showing fracture location

---

## Technologies Used
TensorFlow – Deep learning model training and prediction  
Flask – Backend web server  
NumPy – Numerical operations on image data  
Pillow – Image processing  
OpenCV – Image reading and heatmap generation  
Matplotlib – Training visualization graphs  

---

## Project Folder Structure

bone-fracture-detection-project
│
├── api
│   └── routes.py
│
├── data
│   └── processed
│        ├── fractured
│        └── normal
│
├── models
│   ├── fracture_model.h5
│   └── training_history.json
│
├── notebooks
│   └── visualize_training.py
│
├── src
│   ├── train.py
│   ├── predictor.py
│   ├── preprocess.py
│   ├── model_loader.py
│   └── gradcam.py
│
├── static
│   ├── css
│   │    └── style.css
│   └── uploads
│
├── templates
│   └── index.html
│
├── main.py
├── requirements.txt
└── README.md

---

## How the System Works

1. User uploads an X-ray image through the web interface.
2. Flask backend receives the image.
3. The image is preprocessed (resized and normalized).
4. The trained deep learning model analyzes the image.
5. The model predicts whether the bone is fractured or normal.
6. Grad-CAM generates a heatmap to show the suspected fracture area.
7. The result and heatmap are returned to the web interface.

---

## Dataset Structure

data
 └── processed
      ├── fractured
      │     image1.jpg
      │     image2.jpg
      │
      └── normal
            image3.jpg
            image4.jpg

Each folder contains X-ray images of bones.

---

## Installation

Step 1 – Install Python 3.9 or newer

Check version:
python --version

Step 2 – Install Required Libraries

pip install tensorflow flask numpy pillow opencv-python matplotlib

or

pip install -r requirements.txt

---

## Training the Model

Run:

python src/train.py

This will:
1. Load dataset
2. Train the MobileNetV2 model
3. Save trained model to:

models/fracture_model.h5

Training history is saved in:

models/training_history.json

---

## Running the Web Application

Step 1 – Navigate to project folder

cd bone-fracture-detection-project

Step 2 – Start Flask server

python main.py

Step 3 – Open browser

http://127.0.0.1:5000

Step 4 – Upload X-ray image

The AI will show:
- Prediction
- Confidence percentage
- Fracture heatmap

---

## Explanation of Important Files

main.py  
Starts the Flask server.

routes.py  
Handles API requests and returns prediction results.

train.py  
Trains the deep learning model.

predictor.py  
Loads the trained model and predicts fracture from images.

preprocess.py  
Prepares images before sending them to the model.

gradcam.py  
Creates heatmaps showing where fractures are detected.

model_loader.py  
Loads the trained model from the models folder.

index.html  
Frontend interface to upload images and display results.

---

## Library Explanation

TensorFlow  
Used to build, train, and run the deep learning fracture detection model.

Flask  
Used to create the web server that connects the frontend with the AI model.

NumPy  
Used for numerical operations on image arrays.

Pillow  
Used for opening and handling image files.

OpenCV  
Used for image processing and generating Grad-CAM heatmaps.

Matplotlib  
Used for plotting training graphs such as accuracy and loss.

---

## Example Output

Prediction: Fractured  
Confidence: 92%

A heatmap will appear highlighting the fracture location.

---

## Conclusion

This project demonstrates how Deep Learning and Computer Vision can assist in medical image analysis by detecting fractures from X-ray images automatically.
