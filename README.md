# 😷 MaskGuard AI - Face Mask Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## 📌 Project Overview
**MaskGuard AI** is a deep learning-based computer vision system designed to detect whether a person is wearing a face mask or not in real-time. It utilizes **Transfer Learning** with **MobileNetV2** for high efficiency and accuracy, making it suitable for deployment on resource-constrained devices.

The system includes a **Streamlit Web Dashboard** for easy interaction and a command-line utility for batch processing.

---

## 🚀 Key Features
- **Real-Time Detection**: instantly identifies masked and unmasked faces.
- **High Accuracy (~99%)**: Trained on a balanced dataset of "With Mask" and "Without Mask" images.
- **Robust Face Detection**: Uses a multi-stage **Haar Cascade** fallback system to detect faces even in difficult lighting or angles.
- **Interactive UI**: A modern, dark-themed dashboard powered by **Streamlit**.
- **Visual Feedback**: Clearly marks faces with Green (Safe) or Red (Risk) bounding boxes.

---

## 🧠 AI & Machine Learning Tech Stack

### 1. The Core Model: **MobileNetV2**
We use **MobileNetV2** as the base model for feature extraction. 
- **Why MobileNetV2?** It is a lightweight Convolutional Neural Network (CNN) designed for mobile and embedded vision applications. It uses "depthwise separable convolutions" to reduce the number of parameters while maintaining high accuracy.
- **Transfer Learning:** We load the model with weights pre-trained on **ImageNet** and freeze the base layers. We then add a custom "head" model (Fully Connected Layers) which is trained specifically for our mask dataset.

### 2. Model Architecture
- **Input Layer:** 224x224 RGB Images.
- **Base Model:** MobileNetV2 (Pre-trained, Frozen).
- **Pooling:** AveragePooling2D (7x7).
- **Flatten Layer:** Converts 2D feature maps to 1D vector.
- **Dense Layer:** 128 Neurons (ReLU Activation).
- **Dropout:** 0.5 (To prevent overfitting).
- **Output Layer:** 2 Neurons (Softmax Activation) -> `[Mask, No_Mask]`.

### 3. Face Detection Algorithm
We use **OpenCV's Haar Cascade Classifiers** to locate faces in an image before passing them to the mask detector. The system implements a robust fallback mechanism:
1.  Tries `haarcascade_frontalface_alt2.xml` (High Precision).
2.  Fallbacks to `haarcascade_frontalface_default.xml` (High Recall).
3.  Adjusts parameters (`minNeighbors`) dynamically to catch obscure faces.

---

## 📂 Project Structure
```bash
Facemask-Detection/
│
├── dataset/                   # Training dataset (with_mask / without_mask)
├── mask_detector.keras        # The trained AI Model file
├── train_model.py             # Script to train the model from scratch
├── detect_mask_image.py       # Script to detect masks in a single image (CLI)
├── streamlit_app.py           # The main Web Application (Dashboard)
├── requirements.txt           # Python dependencies
└── README.md                  # Project Documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- A virtual environment (recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Facemask-Detection.git
cd Facemask-Detection
```

### 2. Create and Activate Virtual Environment
```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` is missing, install manually: `pip install tensorflow opencv-python imutils matplotlib scikit-learn streamlit streamlit-extras`)*

---

## 💻 How to Run

### 1. Run the Web Dashboard (Recommended)
This launches the modern UI where you can upload images and see analytics.
```bash
streamlit run streamlit_app.py
```
- Open the URL provided in the terminal (usually `http://localhost:8501`).
- Upload an image and click **"Run Analysis"**.

### 2. Run Image Test via Command Line
To test a specific image without the web UI:
```bash
python detect_mask_image.py --image dataset/with_mask/image_name.jpg
```

### 3. Re-Train the Model
If you want to train the model on a new dataset:
1.  Place your images in `dataset/with_mask` and `dataset/without_mask`.
2.  Run the training script:
```bash
python train_model.py
```
This will generate a new `mask_detector.keras` file and a training accuracy plot `plot.png`.

---

## 📊 Performance
The model was trained using the **Adam Optimizer** with a learning rate of `1e-4` and **Binary Crossentropy** loss.
- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~99%

---

## 📜 License
This project is open-source and available for educational purposes.

---
*Created by Ahmad*
