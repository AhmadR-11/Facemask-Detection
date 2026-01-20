import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

def verify():
    model_path = "mask_detector.keras"
    image_path = "dataset/with_mask/with_mask_1.jpg"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded.")

    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image via cv2.")
        return
        
    # Preprocess
    # We need to extract the face first ideally, but for a quick check on a dataset image 
    # (which are cropped faces usually? Let's assume dataset images are faces or contain them)
    # The training script loads images and resizes to 224x224. 
    # Let's resize the whole image to 224x224 and predict, assuming the dataset image is mostly face.
    # The training data loop does: load_img -> target_size=(224, 224) -> img_to_array -> preprocess_input.
    
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    print("Running prediction...")
    (mask, withoutMask) = model.predict(image)[0]

    label = "Mask" if mask > withoutMask else "No Mask"
    print(f"Prediction: {label}")
    print(f"Confidence: Mask={mask:.4f}, No Mask={withoutMask:.4f}")

if __name__ == "__main__":
    verify()
