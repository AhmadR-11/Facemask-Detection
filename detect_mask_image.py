# detect_mask_image.py
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

def mask_image():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	ap.add_argument("-f", "--face", type=str,
		default="face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
		default="mask_detector.keras",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	# Using OpenCV's built-in Haar Cascade for simplicity and portability
	# Alternatively, if you have a DNN model, we could use that. 
	# For "best concept" typically DNN is better, but requires files.
	# We'll use Haar Cascade if available, or try to load a DNN if the user provides it.
	
	# Assuming Haar Cascade for now as it's built-in strictly or readily available. 
	# Actually, to make it robust, let's use the one included in cv2 data.
	cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
	faceNet = cv2.CascadeClassifier(cascade_path)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	model = load_model(args["model"])

	# load the input image
	image = cv2.imread(args["image"])
	orig = image.copy()
	(h, w) = image.shape[:2]

	# detect faces in the image
	# Haar cascade detection
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceNet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

	# loop over the detections
	print(f"[INFO] found {len(faces)} faces")
	
	for (x, y, w_box, h_box) in faces:
		# extract the face ROI
		face = image[y:y + h_box, x:x + w_box]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# pass the face through the model to determine if the face has a mask or not
		(mask, withoutMask) = model.predict(face)[0]

		# determine the class label and color we'll use to draw the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output frame
		cv2.putText(image, label, (x, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (x, y), (x + w_box, y + h_box), color, 2)

	# show the output image
	cv2.imshow("Output", image)
	cv2.waitKey(0)
	
if __name__ == "__main__":
	mask_image()
