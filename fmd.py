import pygame
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def play_beep():
    pygame.mixer.init()
    beep_sound = pygame.mixer.Sound("mixkit-facility-alarm-908.wav")  # Replace "beep.wav" with the path to your beep sound file
    beep_sound.play()
# Load the FaceNet model
faceNet = cv2.dnn.readNet("C:\Users\sreel\Downloads\Real-Time Face Mask Detection OpenCV Python\face_detector\deploy.prototxt","res10_300x300_ssd_iter_140000.caffemodel")

# Load the MaskNet model
maskNet = load_model("mask_detector.model")
def process_image(frame, faceNet, maskNet):
    # Your image processing logic here
    # Resize the frame for faster processing (optional)
    resized_frame = cv2.resize(frame, (300, 300))
    
    # Perform face detection using faceNet
    blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Adjust confidence threshold as needed
            # Compute the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Perform mask prediction on detected faces
    predictions = []
    for face in faces:
        # Preprocess the face for the maskNet
        faceBlob = cv2.dnn.blobFromImage(face, 1.0, (224, 224), (104.0, 177.0, 123.0))
        maskNet.setInput(faceBlob)

        # Make predictions
        preds = maskNet.predict(faceBlob)
        predictions.append(preds)

    return locs, predictions
def detect_and_predict_mask(frame, faceNet, maskNet):
    # Your face detection and mask prediction logic here
    # Process the image to detect faces and predict masks
    locs, predictions = process_image(frame, faceNet, maskNet)

    results = []

    # Loop over the detected faces and their corresponding predictions
    for (box, pred) in zip(locs, predictions):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred[0]

        # Determine the class label and color for the bounding box
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Include the probability in the label
        label = f"{label}: {max(mask, withoutMask):.2f}"

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        results.append((box, label))

    return frame, results
# Load your models and initialize the video stream as you did before
# Ask the user for input (image or live stream)
    print("Choose an option:")
    print("1. Image")
    print("2. Live Stream")
option = input("Enter the option (1 or 2): ")

if option == "1":
    # Process an image
    image_path = input("Enter the path to the image: ")
    frame = cv2.imread(image_path)
    process_image(frame, faceNet, maskNet)
elif option == "2":
    # Process a live stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
   
    sound_playing = False  # Initialize the sound_playing flag
   
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        # Rest of your live stream processing logic here
        # ...
        # ...
       
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
else:
    print("Invalid option. Please choose 1 for image processing or 2 for live stream processing.")
