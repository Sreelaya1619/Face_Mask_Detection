# Face Mask Detection using CNN

This project involves building a Convolutional Neural Network (CNN) model to detect whether a person is wearing a face mask or not. The model is trained on a dataset of images labeled as "with mask" and "without mask."

## Features

- Detects face masks in images using a trained CNN model.
- Achieves high accuracy in real-time mask detection.
- Provides a foundation for deployment in public safety systems or health monitoring applications.

## Technologies Used

- **Python**: Programming language.
- **TensorFlow/Keras**: Deep learning libraries for building and training the CNN model.
- **OpenCV**: For real-time video processing and face detection.
- **Matplotlib**: To visualize data and training progress.

## Dataset

- The dataset contains two categories: 
  1. **With Mask**: Images of people wearing face masks.
  2. **Without Mask**: Images of people without face masks.

- You can use publicly available datasets like the "Face Mask Detection" dataset from Kaggle or create your own dataset.

## Model Architecture

The CNN model is built using the following layers:
- Convolutional layers for feature extraction.
- MaxPooling layers for downsampling.
- Fully connected layers for classification into "Mask" and "No Mask" categories.

## How to Run the Project

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/face-mask-detection.git

2.Install the required dependencies:

    pip install -r requirements.txt

3.Download the dataset and place it in the appropriate directory.

4.Train the model:

    python train.py

File Structure

    face-mask-detection/
    ├── dataset/
    │   ├── with_mask/
    │   └── without_mask/
    ├── models/
    ├── train.py
    ├── detect_mask.py
    ├── requirements.txt
    └── README.md

usage

* Train the model with the dataset and evaluate its performance.
* Use the pre-trained model to detect face masks in real-time using your webcam or process images from a folder.
