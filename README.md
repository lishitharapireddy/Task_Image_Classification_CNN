# Image Classification using CNN

## Project Overview

This project implements an Image Classification model using a Convolutional Neural Network (CNN) with TensorFlow and Keras. The model is trained on the MNIST handwritten digit dataset to classify images into one of ten digit classes (0–9).

## Objective

The objective of this project is to build, train, evaluate, and save a CNN model capable of accurately recognizing handwritten digits.

## Technologies Used

* Python 3.11
* TensorFlow
* Keras
* NumPy
* Matplotlib

## Dataset

The project uses the **MNIST Handwritten Digits Dataset**, which contains:

* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 pixels
* 10 output classes (Digits 0–9)

## Model Architecture

* Conv2D (32 filters, 3×3 kernel, ReLU)
* MaxPooling2D (2×2)
* Flatten
* Dense (128 neurons, ReLU)
* Dense (10 neurons, Softmax)

## Training Details

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Metrics: Accuracy
* Epochs: 5

## Results

* Test Accuracy: **98.66%**
* Test Loss: **0.0425**

## Project Structure

Task_Image_Classification_CNN/

* dataset/
* model/
* notebook/
* report/
* screenshots/
* requirements.txt
* README.md

## How to Run

1. Create and activate a virtual environment.
2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```
3. Run the project:

   ```
   python notebook/image_classification.py
   ```

## Author

Lishitha Rapireddy
