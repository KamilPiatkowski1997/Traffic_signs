# Traffic_signs

German Traffic Sign Recognition with CNN and Adversarial Attacks

1. Project Overview

This project implements a Convolutional Neural Network (CNN) to classify German Traffic Signs using the GTSRB dataset.
The pipeline includes:

Dataset loading and exploration

Image preprocessing and augmentation

CNN model training and evaluation

Real-world image inference

Generation of adversarial examples using gradient-based perturbations

The model is trained to classify 43 different traffic sign classes.

2. Dataset

The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB).

Download

The dataset is cloned directly from Bitbucket:

https://bitbucket.org/jadslim/german-traffic-signs


It contains:

train.p – training data

valid.p – validation data

test.p – test data

signnames.csv – mapping of class IDs to traffic sign names

Image Properties

Resolution: 32 × 32

Channels: RGB (converted to grayscale during preprocessing)

Number of classes: 43

3. Environment & Dependencies

This project is designed to run in Google Colab.

Required Libraries

Python 3.x

TensorFlow / Keras

NumPy

OpenCV (cv2)

Matplotlib

Pandas

Pillow

Requests

All dependencies are imported at the start of the notebook.

4. Data Preprocessing

Each image undergoes the following steps:

Convert to grayscale

Histogram equalization (enhances contrast)

Normalization (pixel values scaled to [0, 1])

Reshaping to (32, 32, 1)

This improves learning efficiency and model stability.

5. Data Augmentation

To improve generalization, data augmentation is applied using ImageDataGenerator:

Width & height shifts

Zoom

Shear

Rotation

Augmentation is applied only to the training set.

6. Model Architecture

A custom CNN architecture is used:

4 Convolutional layers (ReLU activation)

2 MaxPooling layers

Fully connected dense layer (500 neurons)

Dropout for regularization

Softmax output layer (43 classes)

Optimizer & Loss

Optimizer: Adam

Loss function: Categorical Crossentropy

Metric: Accuracy

7. Model Training

Batch size: 50

Epochs: 10

Training uses augmented data

Validation is performed on a separate validation set

Training and validation loss and accuracy curves are plotted for performance monitoring.

8. Model Evaluation

The trained model is evaluated on the test dataset.

Outputs:

Test loss

Test accuracy

This provides an unbiased estimate of model performance.

9. Real-World Image Prediction

The model supports inference on real-world traffic sign images fetched from a URL.

Steps:

Download image

Resize to 32 × 32

Apply same preprocessing pipeline

Predict sign class

Display predicted sign name

10. Adversarial Attack (FGSM-style)

An adversarial attack is implemented using gradient-based perturbations.

Method

Compute gradients of the loss w.r.t. input image

Apply a small perturbation (ε = 0.1)

Observe change in model prediction

Purpose

Demonstrates how small, human-imperceptible changes can cause:

Misclassification

Reduced model robustness

Both original and adversarial predictions are displayed visually.

11. Results

The CNN achieves high accuracy on the test set

The model performs well on real-world images

Adversarial examples can successfully fool the classifier

12. Conclusion

This project demonstrates:

End-to-end traffic sign recognition

Effective CNN training with augmentation

Practical vulnerability of deep learning models to adversarial attacks

It highlights the importance of robust model design and adversarial defense techniques in real-world applications.

13. Future Improvements

Adversarial training for robustness

Use of deeper architectures (ResNet, MobileNet)

Support for colored inputs

Stronger attacks (PGD, BIM)
