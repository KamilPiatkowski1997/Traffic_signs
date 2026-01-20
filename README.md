# Traffic_signs

German Traffic Sign Recognition with CNN and Adversarial Examples

This project implements a convolutional neural network (CNN) to classify German traffic signs using the GTSRB dataset. The pipeline includes loading and preprocessing the data, augmenting training images, training a CNN model, evaluating performance, and demonstrating adversarial attacks that slightly modify images to fool the model.

The dataset contains 43 classes of traffic signs. Each image is 32×32 pixels and originally in RGB. For efficient learning, images are converted to grayscale, histogram-equalized, normalized, and reshaped to match the model input.

Data augmentation is applied during training, including small shifts, rotations, zoom, and shear, to improve the model’s generalization. The CNN consists of multiple convolutional and pooling layers, followed by a dense layer with dropout for regularization, and a softmax output layer for classification. The model is trained with the Adam optimizer and categorical crossentropy loss.

After training, the model is evaluated on a test set to determine its accuracy. It can also predict real-world traffic sign images after preprocessing them in the same way as the training data. Additionally, a simple adversarial attack demonstrates how small perturbations to an image can cause the model to misclassify it, highlighting potential vulnerabilities of deep learning models.

This implementation demonstrates both the effectiveness of CNNs for traffic sign recognition and the importance of considering robustness against adversarial inputs in safety-critical applications.
