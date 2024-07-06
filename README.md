
Image Classification with CIFAR-10 using VGG16
This project involves image classification on the CIFAR-10 dataset using a pre-trained VGG16 model with additional custom layers.

Project Structure
Data Preparation

The CIFAR-10 dataset is loaded and split into training and testing sets.
Class names are defined for easier interpretation of results.
Model Architecture

A pre-trained VGG16 model is used as the base model with its top layers removed (include_top=False).
Custom layers are added on top of the base model:
GlobalAveragePooling2D layer
Dense layer with 256 units and ReLU activation
Dense layer with 10 units (number of classes) and softmax activation
Model Compilation

The model is compiled with Adam optimizer and sparse categorical cross-entropy loss.
Accuracy is used as the evaluation metric.
Training the Model

The model is trained for 10 epochs with the training data.
Validation is performed on the test data.
Evaluation

The model's performance is evaluated on the test data.
Test accuracy is printed.
Inference

A function is defined for making predictions on new images.
For each image, the predicted class and confidence score are displayed along with the image.
