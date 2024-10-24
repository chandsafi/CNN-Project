Facial Expression Recognition using CNN
Project Overview
This project implements a Convolutional Neural Network (CNN) to classify facial expressions from images. The model is trained to detect various emotions, such as happiness, sadness, anger, surprise, and more, based on facial images from the dataset.

The main goal of the project is to develop an efficient CNN architecture for recognizing facial expressions and improve its accuracy using data augmentation, hyperparameter tuning, and other optimization techniques.

Table of Contents
Project Overview
Dataset
Model Architecture
Installation
Training and Evaluation
Challenges Faced
Results
Future Improvements
License
Dataset
The dataset used for this project is the Expression in-the-Wild (ExpW) dataset. It contains around 40,000 labeled images, which are divided into:

Training set: 29,000 images
Test set: 1,000 images
Each image in the dataset is labeled with one of the following emotions: Happy, Sad, Angry, Fear, Surprise, Neutral, and Disgust.

Model Architecture
The CNN model consists of the following key components:

Convolutional Layers: Used for feature extraction from input images.
Max Pooling Layers: To reduce the dimensionality of the feature maps.
Fully Connected Layers: For final classification of facial expressions.
Dropout: Applied for regularization to prevent overfitting.
Architecture Summary:
Input: 48x48 grayscale images.
Conv -> Max Pool -> Conv -> Max Pool -> Dense -> Output (Softmax).
Installation
To run this project on your local machine, follow these steps:

Copy code
python train.py
Training and Evaluation
The model is trained using the following settings:

Loss Function: Categorical Cross-Entropy.
Optimizer: Adam optimizer with an initial learning rate of 0.001.
Batch Size: 32.
Number of Epochs: 50.
After training, the model is evaluated on the test dataset. The metrics used for evaluation include accuracy, precision, recall, and F1-score.

Challenges Faced
Internet Connectivity Issues: Training the model was challenging due to intermittent internet connectivity, which caused delays in the training process.
Complex CNN Architecture: The initial model was too complex for my local machine's computational resources, leading to troubleshooting and redesigning the model.
Suboptimal Accuracy: Despite applying data augmentation and hyperparameter tuning, the accuracy was not as high as expected.
Results
The model achieved an accuracy of 50.17% on the validation dataset. Precision, recall, and F1-score were computed for each emotion class, with the model performing better on emotions like "Happy" but struggling with others like "Disgust" and "Fear."

Future Improvements
Model Optimization: Try using a pre-trained model like VGGFace or ResNet to improve accuracy and reduce training time.
Data Augmentation: Incorporate more advanced augmentation techniques such as random occlusion or brightness adjustment.
Better Hyperparameter Tuning: Explore more hyperparameter combinations to boost the model's performance.
