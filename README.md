# COVID-19 X-Ray Classification CNN

## Pneumonia Detection on COVID-19 Chest X-Rays using CNN

### Project Overview
This repository contains a custom Convolutional Neural Network (CNN) implemented from scratch using PyTorch for detecting pneumonia in COVID-19 chest X-ray images. The model achieves **97.50% accuracy** on the test dataset, demonstrating both the effectiveness of the architecture and my understanding of deep learning fundamentals.

## Model Architecture
The CNN architecture implemented in this project consists of:
- **4 convolutional layers** with increasing filter sizes (32, 64, 128, 256)
- **Max pooling layers** after each convolution for dimensionality reduction
- **ReLU activation functions** to introduce non-linearity
- **Two fully connected layers** (512 neurons in the hidden layer)
- **Output layer** for classification

## Performance Metrics
The model was evaluated on a held-out test set with impressive results:
- **Accuracy**: 97.50%
- **Precision**: 97.62%
- **Recall**: 97.50%
- **F1 Score**: 97.50%

## Implementation Details
- **Framework**: PyTorch
- **Training**: 8 epochs with Adam optimizer (learning rate = 0.001)
- **Loss Function**: Cross-Entropy Loss
- **Input Processing**: Images resized to 256×256 and normalized
- **Data Augmentation**: Basic transformations applied to improve model generalization

## Training Progress
The model showed consistent improvement during training:
```
Epoch [1/8], Loss: 1.1632, Accuracy: 54.73%
Epoch [2/8], Loss: 0.3136, Accuracy: 83.78%
Epoch [3/8], Loss: 0.2540, Accuracy: 91.89%
Epoch [4/8], Loss: 0.2094, Accuracy: 94.59%
Epoch [5/8], Loss: 0.1705, Accuracy: 93.92%
Epoch [6/8], Loss: 0.1189, Accuracy: 95.27%
Epoch [7/8], Loss: 0.0973, Accuracy: 96.62%
Epoch [8/8], Loss: 0.0721, Accuracy: 96.62%
```

## Purpose
This project was developed to demonstrate:
1. My understanding of CNN architectures and implementation from scratch.
2. Ability to process and prepare specialized medical imaging data for deep learning.
3. Skills in model training, optimization, and evaluation.
4. Practical application of deep learning to a critical healthcare problem related to COVID-19.


## Requirements
- Python
- PyTorch
- torchvision
- scikit-learn
- pandas
