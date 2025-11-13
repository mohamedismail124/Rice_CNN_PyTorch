 🌾 Rice-CNN-PyTorch

A Convolutional Neural Network (CNN) model built using **PyTorch** to classify different types of rice grains based on image data.  
This project demonstrates the process of data preprocessing, model building, training, evaluation, and saving the trained model.

---

 🚀 Project Overview

This project aims to automatically **classify rice varieties** using deep learning techniques.  
It uses a CNN architecture implemented with **PyTorch** for accurate image-based classification.

---

 🧠 Model Architecture

The CNN model consists of:
- 2 Convolutional layers with ReLU activation
- MaxPooling layers for downsampling
- 2 Fully Connected (Linear) layers
- Output layer for final classification

```python
class MyCNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(MyCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*32*32, 128)
        self.fc2 = nn.Linear(128, num_classes)
