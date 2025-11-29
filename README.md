# Ensemble Approach for Plant Disease Classification

This project presents a deep-learning–based cotton leaf and plant disease classification system using transfer learning, ensemble modeling, and a meta-learning stacking approach. The final ensemble model significantly improves prediction reliability and outperforms individual CNN models.



## 📂 Project Structure
├── Dataset/
│   ├── Fresh Cotton Leaf
│   ├── Fresh Cotton Plant
│   ├── Diseased Cotton Leaf
│   ├── Diseased Cotton Plant
├── Models/
│   ├── VGG19
│   ├── ResNet18
│   ├── InceptionV3
│   └── Meta-Model (Stacking)
├── Code/
│   ├── Training Scripts
│   ├── Model Evaluation
│   ├── Ensemble Learning
└── README.md

## 🌱 Project Overview

This work aims to automatically classify cotton leaf and plant diseases using image data and advanced deep learning techniques.

• Uses VGG19, ResNet18, and InceptionV3 individually.

• Combines VGG19 + ResNet18 using a stacked ensemble (meta-learning) for final prediction.

• Evaluates models using Accuracy, Precision, Recall, and F1-Score.

• Shows that the meta-model ensemble outperforms individual models.


## 🖼️ Dataset Details

The dataset includes four classes:

• Fresh Cotton Leaf
• Fresh Cotton Plant
• Diseased Cotton Leaf
• Diseased Cotton Plant

Data is split into:

• Training set
• Validation set
• Test set

Data augmentation techniques applied include:
• Horizontal & vertical flipping
• Zooming
• Rotations
• Scaling
• Normalization


## ⚙️ Pipeline / Workflow


1️⃣ Image Acquisition

• High-resolution cotton leaf and plant images
• Real-world lighting and orientation variations

2️⃣ Preprocessing

• Resize to 224×224
• Denoising
• Color normalization

3️⃣ Data Augmentation

• Flips
• Rotations
• Zoom transformations

4️⃣ Feature Extraction

• CNN-based feature extraction (VGG19, ResNet18, InceptionV3)

5️⃣ Image Classification

• Individual CNN models trained separately

6️⃣ Meta-Model

• Softmax outputs of VGG19 & ResNet18 concatenated
• Fed into a dense neural network meta-learner
• Produces final prediction

## 🧠 Models Used
**✔ VGG19 (Best Individual Model)**

• Pre-trained on ImageNet

• Modified with custom classifier layers

• Achieved 93% test accuracy

• Excellent precision, recall, and F1-score



**✔ ResNet18**

• Residual connections prevent vanishing gradients

• Achieved 60% test accuracy

• Lower performance due to class imbalance sensitivity 


**✔ InceptionV3**

• Strong multi-scale feature extraction

• Achieved 100% test accuracy, but considered overfitting

• Excluded from the ensemble 


**✔ Meta-Model (Stacking Ensemble)**

• Combines softmax outputs of VGG19 + ResNet18

• Trained on Level-1 dataset of 8-dimensional probability vectors

• Achieved 97.13% accuracy


## 📊 Evaluation Metrics


• Accuracy
• Precision
• Recall (Sensitivity)
• F1-Score
• Confusion Matrix

Each model was evaluated using a held-out test set of 106 images.

## 🏆 Model Performance Summary
| Model                | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| VGG19               | 93%      | 96%       | 93%    | 96%      |
| ResNet18            | 60%      | 60%       | 60%    | 60%      |
| InceptionV3         | 100%     | 98%       | 100%   | 98%      |
| Meta-Model Ensemble | 97.13%   | 97.27%    | 97.17% | 89.92%   |


## 🚀 Key Findings

• Stacked ensemble improved performance beyond individual models.
• InceptionV3 overfitted despite high accuracy.
• VGG19 offered the best balance of stability and accuracy among base models.
• Meta-learning produced the most robust classifier.
• Confirms ensemble learning is ideal for real-world crop disease detection
