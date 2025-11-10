# 🫀 Cardiovascular Disease Analysis and Prediction using Deep Learning  

This project explores the application of **Deep Learning** models for **Cardiovascular Disease (CVD) Prediction** using **ECG image data**.  
It demonstrates how **Convolutional Neural Networks (CNN)** and **VGG16 (Transfer Learning)** can automatically extract features from medical images to predict disease presence accurately.  

Developed as part of a research initiative at **VIT-AP University**, this study also utilizes **GridSearchCV** for hyperparameter optimization and follows a complete **data preprocessing → training → evaluation** pipeline.  

---

## 🔍 Overview  

Cardiovascular diseases (CVDs) are among the leading causes of global mortality.  
Traditional diagnostic methods such as ECG interpretation rely heavily on human expertise and are prone to error.  
This project proposes an **AI-driven approach** for automating and improving diagnostic accuracy using **deep learning models** that learn to recognize disease-specific ECG patterns.

---

## 🎯 Objectives  

- Develop deep learning models to predict **cardiovascular disease** from **ECG images**.  
- Compare the performance of **Custom CNN** and **Pre-trained VGG16** models.  
- Optimize the CNN architecture using **GridSearchCV** for the best hyperparameter combination.  
- Evaluate models based on **accuracy, precision, recall, and F1-score**.  
- Demonstrate the potential of AI in enhancing **automated clinical decision support systems**.  

---

## 🧠 Key Concepts  

### 🔹 Deep Learning  
Deep learning enables **automatic feature extraction** from medical data without manual engineering.  
- **CNN:** Learns spatial hierarchies and texture patterns from ECG images.  
- **VGG16:** A deep pre-trained network (16 layers) capable of extracting complex medical image features.  

### 🔹 GridSearchCV  
Used for **hyperparameter tuning** in CNN:
- Tested different combinations of learning rate, kernel size, dropout rate, and batch size.  
- Improved training efficiency and accuracy through **K-Fold cross-validation**.  

---

## ⚙️ Project Architecture  

