# Cardiovascular Disease Analysis and Prediction using Deep Learning  

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
```text
CardioVascular-Disease-Prediction/
│
├── CNN.ipynb             # Custom CNN model implementation  
├── VGG16.ipynb           # Transfer Learning with VGG16  
├── grid_search.ipynb     # CNN Hyperparameter optimization using GridSearchCV  
├── dataset/              # ECG images (Normal vs Abnormal)  
├── results/              # Model outputs, confusion matrices, and plots  
└── Project Report.pdf    # IEEE-style project documentation  
```

---

## 🧩 Methodology  

### **1️⃣ Data Preprocessing**  
- Normalization and resizing of ECG images.  
- Data augmentation (rotation, flip, brightness adjustment).  
- Splitting dataset into **70% training**, **15% validation**, **15% testing**.  

### **2️⃣ Model Development**  
- Implemented two architectures:
  - **Custom CNN:** Trained from scratch.  
  - **VGG16:** Fine-tuned for transfer learning.  

### **3️⃣ Hyperparameter Tuning**  
- Used **GridSearchCV** to find optimal values for:
  - Learning Rate  
  - Dropout Rate  
  - Batch Size  
  - Number of Filters  

### **4️⃣ Model Evaluation**  
- Metrics: **Accuracy**, **Precision**, **Recall**, **F1-score**.  
- Used **Confusion Matrix** for class-wise performance visualization.  
- Plotted **Accuracy vs Epochs** and **Loss vs Epochs** graphs for both models.  

---

## 📊 Dataset Description  

| Class | Patients | Total Images | Training Images |
|--------|-----------|---------------|-----------------|
| Normal ECG | 240 | 2880 | 239 |
| Abnormal Heartbeat | 172 | 2064 | 172 |
| **Total** | 412 | 4944 | 411 |

**Source:** ECG image dataset (classified as Normal and Abnormal Heartbeats).  

---

## 🧮 Mathematical Formulations  

**Convolution:**  
\[
FeatureMap = \sum (K[i,j] * I[i,j]) + b
\]

**ReLU Activation:**  
\[
f(x) = max(0, x)
\]

**Pooling Operation:**  
\[
P(i,j) = max(F(i+m, j+n))
\]

**Fully Connected Layer:**  
\[
y = W * x + b
\]

**Softmax Output:**  
\[
S_i = \frac{e^{z_i}}{\sum e^{z_j}}
\]

---

## 🧠 Experimental Results  

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|----------|-----------|
| **CNN (GridSearchCV Optimized)** | **93.64%** | 0.92 | 0.93 | 0.92 |
| **VGG16 (Transfer Learning)** | **76.00%** | 0.75 | 0.77 | 0.76 |

**CNN outperformed VGG16** in accuracy due to dataset size and optimization,  
while **VGG16 showed better feature extraction consistency** and generalization.  

---

### 📈 Graphs and Visualizations  
- **Training vs Validation Accuracy** (CNN & VGG16)  
- **Training vs Validation Loss**  
- **Confusion Matrices**  
- **Predicted ECG Output Samples**  

---

## 🧠 Tools and Technologies  

- **Language:** Python  
- **Frameworks:** TensorFlow, Keras  
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn  
- **Platform:** Google Colab (GPU-enabled: NVIDIA Tesla T4, 16GB VRAM)  
- **Model Optimization:** GridSearchCV  
- **Dataset Type:** ECG medical image data  

---

## 🔬 Research Highlights  

- Implementation demonstrates **deep learning’s potential** in healthcare AI.  
- **Automation of cardiovascular diagnosis** reduces human dependency and diagnostic error.  
- **GridSearchCV optimization** ensures the CNN achieves high generalization.  
- The study sets a foundation for **future clinical AI integration** (e.g., real-time monitoring, wearable devices).  

---

## 🚀 Future Enhancements  

- Integrate advanced architectures (ResNet, Inception, or Transformer-based).  
- Develop **real-time ECG diagnosis system** using wearable sensors.  
- Apply **Explainable AI (XAI)** for transparent medical predictions.  
- Extend to **multi-modal health data** (EHR + imaging).  

---

## 🏁 Conclusion  

This project successfully demonstrates how **deep learning** can enhance **cardiovascular disease diagnosis**.  
By combining **CNN**, **VGG16**, and **GridSearchCV**, the study achieved a **93.64% accuracy**,  
showcasing the impact of AI in medical imaging and predictive healthcare.  


