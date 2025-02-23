# Deep_Fake_Detection-


---

# **Predicting AI-Generated vs. Real Images**  

### 📌 **Project Overview**  
With the rise of **AI-generated images**, distinguishing between real and synthetic images has become a crucial task. This project implements **Machine Learning (ML) and Deep Learning (DL) models** to classify images as either AI-generated or real.  

### 🚀 **Key Features**  
- **Preprocessing & Feature Engineering**: Image cleaning, normalization, and class balancing (SMOTE).  
- **Machine Learning Models**: Support Vector Machine (SVM) and K-Nearest Neighbors (KNN).  
- **Deep Learning Approach**: A **CNN-based model** with five convolutional layers, batch normalization, and dropout regularization.  
- **Performance Evaluation**: Metrics such as accuracy, precision, recall, and F1-score.  

---

## 📂 **Dataset & Preprocessing**  
- **Data Cleaning**: Removed corrupted and irrelevant images.  
- **Feature Extraction**: Extracted key image features for ML models.  
- **Normalization**: Standardized image features for better model performance.  
- **Data Augmentation**: Applied rotation, zoom, and brightness adjustments.  
- **Class Balancing**: Used **SMOTE** to handle imbalanced data.  

---

## 🏗 **Model Implementation**  
### **1️⃣ Support Vector Machine (SVM)**  
- Tested **Linear & RBF kernels**.  
- Hyperparameter tuning using **Grid Search**.  
- **10-fold Cross-Validation** for robustness.  

### **2️⃣ K-Nearest Neighbors (KNN)**  
- Used **Euclidean Distance** for similarity measurement.  
- Determined the best **K-value via cross-validation**.  

### **3️⃣ Convolutional Neural Network (CNN)**  
- **Architecture:**  
  - 5 **Convolutional Layers** (32, 64, 128, 256, 512 filters).  
  - **Batch Normalization & Dropout** for regularization.  
  - Fully Connected **Dense Layer (512 neurons)**.  
  - **Sigmoid Activation** for binary classification.  
- **Training Details:**  
  - Optimizer: **Adam** (LR = 0.0005).  
  - Loss Function: **Binary Cross-Entropy**.  
  - **Callbacks:** EarlyStopping & ReduceLROnPlateau.  

---

## 📊 **Results**  
🔹 **SVM & KNN Performance:** (Include accuracy, precision, recall, and F1-score results).  
🔹 **CNN Performance:** (Mention accuracy & generalization findings).  

---

## 💻 **How to Run the Project**  
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```
### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```
### **3️⃣ Run the Notebook**  
Open **Jupyter Notebook** and run `AI_Image_Classification.ipynb`.  

---

## 🎯 **Future Improvements**  
- **Enhancing the CNN model** with more complex architectures (e.g., ResNet, EfficientNet).  
- **Testing additional ML models** (e.g., Random Forest, XGBoost).  
- **Expanding dataset** for better generalization.  

---

## 🤝 **Contributors**  
- **Hana El Gabry**  
- Zaina ahmed
- noor Akram
- Malak mohamed
- Khetam almasarany  

