# 💳 Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset is highly imbalanced, making fraud detection a challenging classification problem. The objective is to build a model that can accurately identify fraudulent transactions while minimizing false positives.

---

## 📌 Table of Contents
- Project Overview  
- Dataset  
- Methodology  
- Results  
- Conclusion  
- Installation  
- Usage  
- Acknowledgments  

---

## 📊 Project Overview

This project follows a complete machine learning workflow:

- Data preprocessing and feature scaling  
- Exploratory Data Analysis (EDA)  
- Handling class imbalance  
- Training multiple machine learning models  
- Evaluating and selecting the best model  

---

## 📁 Dataset

The dataset used contains anonymized credit card transactions with the following features:

- **Time:** Time elapsed between transactions  
- **Amount:** Transaction amount  
- **V1 to V28:** PCA-transformed features  
- **Class:** Target variable (0 = Legitimate, 1 = Fraud)  

🔹 The dataset is highly imbalanced, with fraudulent transactions forming a very small percentage of the data.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Checked for missing values and duplicates  
- Scaled `Amount` and `Time` features using StandardScaler  
- Normalized data for better model performance  

---

### 2. Exploratory Data Analysis (EDA)
- Visualized class distribution (fraud vs non-fraud)  
- Correlation heatmap to identify important features  
- Analysis of transaction patterns  

---

### 3. Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)**  
- Balanced the dataset to improve fraud detection  

---

### 4. Model Training
- Split data into training and testing sets  
- Trained multiple models:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - XGBoost  

---

### 5. Model Evaluation
- Evaluated using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - ROC-AUC Score  
- Used confusion matrix to analyze predictions  

---

## 📈 Results

| Model               | Performance |
|--------------------|------------|
| Logistic Regression | Good baseline performance |
| Decision Tree       | Moderate performance |
| Random Forest       | Improved accuracy |
| XGBoost             | Best overall performance |

---

## ✅ Conclusion

- Machine learning models can effectively detect fraudulent transactions  
- Handling class imbalance is crucial for better fraud detection  
- **XGBoost performed best** among all models  
- Recall is a critical metric in fraud detection to minimize missed fraud cases  

