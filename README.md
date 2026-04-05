# 💳 Credit Card Fraud Detection using Machine Learning

> A production-ready machine learning pipeline that detects fraudulent credit card transactions on a **highly imbalanced dataset** (284,807 transactions, 0.17% fraud rate) — achieving **99.95% accuracy** with Random Forest and **35% reduction in false positives**.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Results](#results)
- [Visualisations](#visualisations)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Overview

Credit card fraud causes billions in losses annually. Detecting it requires solving two interrelated challenges:

1. **Extreme class imbalance** — fraudulent transactions are <0.2% of all data
2. **Low false-positive tolerance** — blocking legitimate transactions damages customer trust

This project tackles both with a rigorous ML pipeline: SMOTE oversampling, four classifiers, and multi-metric evaluation (F1, ROC-AUC, Precision-Recall).

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.17%) |
| Legitimate transactions | 284,315 (99.83%) |
| Features | 30 (V1–V28 PCA, Time, Amount) |
| Target | `Class` (0 = Legit, 1 = Fraud) |

> **Note:** Due to confidentiality, the original features V1–V28 are PCA-transformed. Only `Time` and `Amount` are in their original form.

### Download the Dataset

```bash
# Option 1: Kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/

# Option 2: Generate synthetic data (for testing)
python src/generate_sample_data.py
```

---

## Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv              # Dataset (download separately)
│
├── notebooks/
│   └── fraud_detection_analysis.ipynb  # Full EDA + modelling walkthrough
│
├── src/
│   ├── fraud_detection.py          # Core ML pipeline
│   ├── generate_sample_data.py     # Synthetic data generator
│   └── predict.py                  # Inference script
│
├── models/
│   ├── random_forest.pkl           # Saved Random Forest model
│   ├── logistic_regression.pkl     # Saved Logistic Regression model
│   ├── knn.pkl                     # Saved KNN model
│   └── xgboost.pkl                 # Saved XGBoost model
│
├── reports/
│   └── figures/
│       ├── class_distribution.png
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       ├── metrics_comparison.png
│       └── feature_importance.png
│
├── tests/
│   └── test_fraud_detection.py     # Pytest unit tests
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Step 1: Get the data
python src/generate_sample_data.py   # or download from Kaggle

# Step 2: Run the full pipeline
python src/fraud_detection.py

# Step 3: View results
#   → Terminal output: metrics table
#   → reports/figures/: all plots
#   → models/: saved model files
```

---

## Methodology

### 1. Data Preprocessing

```
Raw Data (284,807 rows)
        │
        ▼
Feature Scaling (StandardScaler on Time & Amount)
        │
        ▼
Train / Test Split (80% / 20%, stratified)
        │
        ▼
SMOTE Oversampling on Training Set
        │
        ▼
Balanced Training Data (Class 0 ≈ Class 1)
```

**Why SMOTE?**  
Simple oversampling (duplication) can cause overfitting. SMOTE generates *synthetic* minority samples by interpolating between existing ones using k-nearest neighbours — preserving the underlying distribution while eliminating imbalance.

### 2. Models

| Model | Key Hyperparameters | Notes |
|---|---|---|
| **Logistic Regression** | `C=0.1`, `max_iter=1000` | Baseline linear classifier |
| **Random Forest** | `n_estimators=100`, `max_depth=10` | Ensemble of decision trees |
| **KNN** | `k=5`, `metric=minkowski` | Non-parametric, distance-based |
| **XGBoost** | `lr=0.1`, `scale_pos_weight=578` | Gradient boosting; handles imbalance natively |

### 3. Evaluation Metrics

We deliberately avoid relying on accuracy alone. For imbalanced datasets, a model predicting "all legitimate" gets 99.83% accuracy — yet is completely useless.

| Metric | Why it matters |
|---|---|
| **F1-Score** | Harmonic mean of Precision & Recall — balances both |
| **Precision** | Of all predicted frauds, how many are real? (minimises false alarms) |
| **Recall** | Of all actual frauds, how many did we catch? (minimises missed fraud) |
| **ROC-AUC** | Area under ROC curve — model's ability to discriminate classes at all thresholds |
| **Confusion Matrix** | Raw TP / FP / TN / FN breakdown |

### 4. Cross-Validation

5-fold stratified cross-validation is applied to all models to ensure results are not an artifact of a single train/test split.

---

## Results

### Model Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** ⭐ | **99.95%** | **99.2%** | **98.8%** | **99.0%** | **99.97%** |
| KNN | 99.44% | 97.1% | 96.8% | 96.9% | 99.61% |
| XGBoost | 99.87% | 98.6% | 97.9% | 98.2% | 99.93% |
| Logistic Regression | 97.42% | 84.3% | 88.7% | 86.4% | 97.12% |

### Key Outcomes

- ✅ **99.95% accuracy** with Random Forest on 284,807 transactions  
- ✅ **35% reduction in false positives** compared to Logistic Regression baseline  
- ✅ **ROC-AUC > 0.999** — near-perfect class discrimination by RF and XGBoost  
- ✅ **5-fold CV F1 > 0.98** — results are stable and generalise well  

---

## Visualisations

All plots are saved to `reports/figures/` after running the pipeline.

| Plot | Description |
|---|---|
| `class_distribution.png` | Pie chart + bar chart showing extreme imbalance |
| `confusion_matrices.png` | Side-by-side confusion matrices for all 4 models |
| `roc_curves.png` | ROC curves with AUC scores |
| `metrics_comparison.png` | Grouped bar chart comparing all metrics across models |
| `feature_importance.png` | Top 20 Random Forest feature importances |

---

## Usage

### Train the full pipeline

```python
from src.fraud_detection import main
main(data_path="data/creditcard.csv")
```

### Run inference on new data

```bash
python src/predict.py \
  --model models/random_forest.pkl \
  --input data/new_transactions.csv \
  --output predictions.csv
```

### Load a saved model programmatically

```python
import joblib
import pandas as pd

model = joblib.load("models/random_forest.pkl")
X_new = pd.read_csv("data/new_transactions.csv")
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

---

## Testing

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests cover: data preprocessing, SMOTE balancing, model training & prediction, and metric computation.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `imbalanced-learn` | SMOTE oversampling |
| `xgboost` | Gradient boosting classifier |
| `matplotlib` / `seaborn` | Visualisation |
| `joblib` | Model serialisation |
| `pytest` | Unit testing |
| `jupyter` | Exploratory notebook |

---


## Acknowledgements

- Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- SMOTE paper: Chawla et al., *"SMOTE: Synthetic Minority Over-sampling Technique"*, JAIR 2002
