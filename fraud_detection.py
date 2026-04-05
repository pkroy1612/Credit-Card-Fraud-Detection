"""
Credit Card Fraud Detection - Main Pipeline
==========================================
Author: Your Name
Date: February 2024 - March 2024

This module implements the complete ML pipeline for credit card fraud detection,
including data preprocessing, SMOTE oversampling, model training, and evaluation.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & EXPLORATION
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the creditcard dataset from CSV."""
    print(f"[INFO] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}\n")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Print basic EDA statistics."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(df.describe())
    print("\n[INFO] Missing values:")
    print(df.isnull().sum())
    print("\n[INFO] Class distribution:")
    counts = df["Class"].value_counts()
    print(counts)
    fraud_pct = counts[1] / len(df) * 100
    print(f"\n  → Legitimate transactions : {counts[0]:,} ({100 - fraud_pct:.2f}%)")
    print(f"  → Fraudulent transactions : {counts[1]:,} ({fraud_pct:.4f}%)")
    print(f"  → Class Imbalance Ratio   : {counts[0] / counts[1]:.0f}:1\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the dataset:
    - Scale 'Amount' and 'Time' features
    - Drop originals after scaling
    - Split into train/test with stratification
    """
    print("[INFO] Preprocessing data...")

    scaler = StandardScaler()

    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"]   = scaler.fit_transform(df[["Time"]])
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train set : {X_train.shape[0]:,} samples")
    print(f"  Test set  : {X_test.shape[0]:,} samples\n")
    return X_train, X_test, y_train, y_test, scaler


def apply_smote(X_train, y_train):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to
    balance the training set.
    """
    print("[INFO] Applying SMOTE oversampling...")
    before = dict(zip(*np.unique(y_train, return_counts=True)))
    print(f"  Before SMOTE → Class 0: {before[0]:,}  |  Class 1: {before[1]:,}")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    after = dict(zip(*np.unique(y_res, return_counts=True)))
    print(f"  After  SMOTE → Class 0: {after[0]:,}  |  Class 1: {after[1]:,}\n")
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return a dictionary of all classifiers to train."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.1, solver="lbfgs", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, metric="minkowski", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=578,   # handles class imbalance natively
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
    }


def train_models(models: dict, X_train, y_train) -> dict:
    """Train all models and return fitted instances."""
    trained = {}
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  ✓ {name} trained successfully.")
    print()
    return trained


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute all metrics for a single model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model"    : name,
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall"   : recall_score(y_test, y_pred),
        "F1-Score" : f1_score(y_test, y_pred),
        "ROC-AUC"  : roc_auc_score(y_test, y_proba),
    }
    return metrics


def evaluate_all_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all models and print a summary table."""
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    results = []
    for name, model in trained_models.items():
        m = evaluate_model(name, model, X_test, y_test)
        results.append(m)
        print(f"\n  ── {name} ──")
        print(f"     Accuracy  : {m['Accuracy']:.4%}")
        print(f"     Precision : {m['Precision']:.4%}")
        print(f"     Recall    : {m['Recall']:.4%}")
        print(f"     F1-Score  : {m['F1-Score']:.4%}")
        print(f"     ROC-AUC   : {m['ROC-AUC']:.4%}")
    print()
    return pd.DataFrame(results).set_index("Model")


def cross_validate_models(trained_models: dict, X_train, y_train) -> None:
    """Run 5-fold stratified cross-validation on training data."""
    print("=" * 60)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in trained_models.items():
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="f1", n_jobs=-1)
        print(f"  {name:22s} → CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df["Class"].value_counts()
    labels = ["Legitimate (0)", "Fraudulent (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    axes[0].pie(counts, labels=labels, colors=colors,
                autopct="%1.2f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")

    axes[1].bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Transaction Counts", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[1].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_confusion_matrices(trained_models: dict, X_test, y_test) -> None:
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained_models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"],
                    ax=ax, linewidths=0.5, linecolor="white")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_roc_curves(trained_models: dict, X_test, y_test) -> None:
    plt.figure(figsize=(9, 7))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    for (name, model), color in zip(trained_models.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})",
                 color=color, linewidth=2.5)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)

    path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_metrics_comparison(results_df: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(results_df.index))
    width = 0.15
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, results_df[metric],
                      width, label=metric, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Performance Metrics Comparison — All Models",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(results_df.index, fontsize=11)
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_feature_importance(trained_models: dict, feature_names: list) -> None:
    rf = trained_models.get("Random Forest")
    if rf is None:
        return

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 7))
    sns.barplot(x=importances[indices],
                y=[feature_names[i] for i in indices],
                palette="viridis")
    plt.title("Top 20 Feature Importances — Random Forest",
              fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_models(trained_models: dict, output_dir: str = "models") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, model in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join(output_dir, filename)
        joblib.dump(model, path)
        print(f"[INFO] Saved model: {path}")


def load_model(filepath: str):
    return joblib.load(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main(data_path: str = "data/creditcard.csv") -> None:
    print("\n" + "=" * 60)
    print("  CREDIT CARD FRAUD DETECTION — ML PIPELINE")
    print("=" * 60 + "\n")

    # 1. Load & explore
    df = load_data(data_path)
    explore_data(df)
    plot_class_distribution(df)

    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # 3. SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 4. Train
    models = get_models()
    trained_models = train_models(models, X_train_res, y_train_res)

    # 5. Evaluate
    results_df = evaluate_all_models(trained_models, X_test, y_test)
    cross_validate_models(trained_models, X_train_res, y_train_res)

    # 6. Visualise
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test)
    plot_metrics_comparison(results_df)
    plot_feature_importance(trained_models, list(X_train.columns))

    # 7. Save
    save_models(trained_models)

    print("\n[DONE] Pipeline complete. Check reports/figures/ for all plots.")
    print(results_df.to_string())


if __name__ == "__main__":
    main()
