"""
tests/test_fraud_detection.py
=============================
Unit tests for the fraud detection pipeline.

Run with:
    pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraud_detection import (
    explore_data,
    preprocess_data,
    apply_smote,
    get_models,
    train_models,
    evaluate_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a minimal synthetic credit card dataframe."""
    np.random.seed(0)
    n = 1000
    df = pd.DataFrame(
        np.random.randn(n, 28),
        columns=[f"V{i}" for i in range(1, 29)]
    )
    df["Time"]   = np.random.uniform(0, 172800, n)
    df["Amount"] = np.abs(np.random.exponential(85, n))
    df["Class"]  = np.where(np.random.rand(n) < 0.02, 1, 0)
    return df


# ── EDA ───────────────────────────────────────────────────────────────────────

def test_explore_data_runs(sample_df, capsys):
    explore_data(sample_df)
    captured = capsys.readouterr()
    assert "Class distribution" in captured.out


# ── Preprocessing ─────────────────────────────────────────────────────────────

def test_preprocess_creates_splits(sample_df):
    X_train, X_test, y_train, y_test, scaler = preprocess_data(sample_df.copy())
    assert len(X_train) + len(X_test) == len(sample_df)
    assert "Time"   not in X_train.columns
    assert "Amount" not in X_train.columns
    assert "scaled_amount" in X_train.columns
    assert "scaled_time"   in X_train.columns


def test_preprocess_stratification(sample_df):
    X_train, X_test, y_train, y_test, _ = preprocess_data(sample_df.copy())
    train_fraud_rate = y_train.mean()
    test_fraud_rate  = y_test.mean()
    # Should be within 2 percentage points
    assert abs(train_fraud_rate - test_fraud_rate) < 0.02


# ── SMOTE ─────────────────────────────────────────────────────────────────────

def test_smote_balances_classes(sample_df):
    X_train, _, y_train, _, _ = preprocess_data(sample_df.copy())
    X_res, y_res = apply_smote(X_train, y_train)
    counts = pd.Series(y_res).value_counts()
    assert counts[0] == counts[1], "SMOTE should produce perfectly balanced classes"


def test_smote_increases_minority(sample_df):
    X_train, _, y_train, _, _ = preprocess_data(sample_df.copy())
    minority_before = (y_train == 1).sum()
    _, y_res = apply_smote(X_train, y_train)
    minority_after = (y_res == 1).sum()
    assert minority_after > minority_before


# ── Models ────────────────────────────────────────────────────────────────────

def test_get_models_returns_four():
    models = get_models()
    assert len(models) == 4
    assert "Logistic Regression" in models
    assert "Random Forest"       in models
    assert "KNN"                 in models
    assert "XGBoost"             in models


def test_models_train_and_predict():
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        weights=[0.95, 0.05], random_state=42
    )
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    models = get_models()
    trained = train_models(models, X_train, y_train)

    for name, model in trained.items():
        preds = model.predict(X_test)
        assert len(preds) == len(X_test), f"{name} prediction length mismatch"
        assert set(preds).issubset({0, 1}),  f"{name} produced unexpected classes"


# ── Evaluation ────────────────────────────────────────────────────────────────

def test_evaluate_model_returns_all_metrics():
    X, y = make_classification(
        n_samples=300, n_features=10, weights=[0.9, 0.1], random_state=0
    )
    model = LogisticRegression(max_iter=500).fit(X[:240], y[:240])
    metrics = evaluate_model("LR", model, X[240:], y[240:])

    for key in ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        assert key in metrics, f"Missing metric: {key}"

    assert 0 <= metrics["Accuracy"]  <= 1
    assert 0 <= metrics["Precision"] <= 1
    assert 0 <= metrics["Recall"]    <= 1
    assert 0 <= metrics["F1-Score"]  <= 1
    assert 0 <= metrics["ROC-AUC"]   <= 1
