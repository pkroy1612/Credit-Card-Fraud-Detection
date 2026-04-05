"""
predict.py
==========
Load a saved model and run inference on new transaction data.

Usage:
    python src/predict.py --model models/random_forest.pkl --input data/new_transactions.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same scaling used during training."""
    scaler = StandardScaler()
    if "Amount" in df.columns:
        df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
        df.drop("Amount", axis=1, inplace=True)
    if "Time" in df.columns:
        df["scaled_time"] = scaler.fit_transform(df[["Time"]])
        df.drop("Time", axis=1, inplace=True)
    if "Class" in df.columns:
        df.drop("Class", axis=1, inplace=True)
    return df


def predict(model_path: str, data_path: str) -> pd.DataFrame:
    print(f"[INFO] Loading model  : {model_path}")
    model = joblib.load(model_path)

    print(f"[INFO] Loading data   : {data_path}")
    df = pd.read_csv(data_path)
    original_df = df.copy()

    df = preprocess_input(df)
    preds  = model.predict(df)
    probas = model.predict_proba(df)[:, 1]

    results = original_df.copy()
    results["Predicted_Class"]      = preds
    results["Fraud_Probability"]    = probas
    results["Risk_Level"] = pd.cut(
        probas,
        bins=[0, 0.3, 0.7, 1.0],
        labels=["Low", "Medium", "High"]
    )

    n_fraud = (preds == 1).sum()
    print(f"\n[RESULTS] Transactions analysed : {len(preds):,}")
    print(f"[RESULTS] Predicted fraudulent  : {n_fraud:,}  ({n_fraud/len(preds)*100:.2f}%)")
    print(f"[RESULTS] Predicted legitimate  : {len(preds)-n_fraud:,}")

    high_risk = results[results["Risk_Level"] == "High"]
    if not high_risk.empty:
        print(f"\n[ALERT] ⚠  {len(high_risk)} HIGH-RISK transactions detected!")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud — Inference Script"
    )
    parser.add_argument("--model", required=True,
                        help="Path to saved .pkl model file")
    parser.add_argument("--input", required=True,
                        help="Path to input CSV file")
    parser.add_argument("--output", default="predictions.csv",
                        help="Path to save predictions CSV (default: predictions.csv)")
    args = parser.parse_args()

    results = predict(args.model, args.input)
    results.to_csv(args.output, index=False)
    print(f"\n[INFO] Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
