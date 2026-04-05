"""
generate_sample_data.py
=======================
Generates a synthetic credit card fraud dataset that mirrors the structure
of the real Kaggle 'creditcard.csv' dataset (284,807 transactions, ~0.17% fraud).

Usage:
    python src/generate_sample_data.py

Output:
    data/creditcard.csv
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_LEGIT  = 284_315
N_FRAUD  = 492
N_TOTAL  = N_LEGIT + N_FRAUD
N_PCA    = 28         # V1 … V28

print(f"[INFO] Generating {N_TOTAL:,} synthetic transactions ...")

# ── PCA features (V1–V28) ────────────────────────────────────────────────────
# Legitimate: centred near 0, small variance
legit_pca  = np.random.randn(N_LEGIT, N_PCA) * 1.5

# Fraudulent: shifted means to separate the class
fraud_means = np.linspace(-3, 3, N_PCA)
fraud_pca   = np.random.randn(N_FRAUD, N_PCA) * 0.8 + fraud_means

pca_cols = pd.DataFrame(
    np.vstack([legit_pca, fraud_pca]),
    columns=[f"V{i}" for i in range(1, N_PCA + 1)]
)

# ── Time & Amount ─────────────────────────────────────────────────────────────
time_legit = np.random.uniform(0, 172_800, N_LEGIT)
time_fraud = np.random.uniform(0, 172_800, N_FRAUD)

amount_legit = np.abs(np.random.exponential(scale=85, size=N_LEGIT))
amount_fraud = np.abs(np.random.exponential(scale=120, size=N_FRAUD))

# ── Combine ───────────────────────────────────────────────────────────────────
df = pd.concat([pca_cols], axis=1)
df.insert(0, "Time",   np.concatenate([time_legit,  time_fraud]))
df["Amount"] = np.concatenate([amount_legit, amount_fraud])
df["Class"]  = np.concatenate([np.zeros(N_LEGIT, dtype=int),
                                np.ones(N_FRAUD,  dtype=int)])

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
out_path = "data/creditcard.csv"
df.to_csv(out_path, index=False)

print(f"[INFO] Saved to {out_path}")
print(f"[INFO] Shape : {df.shape}")
print(f"[INFO] Fraud : {df['Class'].sum():,}  ({df['Class'].mean()*100:.2f}%)")
print("[DONE]")
