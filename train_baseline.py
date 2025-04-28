"""
Baseline numeric‑only models:
  • LogisticRegression
  • RandomForestClassifier

Expects data split Parquets in config.SPLIT_DIR
"""

import os, joblib, json, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                             precision_recall_curve)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import config
warnings.filterwarnings("ignore")

NUMERIC_COLS = ["n_posts", "mean_score"]   # expand later

# ---------- helper ----------
def load_split(name):
    df = pd.read_parquet(f"{config.SPLIT_DIR}/{name}.parquet")
    X = df[NUMERIC_COLS].values
    y = df["weak_label"].values
    return X, y

# ---------- load ----------
X_tr, y_tr = load_split("train")
X_va, y_va = load_split("valid")
X_te, y_te = load_split("test")

scaler = StandardScaler().fit(X_tr)
X_tr, X_va, X_te = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

# compute class weights from training labels
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
cw_dict = dict(zip([0, 1], cw))

# ---------- Logistic ----------
logreg = LogisticRegression(max_iter=200, class_weight=cw_dict)
logreg.fit(X_tr, y_tr)
va_pred = logreg.predict_proba(X_va)[:, 1]

print("\nLogisticRegression validation — AP:",
      f"{average_precision_score(y_va, va_pred):.3f}")
print(classification_report(y_va, va_pred > 0.5, digits=3))

# ---------- RandomForest ----------
rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=config.RANDOM_SEED)
rf.fit(X_tr, y_tr)
va_pred = rf.predict_proba(X_va)[:, 1]

print("\nRandomForest validation — AP:",
      f"{average_precision_score(y_va, va_pred):.3f}")
print(classification_report(y_va, va_pred > 0.5, digits=3))

# ---------- final test on best model (pick RF here) ----------
best = rf
te_pred = best.predict_proba(X_te)[:, 1]
print("\n[RF] test AP:", f"{average_precision_score(y_te, te_pred):.3f}")
print(classification_report(y_te, te_pred > 0.5, digits=3))

os.makedirs(config.MODEL_DIR, exist_ok=True)
joblib.dump({"scaler": scaler, "model": best},
            f"{config.MODEL_DIR}/rf_numeric.joblib", compress=3)
print("✓ saved →", f"{config.MODEL_DIR}/rf_numeric.joblib")
