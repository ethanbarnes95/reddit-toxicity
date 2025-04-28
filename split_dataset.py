"""
Split data/user_level.parquet → train / valid / test CSVs
Keeps class ratio with sklearn StratifiedShuffleSplit.
"""
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import config

SRC       = "data/processed/user_level.parquet"
OUT_DIR   = "data/processed/splits"
RANDOM_SEED = getattr(config, "RANDOM_SEED", 42)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_parquet(SRC)
X = df.drop(columns=["weak_label"])
y = df["weak_label"].values

splitter = StratifiedShuffleSplit(
    n_splits=1, test_size=0.30, random_state=RANDOM_SEED
)
train_idx, temp_idx = next(splitter.split(X, y))

# secondary split → valid / test
valid_split = StratifiedShuffleSplit(
    n_splits=1, test_size=0.50, random_state=RANDOM_SEED
)
valid_idx, test_idx = next(valid_split.split(
    X.iloc[temp_idx], y[temp_idx]
))

df.iloc[train_idx].to_parquet(f"{OUT_DIR}/train.parquet", index=False)
df.iloc[temp_idx[valid_idx]].to_parquet(f"{OUT_DIR}/valid.parquet", index=False)
df.iloc[temp_idx[test_idx]].to_parquet (f"{OUT_DIR}/test.parquet",  index=False)

print("✓  splits saved   train:", len(train_idx),
      " valid:", len(valid_idx), " test:", len(test_idx))
