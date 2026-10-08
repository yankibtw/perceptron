from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

df = pd.read_csv("data/data.csv", header=None)

labels = df.iloc[:, 1].astype(str).to_numpy()
y = (labels == "M").astype(np.int64)

features = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
X = features.to_numpy(dtype=np.float64)

print("Первые 5 строк:")
print(df.head())
print(f"Количество пропущенных значений: {np.isnan(X).sum()}")

idx = np.arange(len(X))

train_idx, temp_idx = train_test_split(
    idx,
    test_size=VAL_SIZE + TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
    random_state=RANDOM_STATE,
    stratify=y[temp_idx],
)

X_train, X_val, X_test = X[train_idx].copy(), X[val_idx].copy(), X[test_idx].copy()
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

medians = np.nanmedian(X_train, axis=0)

nan_mask_train = np.isnan(X_train)
nan_mask_val = np.isnan(X_val)
nan_mask_test = np.isnan(X_test)

X_train[nan_mask_train] = np.take(medians, np.where(nan_mask_train)[1])
X_val[nan_mask_val] = np.take(medians, np.where(nan_mask_val)[1])
X_test[nan_mask_test] = np.take(medians, np.where(nan_mask_test)[1])

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

datasets = (
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test),
)

for name, X_part, y_part in datasets:
    pd.DataFrame(X_part).to_csv(f"data/X_{name}.csv", index=False)
    pd.DataFrame(y_part, columns=["diagnosis"]).to_csv(f"data/y_{name}.csv", index=False)

pd.DataFrame({"median": medians, "mean": mean, "std": std}).to_csv("data/params.csv", index=False)

print("\nРазбиение:")
print(f"Train: {len(X_train)} ({len(X_train) / len(X) * 100:.2f}%)")
print(f"Val:   {len(X_val)} ({len(X_val) / len(X) * 100:.2f}%)")
print(f"Test:  {len(X_test)} ({len(X_test) / len(X) * 100:.2f}%)")

print("\nПроцент классов в распределении:")
for name, target_labels in [("All", y), ("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    benign_percent = np.mean(target_labels == 0) * 100
    malignant_percent = np.mean(target_labels == 1) * 100
    print(f"{name:5s}: Benign = {benign_percent:.2f}% | Malignant = {malignant_percent:.2f}%")

plt.figure(figsize=(7, 5))
plt.bar(["B", "M"], [(y == 0).sum(), (y == 1).sum()])
plt.title("Распределение классов")
plt.ylabel("Количество объектов")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()