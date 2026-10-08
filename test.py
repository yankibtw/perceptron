from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

MODEL_FILE = Path("model/mlp_model.csv")
INPUT_FILE = Path("data/X_test.csv")
TARGET_FILE = Path("data/y_test.csv")
OUTPUT_FILE = Path("model/predictions.csv")
CONFUSION_MATRIX_FILE = Path("img/confusion_matrix.png")


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def main():
    for file_path in (MODEL_FILE, INPUT_FILE, TARGET_FILE):
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}\nСначала запустите data.py и train.py.")

    model_df = pd.read_csv(MODEL_FILE)

    def vector(name):
        rows = model_df[model_df["parameter"] == name].sort_values("row")
        if rows.empty:
            raise ValueError(f"В модели отсутствует параметр: {name}")
        return rows["value"].astype(float).to_numpy()

    def matrix(name, shape):
        rows = model_df[model_df["parameter"] == name]
        if rows.empty:
            raise ValueError(f"В модели отсутствует параметр: {name}")

        result = np.zeros(shape, dtype=np.float64)

        for _, row in rows.iterrows():
            result[int(row["row"]), int(row["column"])] = float(row["value"])

        return result

    layer_sizes = vector("layer_size").astype(int).tolist()

    weights = [matrix(f"weight_{i}", (layer_sizes[i + 1], layer_sizes[i])) for i in range(len(layer_sizes) - 1)]
    biases = [vector(f"bias_{i}") for i in range(len(layer_sizes) - 1)]

    labels = (model_df[model_df["parameter"] == "class_label"].sort_values("row")["value"].astype(str).to_numpy())

    medians = vector("median")
    mean = vector("mean")
    std = vector("std")

    X_df = pd.read_csv(INPUT_FILE)
    X = X_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    if X.shape[1] != len(mean):
        raise ValueError(f"Число признаков не совпадает: получено {X.shape[1]}, ожидалось {len(mean)}.")

    if np.isnan(X).any():
        raise ValueError("В X_test.csv обнаружены пропущенные значения.")

    def predict_one(x):
        a = x

        for w, b in zip(weights[:-1], biases[:-1]):
            a = sigmoid(w @ a + b)

        return softmax(weights[-1] @ a + biases[-1])

    probabilities = np.vstack([predict_one(x) for x in X])
    y_pred = np.argmax(probabilities, axis=1)

    results = pd.DataFrame({
        "probability_B": probabilities[:, 0],
        "probability_M": probabilities[:, 1],
        "predicted_class": labels[y_pred],
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    y_df = pd.read_csv(TARGET_FILE)
    y_raw = y_df.iloc[:, 0].to_numpy()

    if len(y_raw) != len(y_pred):
        raise ValueError("Количество тестовых меток не совпадает с количеством предсказаний.")

    if set(map(str, y_raw)).issubset({"B", "M"}):
        y_true = (y_raw.astype(str) == "M").astype(int)
    else:
        y_true = y_raw.astype(int)

    print("\nОтчёт о классификации:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["B", "M"],
            zero_division=0,
        )
    )

    p_m = np.clip(probabilities[:, 1], 1e-12, 1.0 - 1e-12)
    bce = -np.mean(y_true * np.log(p_m) + (1 - y_true) * np.log(1 - p_m))

    print(f"Binary cross-entropy: {bce:.6f}")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["B", "M"])
    plt.yticks([0, 1], ["B", "M"])

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=color,
            )

    plt.colorbar()
    plt.tight_layout()

    CONFUSION_MATRIX_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=150)
    plt.close()

    print("Матрица ошибок сохранена:", CONFUSION_MATRIX_FILE)

if __name__ == "__main__":
    main()