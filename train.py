import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[32, 16, 8, 4, 2])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_df(a):
    return a * (1.0 - a)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def save_model(path, layer_sizes, weights, biases, medians, mean, std):
    rows = []

    def add_vector(name, vec):
        for i, value in enumerate(vec):
            rows.append({"parameter": name, "row": i, "column": -1, "value": float(value)})

    def add_matrix(name, matrix):
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                rows.append(
                    {
                        "parameter": name,
                        "row": r,
                        "column": c,
                        "value": float(matrix[r, c]),
                    }
                )

    for i, size in enumerate(layer_sizes):
        rows.append(
            {"parameter": "layer_size", "row": i, "column": -1, "value": int(size)}
        )

    rows.extend(
        [
            {"parameter": "class_label", "row": 0, "column": -1, "value": "B"},
            {"parameter": "class_label", "row": 1, "column": -1, "value": "M"},
        ]
    )

    for i, (w, b) in enumerate(zip(weights, biases)):
        add_matrix(f"weight_{i}", w)
        add_vector(f"bias_{i}", b)

    add_vector("median", medians)
    add_vector("mean", mean)
    add_vector("std", std)

    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs должен быть не меньше 1")
    if args.learning_rate <= 0:
        raise ValueError("--lr должен быть больше 0")
    if not args.hidden_layers or any(n < 1 for n in args.hidden_layers):
        raise ValueError("Размер каждого скрытого слоя должен быть положительным")

    x_train_path, y_train_path = ("data/X_train.csv", "data/y_train.csv")
    x_val_path, y_val_path = "data/X_val.csv", "data/y_val.csv"

    X_train = pd.read_csv(x_train_path).to_numpy()
    y_train = pd.read_csv(y_train_path).iloc[:, 0].to_numpy()
    X_val = pd.read_csv(x_val_path).to_numpy()
    y_val = pd.read_csv(y_val_path).iloc[:, 0].to_numpy()

    prep = pd.read_csv("data/params.csv")
    medians, feature_mean, feature_std = (prep[c].to_numpy(dtype=np.float64) for c in ("median", "mean", "std"))

    if X_train.shape[1] != len(feature_mean):
        raise ValueError("Число признаков не совпадает с параметрами предобработки. Сначала запустите data.py")

    rng = np.random.default_rng(args.seed)
    layer_sizes = [X_train.shape[1], *args.hidden_layers, 2]

    weights = [
        rng.normal(0.0, np.sqrt(2.0 / n_in), (n_out, n_in))
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
    ]
    biases = [np.zeros(n_out, dtype=np.float64) for n_out in layer_sizes[1:]]
    velocity_w = [np.zeros_like(w) for w in weights]
    velocity_b = [np.zeros_like(b) for b in biases]

    def forward(x):
        activations = [x]
        for w, b in zip(weights[:-1], biases[:-1]):
            activations.append(sigmoid(w @ activations[-1] + b))
        activations.append(softmax(weights[-1] @ activations[-1] + biases[-1]))
        return activations[-1], activations

    def evaluate(X, y):
        losses, correct = [], 0
        for x, target in zip(X, y):
            probs, _ = forward(x)
            losses.append(-np.log(np.clip(probs[int(target)], 1e-12, 1.0)))
            correct += int(np.argmax(probs) == int(target))
        return float(np.mean(losses)), correct / len(y)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    y_onehot = np.eye(2, dtype=np.float64)[y_train]

    for epoch in range(args.epochs):
        for idx in rng.permutation(len(X_train)):
            x, target = X_train[idx], y_onehot[idx]
            probs, activations = forward(x)

            deltas = [None] * len(weights)
            deltas[-1] = probs - target 

            for layer in range(len(weights) - 2, -1, -1):
                deltas[layer] = (
                    weights[layer + 1].T @ deltas[layer + 1]
                ) * sigmoid_df(activations[layer + 1])

            for i in range(len(weights)):
                grad_w = np.outer(deltas[i], activations[i])
                velocity_w[i] = (
                    args.momentum * velocity_w[i] - args.learning_rate * grad_w
                )
                velocity_b[i] = (
                    args.momentum * velocity_b[i] - args.learning_rate * deltas[i]
                )
                weights[i] += velocity_w[i]
                biases[i] += velocity_b[i]

        train_loss, train_acc = evaluate(X_train, y_train)
        val_loss, val_acc = evaluate(X_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1:03d}/{args.epochs} | Train loss: {train_loss:.5f} | Train acc: {train_acc:.4f} | Validation loss: {val_loss:.5f} | Validation acc: {val_acc:.4f}")

    model_path = "model/mlp_model.csv"
    save_model(model_path, layer_sizes, weights, biases, medians, feature_mean, feature_std)
    epochs = np.arange(1, args.epochs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Train / validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("img/loss_curve.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train / validation accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("img/accuracy_curve.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()