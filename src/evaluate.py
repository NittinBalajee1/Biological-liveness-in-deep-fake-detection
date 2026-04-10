from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import load_config
from .dataset import LABEL_TO_CLASS, get_paths_and_labels, load_split, validate_split
from .features import build_feature_matrix
from .utils import ensure_dir, save_json


def save_confusion_matrix_plot(cm: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["real", "fake"])
    plt.yticks([0, 1], ["real", "fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_feature_importance_plot(
    model_bundle: dict, output_path: str | Path, top_k: int = 15
) -> None:
    pipeline = model_bundle["pipeline"]
    feature_names = model_bundle["feature_names"]
    estimator = pipeline.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        return

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]
    selected_names = [feature_names[index] for index in indices]
    selected_values = importances[indices]

    plt.figure(figsize=(10, 6))
    positions = np.arange(len(selected_names))
    plt.barh(positions, selected_values)
    plt.yticks(positions, selected_names)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained deepfake detector.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_bundle = joblib.load(config["output"]["model_path"])

    test_samples = load_split(config["dataset"]["test_dir"])
    validate_split("test", test_samples)
    test_paths, y_test = get_paths_and_labels(test_samples)

    feature_config = model_bundle["feature_config"]
    X_test, _ = build_feature_matrix(test_paths, feature_config)

    pipeline = model_bundle["pipeline"]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=[LABEL_TO_CLASS[0], LABEL_TO_CLASS[1]],
            zero_division=0,
            output_dict=True,
        ),
    }

    cm = confusion_matrix(y_test, y_pred)

    save_json(metrics, config["output"]["metrics_path"])
    save_confusion_matrix_plot(cm, config["output"]["confusion_matrix_path"])
    save_feature_importance_plot(model_bundle, config["output"]["feature_importance_path"])

    print("Evaluation complete.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Metrics saved to: {config['output']['metrics_path']}")


if __name__ == "__main__":
    main()
