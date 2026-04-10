from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from .config import load_config
from .frequency_maps import FrequencyMapConfig
from .models_cnn import DualBranchDeepfakeDetector
from .torch_data import MultiModalImageDataset
from .train_cnn import evaluate
from .utils import ensure_dir, save_json


def save_confusion_matrix(cm: np.ndarray, output_path: str | Path) -> None:
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
    plt.title("Frequency CNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frequency CNN on the test set.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = torch.load(config["output"]["checkpoint_path"], map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    map_config = FrequencyMapConfig(**checkpoint["map_config"])
    model = DualBranchDeepfakeDetector(
        spatial_backbone=checkpoint["spatial_backbone"],
        pretrained=False,
        freeze_backbone=checkpoint.get("freeze_backbone", False),
        frequency_channels=checkpoint["frequency_channels"],
        fusion_hidden_dim=checkpoint["fusion_hidden_dim"],
        use_attention_fusion=checkpoint.get("use_attention_fusion", True),
        attention_hidden_dim=checkpoint.get("attention_hidden_dim", 64),
        dropout=checkpoint["dropout"],
        use_spatial=checkpoint.get("use_spatial", True),
        use_fft=checkpoint.get("use_fft", True),
        use_dct=checkpoint.get("use_dct", True),
        spatial_logit_bias=checkpoint.get("spatial_logit_bias", 0.0),
        frequency_logit_bias=checkpoint.get("frequency_logit_bias", 0.0),
        attention_temperature=checkpoint.get("attention_temperature", 1.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = MultiModalImageDataset(config["dataset"]["test_dir"], map_config, training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"].get("batch_size", 16),
        shuffle=False,
        num_workers=config["dataset"].get("num_workers", 0),
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss, targets, probs, mean_frequency_attention = evaluate(model, test_loader, criterion, device)
    threshold = float(checkpoint.get("threshold", 0.5))
    preds = (probs >= threshold).astype(np.int32)

    metrics = {
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "f1_score": float(f1_score(targets, preds, zero_division=0)),
        "mean_frequency_attention": mean_frequency_attention,
        "classification_report": classification_report(
            targets,
            preds,
            target_names=["real", "fake"],
            zero_division=0,
            output_dict=True,
        ),
    }

    cm = confusion_matrix(targets, preds)
    save_json(metrics, config["output"]["metrics_path"])
    save_confusion_matrix(cm, config["output"]["confusion_matrix_path"])

    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
