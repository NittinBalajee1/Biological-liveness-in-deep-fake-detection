from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .frequency_maps import FrequencyMapConfig
from .models_cnn import DualBranchDeepfakeDetector
from .torch_data import AugmentationConfig, MultiModalImageDataset
from .utils import ensure_dir, save_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch_inputs, device):
    return {name: tensor.to(device) for name, tensor in batch_inputs.items()}


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    frequency_attention_target: float = 0.0,
    frequency_attention_penalty: float = 0.0,
):
    model.train()
    losses = []
    all_targets = []
    all_probs = []
    mean_frequency_attention = []

    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs = move_batch_to_device(inputs, device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(inputs, return_attention=True)
        if isinstance(output, tuple):
            logits, attention = output
        else:
            logits = output
            attention = None
        loss = criterion(logits, targets)
        if (
            attention is not None
            and frequency_attention_penalty > 0.0
            and attention.shape[1] >= 2
        ):
            frequency_attention = attention[:, 1]
            attention_shortfall = torch.relu(frequency_attention_target - frequency_attention)
            loss = loss + frequency_attention_penalty * attention_shortfall.mean()
            mean_frequency_attention.append(float(frequency_attention.mean().detach().cpu().item()))
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        losses.append(float(loss.item()))
        all_probs.extend(probs.tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    mean_attention_value = (
        float(np.mean(mean_frequency_attention)) if mean_frequency_attention else None
    )
    return float(np.mean(losses)), np.array(all_targets), np.array(all_probs), mean_attention_value


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_targets = []
    all_probs = []
    mean_frequency_attention = []

    for inputs, targets in tqdm(loader, desc="val", leave=False):
        inputs = move_batch_to_device(inputs, device)
        targets = targets.to(device)

        output = model(inputs, return_attention=True)
        if isinstance(output, tuple):
            logits, attention = output
        else:
            logits = output
            attention = None
        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits).cpu().numpy()
        if attention is not None and attention.shape[1] >= 2:
            mean_frequency_attention.append(float(attention[:, 1].mean().cpu().item()))

        losses.append(float(loss.item()))
        all_probs.extend(probs.tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    mean_attention_value = (
        float(np.mean(mean_frequency_attention)) if mean_frequency_attention else None
    )
    return float(np.mean(losses)), np.array(all_targets), np.array(all_probs), mean_attention_value


def metric_bundle(targets: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "f1_score": float(f1_score(targets, preds, zero_division=0)),
    }


def find_best_threshold(targets: np.ndarray, probs: np.ndarray, metric_name: str) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = metric_bundle(targets, probs, best_threshold)
    best_value = best_metrics.get(metric_name, best_metrics["accuracy"])

    for threshold in np.linspace(0.2, 0.8, 25):
        metrics = metric_bundle(targets, probs, float(threshold))
        value = metrics.get(metric_name, metrics["accuracy"])
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frequency-domain CNN.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    map_config = FrequencyMapConfig(
        image_size=config["dataset"]["image_size"],
        normalize_mode=config["preprocessing"].get("normalize_mode", "minmax"),
    )
    augmentation = AugmentationConfig(
        horizontal_flip=config["augmentation"].get("horizontal_flip", 0.5),
        gaussian_blur=config["augmentation"].get("gaussian_blur", 0.15),
        jpeg_noise=config["augmentation"].get("jpeg_noise", 0.2),
        brightness_jitter=config["augmentation"].get("brightness_jitter", 0.15),
        contrast_jitter=config["augmentation"].get("contrast_jitter", 0.15),
    )

    train_dataset = MultiModalImageDataset(
        config["dataset"]["train_dir"], map_config, augmentation=augmentation, training=True
    )
    val_dataset = MultiModalImageDataset(config["dataset"]["val_dir"], map_config, training=False)

    loader_kwargs = {
        "batch_size": config["training"].get("batch_size", 16),
        "num_workers": config["dataset"].get("num_workers", 0),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = DualBranchDeepfakeDetector(
        spatial_backbone=config["model"].get("spatial_backbone", "resnet18"),
        pretrained=config["model"].get("pretrained", False),
        freeze_backbone=config["model"].get("freeze_backbone", False),
        frequency_channels=config["model"].get("frequency_channels", [24, 48, 96]),
        fusion_hidden_dim=config["model"].get("fusion_hidden_dim", 128),
        use_attention_fusion=config["model"].get("use_attention_fusion", True),
        attention_hidden_dim=config["model"].get("attention_hidden_dim", 64),
        dropout=config["model"].get("dropout", 0.35),
        use_spatial=config["model"].get("use_spatial", True),
        use_fft=config["model"].get("use_fft", True),
        use_dct=config["model"].get("use_dct", True),
        spatial_logit_bias=config["model"].get("spatial_logit_bias", 0.0),
        frequency_logit_bias=config["model"].get("frequency_logit_bias", 0.0),
        attention_temperature=config["model"].get("attention_temperature", 1.0),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"].get("learning_rate", 3e-4),
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = Path(config["output"]["checkpoint_path"])
    history_path = Path(config["output"]["history_path"])
    ensure_dir(checkpoint_path.parent)
    ensure_dir(history_path.parent)

    best_score = -1.0
    best_epoch = -1
    patience = config["training"].get("early_stopping_patience", 5)
    threshold = config["training"].get("decision_threshold", 0.5)
    total_epochs = config["training"].get("epochs", 20)
    threshold_metric = config["training"].get("threshold_metric", "accuracy")
    frequency_attention_target = config["training"].get("frequency_attention_target", 0.0)
    frequency_attention_penalty = config["training"].get("frequency_attention_penalty", 0.0)

    history = []
    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")
        train_loss, train_targets, train_probs, train_frequency_attention = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            frequency_attention_target=frequency_attention_target,
            frequency_attention_penalty=frequency_attention_penalty,
        )
        val_loss, val_targets, val_probs, val_frequency_attention = evaluate(
            model, val_loader, criterion, device
        )

        train_metrics = metric_bundle(train_targets, train_probs, threshold)
        tuned_threshold, val_metrics = find_best_threshold(val_targets, val_probs, threshold_metric)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "threshold": tuned_threshold,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "train_frequency_attention": train_frequency_attention,
                "val_frequency_attention": val_frequency_attention,
            }
        )

        print(
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_score']:.4f} "
            f"threshold={tuned_threshold:.2f} "
            f"freq_attn_train={train_frequency_attention if train_frequency_attention is not None else 'n/a'} "
            f"freq_attn_val={val_frequency_attention if val_frequency_attention is not None else 'n/a'}"
        )

        current_score = val_metrics.get(threshold_metric, val_metrics["accuracy"])
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            threshold = tuned_threshold
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "map_config": map_config.__dict__,
                    "spatial_backbone": config["model"].get("spatial_backbone", "resnet18"),
                    "pretrained": config["model"].get("pretrained", False),
                    "freeze_backbone": config["model"].get("freeze_backbone", False),
                    "frequency_channels": config["model"].get("frequency_channels", [24, 48, 96]),
                    "fusion_hidden_dim": config["model"].get("fusion_hidden_dim", 128),
                    "use_attention_fusion": config["model"].get("use_attention_fusion", True),
                    "attention_hidden_dim": config["model"].get("attention_hidden_dim", 64),
                    "dropout": config["model"].get("dropout", 0.35),
                    "use_spatial": config["model"].get("use_spatial", True),
                    "use_fft": config["model"].get("use_fft", True),
                    "use_dct": config["model"].get("use_dct", True),
                    "spatial_logit_bias": config["model"].get("spatial_logit_bias", 0.0),
                    "frequency_logit_bias": config["model"].get("frequency_logit_bias", 0.0),
                    "attention_temperature": config["model"].get("attention_temperature", 1.0),
                    "threshold": threshold,
                },
                checkpoint_path,
            )
            print(f"Saved best model to: {checkpoint_path}")

        if epoch - best_epoch >= patience:
            print("Early stopping triggered.")
            break

    save_json({"history": history, "best_validation_score": best_score}, history_path)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
