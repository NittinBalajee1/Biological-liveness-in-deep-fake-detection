from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .dataset import LABEL_TO_CLASS, get_paths_and_labels, load_split, validate_split
from .features import FrequencyFeatureConfig, build_feature_matrix
from .utils import ensure_dir


def build_model(model_config: dict) -> Pipeline:
    model_type = model_config.get("type", "random_forest").lower()

    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=model_config.get("n_estimators", 300),
            max_depth=model_config.get("max_depth", 14),
            min_samples_split=model_config.get("min_samples_split", 4),
            min_samples_leaf=model_config.get("min_samples_leaf", 2),
            random_state=model_config.get("random_state", 42),
            n_jobs=-1,
        )
        return Pipeline([("model", classifier)])

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            random_state=model_config.get("random_state", 42),
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", classifier),
            ]
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frequency-domain deepfake detector.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)

    train_samples = load_split(config["dataset"]["train_dir"])
    val_samples = load_split(config["dataset"]["val_dir"])
    validate_split("train", train_samples)
    validate_split("val", val_samples)

    feature_config = FrequencyFeatureConfig(
        image_size=config["dataset"].get("image_size", 256),
        radial_bins=config["features"].get("radial_bins", 32),
        log_magnitude=config["features"].get("log_magnitude", True),
        normalize_spectrum=config["features"].get("normalize_spectrum", True),
    )

    train_paths, y_train = get_paths_and_labels(train_samples)
    val_paths, y_val = get_paths_and_labels(val_samples)

    print("Extracting training features...")
    X_train, feature_names = build_feature_matrix(train_paths, feature_config)
    print("Extracting validation features...")
    X_val, _ = build_feature_matrix(val_paths, feature_config)

    model = build_model(config["model"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=[LABEL_TO_CLASS[0], LABEL_TO_CLASS[1]],
            zero_division=0,
        )
    )

    output_model_path = Path(config["output"]["model_path"])
    ensure_dir(output_model_path.parent)
    joblib.dump(
        {
            "pipeline": model,
            "feature_names": feature_names,
            "feature_config": feature_config,
            "config": config,
        },
        output_model_path,
    )
    print(f"Saved model to: {output_model_path}")


if __name__ == "__main__":
    main()
