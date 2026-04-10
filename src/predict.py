from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .features import extract_frequency_features
from .utils import is_image_file


LABEL_TO_CLASS = {0: "real", 1: "fake"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether an image is real or fake.")
    parser.add_argument("--model", required=True, help="Path to trained model bundle.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_path.exists() or not is_image_file(image_path):
        raise FileNotFoundError(f"Image file not found or unsupported: {image_path}")

    model_bundle = joblib.load(model_path)
    pipeline = model_bundle["pipeline"]
    feature_config = model_bundle["feature_config"]

    features, _ = extract_frequency_features(image_path, feature_config)
    prediction = int(pipeline.predict([features])[0])

    probability = None
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "predict_proba"):
        probability = float(pipeline.predict_proba([features])[0][1])

    print(f"Image: {image_path}")
    print(f"Prediction: {LABEL_TO_CLASS[prediction]}")
    if probability is not None:
        print(f"Fake Probability: {probability:.4f}")


if __name__ == "__main__":
    main()
