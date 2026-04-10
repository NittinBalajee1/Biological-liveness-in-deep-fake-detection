from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class FrequencyFeatureConfig:
    image_size: int = 256
    radial_bins: int = 32
    log_magnitude: bool = True
    normalize_spectrum: bool = True


def load_image_grayscale(image_path: str | Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image


def compute_magnitude_spectrum(image: np.ndarray, log_magnitude: bool = True) -> np.ndarray:
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    if log_magnitude:
        magnitude = np.log1p(magnitude)

    return magnitude


def radial_profile(magnitude: np.ndarray, bins: int) -> np.ndarray:
    rows, cols = magnitude.shape
    center_y, center_x = rows // 2, cols // 2
    y_indices, x_indices = np.indices((rows, cols))
    radius = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    max_radius = radius.max()
    bin_edges = np.linspace(0, max_radius, bins + 1)
    profile = np.zeros(bins, dtype=np.float32)

    for i in range(bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if np.any(mask):
            profile[i] = float(magnitude[mask].mean())

    return profile


def band_energy_ratios(magnitude: np.ndarray) -> np.ndarray:
    rows, cols = magnitude.shape
    center_y, center_x = rows // 2, cols // 2
    y_indices, x_indices = np.indices((rows, cols))
    radius = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    max_radius = radius.max()

    low_mask = radius < max_radius * 0.15
    mid_mask = (radius >= max_radius * 0.15) & (radius < max_radius * 0.5)
    high_mask = radius >= max_radius * 0.5

    total_energy = magnitude.sum() + 1e-8
    low_ratio = magnitude[low_mask].sum() / total_energy
    mid_ratio = magnitude[mid_mask].sum() / total_energy
    high_ratio = magnitude[high_mask].sum() / total_energy

    return np.array([low_ratio, mid_ratio, high_ratio], dtype=np.float32)


def spectral_statistics(magnitude: np.ndarray) -> np.ndarray:
    flat = magnitude.flatten().astype(np.float64)
    normalized = flat / (flat.sum() + 1e-12)

    entropy = -np.sum(normalized * np.log2(normalized + 1e-12))
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    maximum = float(np.max(flat))
    concentration = float(np.percentile(flat, 95) / (mean + 1e-8))
    skew_proxy = float(((flat - mean) ** 3).mean() / (std ** 3 + 1e-8))

    return np.array(
        [mean, std, maximum, entropy, concentration, skew_proxy],
        dtype=np.float32,
    )


def extract_frequency_features(
    image_path: str | Path, config: FrequencyFeatureConfig
) -> Tuple[np.ndarray, list[str]]:
    image = load_image_grayscale(image_path, config.image_size)
    magnitude = compute_magnitude_spectrum(image, config.log_magnitude)

    if config.normalize_spectrum:
        magnitude = magnitude / (magnitude.sum() + 1e-8)

    radial = radial_profile(magnitude, config.radial_bins)
    energy = band_energy_ratios(magnitude)
    stats = spectral_statistics(magnitude)

    features = np.concatenate([radial, energy, stats], axis=0).astype(np.float32)
    feature_names = (
        [f"radial_bin_{i}" for i in range(config.radial_bins)]
        + ["low_freq_ratio", "mid_freq_ratio", "high_freq_ratio"]
        + [
            "spectral_mean",
            "spectral_std",
            "spectral_max",
            "spectral_entropy",
            "spectral_concentration",
            "spectral_skew_proxy",
        ]
    )
    return features, feature_names


def build_feature_matrix(
    image_paths: Iterable[str | Path], config: FrequencyFeatureConfig
) -> Tuple[np.ndarray, list[str]]:
    feature_list: List[np.ndarray] = []
    feature_names: list[str] | None = None

    for image_path in image_paths:
        features, names = extract_frequency_features(image_path, config)
        feature_list.append(features)
        if feature_names is None:
            feature_names = names

    if not feature_list:
        raise ValueError("No features could be extracted because the input image list is empty.")

    return np.vstack(feature_list), feature_names or []
