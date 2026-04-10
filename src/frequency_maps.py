from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.fftpack import dct

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass
class FrequencyMapConfig:
    image_size: int = 224
    normalize_mode: str = "minmax"


def preprocess_face(face_bgr: np.ndarray, image_size: int) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def compute_fft_map(image_gray: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(image_gray))
    magnitude = np.log1p(np.abs(spectrum))
    return magnitude.astype(np.float32)


def compute_dct_map(image_gray: np.ndarray) -> np.ndarray:
    transformed = dct(dct(image_gray.T, norm="ortho").T, norm="ortho")
    return np.log1p(np.abs(transformed)).astype(np.float32)


def build_dct_display_map(raw_dct: np.ndarray) -> np.ndarray:
    height, width = raw_dct.shape
    mask = np.ones_like(raw_dct, dtype=np.float32)
    mask[: max(4, height // 8), : max(4, width // 8)] = 0.35
    weighted = raw_dct * mask

    low = float(np.percentile(weighted, 5))
    high = float(np.percentile(weighted, 99))
    scaled = np.clip((weighted - low) / (high - low + 1e-6), 0.0, 1.0)
    return scaled.astype(np.float32)


def normalize_map(freq_map: np.ndarray, mode: str) -> np.ndarray:
    if mode == "standard":
        mean = float(freq_map.mean())
        std = float(freq_map.std()) + 1e-6
        return (freq_map - mean) / std

    min_value = float(freq_map.min())
    max_value = float(freq_map.max())
    return (freq_map - min_value) / (max_value - min_value + 1e-6)


def to_display_image(image: np.ndarray) -> np.ndarray:
    normalized = normalize_map(image, "minmax")
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def build_frequency_artifacts(
    face_bgr: np.ndarray, config: FrequencyMapConfig
) -> dict[str, np.ndarray]:
    face_resized = cv2.resize(
        face_bgr, (config.image_size, config.image_size), interpolation=cv2.INTER_AREA
    )
    gray = preprocess_face(face_bgr, config.image_size)
    raw_fft = compute_fft_map(gray)
    raw_dct = compute_dct_map(gray)
    normalized_fft = normalize_map(raw_fft, config.normalize_mode)
    normalized_dct = normalize_map(raw_dct, config.normalize_mode)
    heatmap_source = to_display_image(raw_fft)
    heatmap = cv2.applyColorMap(heatmap_source, cv2.COLORMAP_INFERNO)
    dct_display = build_dct_display_map(raw_dct)
    dct_display_uint8 = (dct_display * 255.0).clip(0, 255).astype(np.uint8)
    dct_heatmap = cv2.applyColorMap(dct_display_uint8, cv2.COLORMAP_INFERNO)

    return {
        "face_bgr": face_resized,
        "gray_uint8": (gray * 255.0).clip(0, 255).astype(np.uint8),
        "raw_fft": raw_fft,
        "raw_dct": raw_dct,
        "normalized_fft": normalized_fft,
        "normalized_dct": normalized_dct,
        "heatmap_bgr": heatmap,
        "dct_display": dct_display,
        "dct_heatmap_bgr": dct_heatmap,
    }


def build_spatial_tensor(face_bgr: np.ndarray, image_size: int) -> np.ndarray:
    resized = cv2.resize(face_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return (chw - IMAGENET_MEAN) / IMAGENET_STD


def build_frequency_tensor(face_bgr: np.ndarray, config: FrequencyMapConfig) -> np.ndarray:
    artifacts = build_frequency_artifacts(face_bgr, config)
    return np.stack([artifacts["normalized_fft"], artifacts["normalized_dct"]], axis=0).astype(
        np.float32
    )
