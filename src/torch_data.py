from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .frequency_maps import FrequencyMapConfig, build_frequency_tensor, build_spatial_tensor
from .utils import list_images


CLASS_TO_LABEL = {"real": 0, "fake": 1}


@dataclass
class AugmentationConfig:
    horizontal_flip: float = 0.5
    gaussian_blur: float = 0.15
    jpeg_noise: float = 0.2
    brightness_jitter: float = 0.15
    contrast_jitter: float = 0.15


class MultiModalImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        map_config: FrequencyMapConfig,
        augmentation: AugmentationConfig | None = None,
        training: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.map_config = map_config
        self.augmentation = augmentation
        self.training = training
        self.samples: list[tuple[Path, int]] = []

        for class_name, label in CLASS_TO_LABEL.items():
            for path in list_images(self.root_dir / class_name):
                self.samples.append((path, label))

        if not self.samples:
            raise ValueError(
                f"No images found in {self.root_dir}. Expected subfolders 'real' and 'fake'."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentations(self, image_bgr):
        if not self.training or self.augmentation is None:
            return image_bgr

        image = image_bgr.copy()

        if random.random() < self.augmentation.horizontal_flip:
            image = cv2.flip(image, 1)

        if random.random() < self.augmentation.gaussian_blur:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        if random.random() < self.augmentation.jpeg_noise:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(45, 80)]
            success, encoded = cv2.imencode(".jpg", image, encode_params)
            if success:
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if self.augmentation.brightness_jitter > 0:
            brightness_scale = 1.0 + random.uniform(
                -self.augmentation.brightness_jitter, self.augmentation.brightness_jitter
            )
            image = np.clip(image.astype(np.float32) * brightness_scale, 0, 255).astype(np.uint8)

        if self.augmentation.contrast_jitter > 0:
            contrast_scale = 1.0 + random.uniform(
                -self.augmentation.contrast_jitter, self.augmentation.contrast_jitter
            )
            mean = image.mean(axis=(0, 1), keepdims=True)
            image = np.clip((image.astype(np.float32) - mean) * contrast_scale + mean, 0, 255).astype(
                np.uint8
            )

        return image

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unable to read image: {path}")

        image = self._apply_augmentations(image)
        spatial = build_spatial_tensor(image, self.map_config.image_size)
        frequency = build_frequency_tensor(image, self.map_config)
        spatial_tensor = torch.from_numpy(spatial).float()
        frequency_tensor = torch.from_numpy(frequency).float()
        target = torch.tensor(label, dtype=torch.float32)
        return {"spatial": spatial_tensor, "frequency": frequency_tensor}, target
