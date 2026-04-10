from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .utils import ensure_dir


def create_real_like_image(size: int) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    noise = np.random.normal(127, 35, (size, size, 3)).clip(0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(noise, (7, 7), 0)
    cv2.circle(image, (size // 2, size // 2), size // 5, (180, 180, 180), -1)
    cv2.rectangle(image, (size // 4, size // 3), (size // 2, size // 2), (90, 90, 90), -1)
    return image


def create_fake_like_image(size: int) -> np.ndarray:
    base = create_real_like_image(size)
    upscaled = cv2.resize(base, (size // 2, size // 2), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(upscaled, (size, size), interpolation=cv2.INTER_CUBIC)

    for x in range(0, size, 8):
        restored[:, x : x + 1] = np.clip(restored[:, x : x + 1] + 25, 0, 255)
    for y in range(0, size, 8):
        restored[y : y + 1, :] = np.clip(restored[y : y + 1, :] - 20, 0, 255)

    return restored.astype(np.uint8)


def generate_split(split_dir: Path, class_name: str, count: int, size: int) -> None:
    ensure_dir(split_dir / class_name)
    generator = create_real_like_image if class_name == "real" else create_fake_like_image

    for index in range(count):
        image = generator(size)
        output_path = split_dir / class_name / f"{class_name}_{index:03d}.png"
        cv2.imwrite(str(output_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiny demo dataset.")
    parser.add_argument("--output-dir", default="data", help="Directory to place generated images.")
    parser.add_argument("--size", type=int, default=256, help="Image size.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    layout = {
        "train": 40,
        "val": 12,
        "test": 12,
    }

    for split_name, count in layout.items():
        split_dir = output_dir / split_name
        generate_split(split_dir, "real", count, args.size)
        generate_split(split_dir, "fake", count, args.size)

    print(f"Sample dataset created at: {output_dir}")


if __name__ == "__main__":
    main()
