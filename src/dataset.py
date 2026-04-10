from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .utils import list_images


@dataclass
class Sample:
    path: Path
    label: int
    class_name: str


CLASS_TO_LABEL = {"real": 0, "fake": 1}
LABEL_TO_CLASS = {0: "real", 1: "fake"}


def load_split(split_dir: str | Path) -> List[Sample]:
    split_path = Path(split_dir)
    samples: List[Sample] = []

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = split_path / class_name
        for image_path in list_images(class_dir):
            samples.append(Sample(path=image_path, label=label, class_name=class_name))

    return samples


def validate_split(split_name: str, samples: List[Sample]) -> None:
    if not samples:
        raise ValueError(
            f"No images were found in the '{split_name}' split. "
            f"Expected folders like '{split_name}/real' and '{split_name}/fake'."
        )


def get_paths_and_labels(samples: List[Sample]) -> Tuple[list[str], list[int]]:
    paths = [str(sample.path) for sample in samples]
    labels = [sample.label for sample in samples]
    return paths, labels
