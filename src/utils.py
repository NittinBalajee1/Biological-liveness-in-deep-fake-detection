from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def list_images(directory: str | Path) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    )


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def flatten(items: Iterable[Iterable[float]]) -> list[float]:
    return [value for sublist in items for value in sublist]
