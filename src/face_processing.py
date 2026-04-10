from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg"}


@dataclass
class FaceDetectionConfig:
    image_size: int = 224
    face_margin: int = 24
    detector_device: str = "cpu"
    fail_on_missing_face: bool = False


def build_mtcnn(device: str = "cpu"):
    try:
        from facenet_pytorch import MTCNN
    except ImportError as exc:
        raise ImportError(
            "facenet-pytorch is required for face extraction. Install requirements first."
        ) from exc

    return MTCNN(keep_all=True, device=device)


def iter_video_frames(
    video_path: str | Path, frame_stride: int = 10, max_frames: int | None = None
) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frame_index = 0
    emitted = 0

    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_stride == 0:
                yield frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break

            frame_index += 1
    finally:
        capture.release()


def _square_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, margin: int) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width, x2 + margin)
    y2 = min(height, y2 + margin)

    box_width = x2 - x1
    box_height = y2 - y1
    side = max(box_width, box_height)
    center_x = x1 + box_width // 2
    center_y = y1 + box_height // 2

    half_side = side // 2
    left = max(0, center_x - half_side)
    top = max(0, center_y - half_side)
    right = min(width, left + side)
    bottom = min(height, top + side)

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Computed face crop is empty.")
    return crop


def _normalize_rgb_for_mtcnn(frame_bgr: np.ndarray) -> tuple[np.ndarray, Image.Image]:
    if frame_bgr.ndim == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

    frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
    frame_pil = Image.fromarray(frame_rgb).convert("RGB")
    return frame_rgb, frame_pil


def _detect_boxes(detector, frame_bgr: np.ndarray):
    frame_rgb, frame_pil = _normalize_rgb_for_mtcnn(frame_bgr)
    try:
        boxes, probs = detector.detect(frame_pil)
    except RuntimeError as exc:
        if "Could not infer dtype of numpy.uint8" not in str(exc):
            raise
        fallback = np.asarray(frame_pil, dtype=np.float32)
        boxes, probs = detector.detect(fallback)
    return boxes, probs


def detect_largest_face(frame_bgr: np.ndarray, detector, margin: int) -> np.ndarray | None:
    boxes, _ = _detect_boxes(detector, frame_bgr)
    if boxes is None or len(boxes) == 0:
        return None

    largest = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    x1, y1, x2, y2 = [int(value) for value in largest]
    return _square_crop(frame_bgr, x1, y1, x2, y2, margin)


def detect_largest_face_context(
    frame_bgr: np.ndarray, detector, margin: int
) -> tuple[np.ndarray, np.ndarray] | None:
    boxes, _ = _detect_boxes(detector, frame_bgr)
    if boxes is None or len(boxes) == 0:
        return None

    largest = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    x1, y1, x2, y2 = [int(value) for value in largest]
    crop = _square_crop(frame_bgr, x1, y1, x2, y2, margin)
    annotated = frame_bgr.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (64, 196, 230), 2, cv2.LINE_AA)
    return crop, annotated


def load_media_frame(path: str | Path) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Unable to read image: {path}")
    return frame


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS
