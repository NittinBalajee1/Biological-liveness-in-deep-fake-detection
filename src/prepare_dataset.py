from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .face_processing import (
    FaceDetectionConfig,
    build_mtcnn,
    detect_largest_face,
    is_video_file,
    iter_video_frames,
    load_media_frame,
)
from .utils import IMAGE_EXTENSIONS, ensure_dir


def gather_media_groups(input_dir: Path, source_type: str) -> dict[str, list[list[Path]]]:
    media_by_class: dict[str, list[list[Path]]] = defaultdict(list)

    for class_name in ("real", "fake"):
        class_dir = input_dir / class_name
        if not class_dir.exists():
            continue

        if source_type == "videos":
            for path in class_dir.rglob("*"):
                if is_video_file(path):
                    media_by_class[class_name].append([path])
        else:
            grouped_frames: dict[str, list[Path]] = defaultdict(list)
            for path in class_dir.rglob("*"):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                relative = path.relative_to(class_dir)
                if len(relative.parts) >= 3 and relative.parts[0] in {"frames", "landmarks"}:
                    group_key = relative.parts[1]
                elif len(relative.parts) >= 2:
                    group_key = relative.parts[0]
                else:
                    group_key = path.stem
                grouped_frames[group_key].append(path)

            for group_paths in grouped_frames.values():
                media_by_class[class_name].append(sorted(group_paths))

    return media_by_class


def split_groups(groups: list[list[Path]], seed: int) -> dict[str, list[Path]]:
    train_groups, temp_groups = train_test_split(groups, test_size=0.3, random_state=seed)
    val_groups, test_groups = train_test_split(temp_groups, test_size=0.5, random_state=seed)
    return {
        "train": [path for group in train_groups for path in group],
        "val": [path for group in val_groups for path in group],
        "test": [path for group in test_groups for path in group],
    }


def save_face(face, output_path: Path, image_size: int) -> None:
    ensure_dir(output_path.parent)
    resized = cv2.resize(face, (image_size, image_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), resized)


def process_image_dataset(
    paths: list[Path],
    output_root: Path,
    split_name: str,
    class_name: str,
    detector,
    detection_config: FaceDetectionConfig,
    already_cropped: bool,
) -> int:
    saved = 0
    for path in tqdm(paths, desc=f"{split_name}/{class_name}", leave=False):
        frame = load_media_frame(path)
        if already_cropped:
            face = frame
        else:
            face = detect_largest_face(frame, detector, detection_config.face_margin)
            if face is None:
                if detection_config.fail_on_missing_face:
                    raise ValueError(f"No face detected in image: {path}")
                continue

        group_name = path.parent.name
        output_path = output_root / split_name / class_name / f"{group_name}_{path.stem}.png"
        save_face(face, output_path, detection_config.image_size)
        saved += 1
    return saved


def process_video_dataset(
    paths: list[Path],
    output_root: Path,
    split_name: str,
    class_name: str,
    detector,
    detection_config: FaceDetectionConfig,
    frames_per_video: int,
    frame_stride: int,
) -> int:
    saved = 0
    for path in tqdm(paths, desc=f"{split_name}/{class_name}", leave=False):
        for frame_index, frame in enumerate(
            iter_video_frames(path, frame_stride=frame_stride, max_frames=frames_per_video)
        ):
            face = detect_largest_face(frame, detector, detection_config.face_margin)
            if face is None:
                continue

            output_name = f"{path.stem}_frame_{frame_index:03d}.png"
            output_path = output_root / split_name / class_name / output_name
            save_face(face, output_path, detection_config.image_size)
            saved += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare face-cropped dataset splits.")
    parser.add_argument("--input-dir", required=True, help="Input root containing real/ and fake/.")
    parser.add_argument("--output-dir", required=True, help="Output directory for train/val/test.")
    parser.add_argument(
        "--source-type",
        choices=["images", "videos"],
        required=True,
        help="Whether the raw dataset contains images or videos.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frames-per-video", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--face-margin", type=int, default=24)
    parser.add_argument("--detector-device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-on-missing-face", action="store_true")
    parser.add_argument("--already-cropped", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    media_by_class = gather_media_groups(input_dir, args.source_type)
    if not media_by_class["real"] or not media_by_class["fake"]:
        raise ValueError("Expected media files under input_dir/real and input_dir/fake.")

    detector = None if args.already_cropped and args.source_type == "images" else build_mtcnn(args.detector_device)
    detection_config = FaceDetectionConfig(
        image_size=args.image_size,
        face_margin=args.face_margin,
        detector_device=args.detector_device,
        fail_on_missing_face=args.fail_on_missing_face,
    )

    total_saved = 0
    for class_name, groups in media_by_class.items():
        splits = split_groups(groups, args.seed)
        for split_name, split_paths_list in splits.items():
            if args.source_type == "images":
                total_saved += process_image_dataset(
                    split_paths_list,
                    output_dir,
                    split_name,
                    class_name,
                    detector,
                    detection_config,
                    args.already_cropped,
                )
            else:
                total_saved += process_video_dataset(
                    split_paths_list,
                    output_dir,
                    split_name,
                    class_name,
                    detector,
                    detection_config,
                    args.frames_per_video,
                    args.frame_stride,
                )

    print(f"Prepared dataset at: {output_dir}")
    print(f"Saved face crops: {total_saved}")


if __name__ == "__main__":
    main()
