from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
from tqdm import tqdm

from .face_processing import FaceDetectionConfig, build_mtcnn, detect_largest_face, iter_video_frames
from .utils import ensure_dir


def load_split_ids(splits_dir: Path) -> dict[str, set[str]]:
    splits: dict[str, set[str]] = {}
    for split_name in ("train", "val", "test"):
        split_path = splits_dir / f"{split_name}.json"
        pairs = json.loads(split_path.read_text(encoding="utf-8"))
        ids = set()
        for pair in pairs:
            ids.update(pair)
            ids.add("_".join(pair))
            ids.add("_".join(pair[::-1]))
        splits[split_name] = ids
    return splits


def resolve_split(stem: str, split_ids: dict[str, set[str]]) -> str | None:
    for split_name, valid_ids in split_ids.items():
        if stem in valid_ids:
            return split_name
    return None


def build_random_split_map(video_paths: list[Path], seed: int) -> dict[str, str]:
    stems = [path.stem for path in video_paths]
    random.Random(seed).shuffle(stems)
    total = len(stems)
    train_cut = max(1, int(total * 0.7))
    val_cut = max(train_cut + 1, int(total * 0.85)) if total >= 3 else total
    split_map: dict[str, str] = {}
    for stem in stems[:train_cut]:
        split_map[stem] = "train"
    for stem in stems[train_cut:val_cut]:
        split_map[stem] = "val"
    for stem in stems[val_cut:]:
        split_map[stem] = "test"
    if total >= 3:
        if "val" not in split_map.values():
            split_map[stems[-2]] = "val"
        if "test" not in split_map.values():
            split_map[stems[-1]] = "test"
    return split_map


def save_face(face, output_path: Path, image_size: int) -> None:
    ensure_dir(output_path.parent)
    resized = cv2.resize(face, (image_size, image_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), resized)


def process_video_dir(
    video_dir: Path,
    class_name: str,
    output_dir: Path,
    split_ids: dict[str, set[str]],
    detector,
    detection_config: FaceDetectionConfig,
    frame_stride: int,
    max_frames_per_clip: int,
    split_override: dict[str, str] | None = None,
) -> int:
    saved = 0
    videos = sorted(path for path in video_dir.glob("*.mp4"))
    for video_path in tqdm(videos, desc=f"ffpp/{class_name}", leave=False):
        split_name = (
            split_override.get(video_path.stem)
            if split_override is not None
            else resolve_split(video_path.stem, split_ids)
        )
        if split_name is None:
            continue

        frame_count = 0
        for frame_index, frame in enumerate(
            iter_video_frames(
                video_path,
                frame_stride=frame_stride,
                max_frames=max_frames_per_clip,
            )
        ):
            face = detect_largest_face(frame, detector, detection_config.face_margin)
            if face is None:
                if detection_config.fail_on_missing_face:
                    raise ValueError(f"No face detected in: {video_path}")
                continue

            output_name = f"{video_path.stem}_frame_{frame_index:03d}.png"
            output_path = output_dir / split_name / class_name / output_name
            save_face(face, output_path, detection_config.image_size)
            frame_count += 1
            saved += 1

        if frame_count == 0 and detection_config.fail_on_missing_face:
            raise ValueError(f"No usable frames extracted from: {video_path}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare FaceForensics++ splits directly from downloaded videos."
    )
    parser.add_argument(
        "--ffpp-root",
        required=True,
        help=(
            "Path to downloaded FaceForensics++ data containing original_sequences/ and "
            "manipulated_sequences/ video folders."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for train/val/test.")
    parser.add_argument(
        "--compression",
        default="c23",
        choices=["raw", "c23", "c40"],
        help="Compression level to use.",
    )
    parser.add_argument(
        "--manipulations",
        nargs="+",
        default=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
        help="Manipulation methods to include as fake class.",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Prepare only the original/youtube class and skip manipulated videos.",
    )
    parser.add_argument(
        "--splits-dir",
        default="FaceForensics-master/dataset/splits",
        help="Path to the official FaceForensics++ split JSON files.",
    )
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames-per-clip", type=int, default=20)
    parser.add_argument("--face-margin", type=int, default=24)
    parser.add_argument("--detector-device", default="cpu")
    parser.add_argument("--fail-on-missing-face", action="store_true")
    parser.add_argument(
        "--fallback-random-split",
        action="store_true",
        help="Use a random 70/15/15 split over available videos instead of official FF++ splits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ffpp_root = Path(args.ffpp_root)
    output_dir = Path(args.output_dir)
    splits_dir = Path(args.splits_dir)
    ensure_dir(output_dir)

    real_root = ffpp_root / "original_sequences" / "youtube" / args.compression / "videos"
    if not real_root.exists():
        raise FileNotFoundError(f"Real videos not found: {real_root}")

    split_ids = load_split_ids(splits_dir)
    detector = build_mtcnn(args.detector_device)
    detection_config = FaceDetectionConfig(
        image_size=args.image_size,
        face_margin=args.face_margin,
        detector_device=args.detector_device,
        fail_on_missing_face=args.fail_on_missing_face,
    )

    split_override = None
    if args.fallback_random_split:
        all_video_paths = list(real_root.glob("*.mp4"))
        if not args.real_only:
            for manipulation in args.manipulations:
                fake_root = ffpp_root / "manipulated_sequences" / manipulation / args.compression / "videos"
                if fake_root.exists():
                    all_video_paths.extend(fake_root.glob("*.mp4"))
        split_override = build_random_split_map(sorted(all_video_paths), args.seed)

    total_saved = process_video_dir(
        real_root,
        "real",
        output_dir,
        split_ids,
        detector,
        detection_config,
        args.frame_stride,
        args.max_frames_per_clip,
        split_override=split_override,
    )

    if not args.real_only:
        for manipulation in args.manipulations:
            fake_root = ffpp_root / "manipulated_sequences" / manipulation / args.compression / "videos"
            if not fake_root.exists():
                raise FileNotFoundError(f"Manipulation videos not found: {fake_root}")
            total_saved += process_video_dir(
                fake_root,
                "fake",
                output_dir,
                split_ids,
                detector,
                detection_config,
                args.frame_stride,
                args.max_frames_per_clip,
                split_override=split_override,
            )

    print(f"Prepared FaceForensics++ splits at: {output_dir}")
    print(f"Saved face crops: {total_saved}")


if __name__ == "__main__":
    main()
