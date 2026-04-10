from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import load_config
from .face_processing import build_mtcnn, detect_largest_face, load_media_frame
from .frequency_maps import FrequencyMapConfig, build_frequency_artifacts
from .utils import ensure_dir


def save_visual_panel(
    artifacts: dict[str, np.ndarray],
    output_path: str | Path,
    title: str,
    explanation_lines: list[str],
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    face_rgb = cv2.cvtColor(artifacts["face_bgr"], cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(artifacts["heatmap_bgr"], cv2.COLOR_BGR2RGB)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    axes[0].imshow(face_rgb)
    axes[0].set_title("Detected Face")
    axes[0].axis("off")

    axes[1].imshow(artifacts["gray_uint8"], cmap="gray")
    axes[1].set_title("Grayscale Input")
    axes[1].axis("off")

    axes[2].imshow(heatmap_rgb)
    axes[2].set_title("Frequency Heatmap")
    axes[2].axis("off")

    spectrum = axes[3].imshow(artifacts["raw_dct"], cmap="inferno")
    axes[3].set_title("DCT Spectrum")
    axes[3].axis("off")
    figure.colorbar(spectrum, ax=axes[3], fraction=0.046, pad=0.04)

    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.02,
        "\n".join(explanation_lines),
        ha="center",
        va="bottom",
        fontsize=10,
        wrap=True,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize frequency-domain artifacts for one image.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument(
        "--output",
        default="outputs/visualizations/frequency_panel.png",
        help="Path to save the visualization panel.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    image_path = Path(args.image)
    frame = load_media_frame(image_path)

    detector = build_mtcnn(config["preprocessing"].get("detector_device", "cpu"))
    face = detect_largest_face(frame, detector, config["preprocessing"].get("face_margin", 24))
    if face is None:
        raise ValueError("No face detected in the input image.")

    map_config = FrequencyMapConfig(
        image_size=config["dataset"]["image_size"],
        normalize_mode=config["preprocessing"].get("normalize_mode", "minmax"),
    )
    artifacts = build_frequency_artifacts(face, map_config)
    explanation_lines = [
        "Center region corresponds to low-frequency structure; edges correspond to high-frequency detail.",
        "Real faces usually show smoother decay, while manipulated faces often introduce irregular spikes or missing fine detail.",
        "This panel shows FFT heatmap coloring and a raw DCT spectrum so both frequency views are visible.",
    ]
    save_visual_panel(artifacts, args.output, "Frequency Domain Analysis", explanation_lines)
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
