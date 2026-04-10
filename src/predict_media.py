from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import load_config
from .face_processing import (
    build_mtcnn,
    detect_largest_face,
    is_video_file,
    iter_video_frames,
    load_media_frame,
)
from .frequency_maps import (
    FrequencyMapConfig,
    build_frequency_artifacts,
    build_frequency_tensor,
    build_spatial_tensor,
)
from .models_cnn import DualBranchDeepfakeDetector
from .utils import ensure_dir, is_image_file


@torch.no_grad()
def predict_tensor(
    model, face_bgr: np.ndarray, map_config: FrequencyMapConfig, device: str
) -> tuple[float, np.ndarray | None]:
    spatial = torch.from_numpy(build_spatial_tensor(face_bgr, map_config.image_size)).float().unsqueeze(0).to(device)
    frequency = torch.from_numpy(build_frequency_tensor(face_bgr, map_config)).float().unsqueeze(0).to(device)
    output = model({"spatial": spatial, "frequency": frequency}, return_attention=True)
    if isinstance(output, tuple):
        logit, attention = output
        attention_np = attention.squeeze(0).detach().cpu().numpy() if attention is not None else None
    else:
        logit = output
        attention_np = None
    return float(torch.sigmoid(logit).item()), attention_np


def save_prediction_visual(
    output_path: Path,
    media_name: str,
    label: str,
    fake_probability: float,
    artifacts: dict[str, np.ndarray],
    attention_weights: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    face_rgb = cv2.cvtColor(artifacts["face_bgr"], cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(artifacts["heatmap_bgr"], cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    axes[0].imshow(face_rgb)
    axes[0].set_title("Face Crop")
    axes[0].axis("off")

    axes[1].imshow(artifacts["gray_uint8"], cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(heatmap_rgb)
    axes[2].set_title("FFT Heatmap")
    axes[2].axis("off")

    dct_view = axes[3].imshow(artifacts["raw_dct"], cmap="inferno")
    axes[3].set_title("DCT Spectrum")
    axes[3].axis("off")
    fig.colorbar(dct_view, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{media_name} | Prediction: {label} | Fake probability: {fake_probability:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Dual-branch model uses spatial RGB cues together with FFT and DCT frequency patterns. Bright center shows low-frequency structure; outer regions show fine detail and noise.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    if attention_weights is not None and len(attention_weights) == 2:
        fig.text(
            0.5,
            0.08,
            f"Attention weights -> Spatial: {attention_weights[0]:.3f} | Frequency: {attention_weights[1]:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict real/fake for one image or video.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--media", required=True, help="Path to image or video.")
    parser.add_argument("--media-type", choices=["image", "video"], required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = torch.load(config["output"]["checkpoint_path"], map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DualBranchDeepfakeDetector(
        spatial_backbone=checkpoint["spatial_backbone"],
        pretrained=False,
        freeze_backbone=checkpoint.get("freeze_backbone", False),
        frequency_channels=checkpoint["frequency_channels"],
        fusion_hidden_dim=checkpoint["fusion_hidden_dim"],
        use_attention_fusion=checkpoint.get("use_attention_fusion", True),
        attention_hidden_dim=checkpoint.get("attention_hidden_dim", 64),
        dropout=checkpoint["dropout"],
        use_spatial=checkpoint.get("use_spatial", True),
        use_fft=checkpoint.get("use_fft", True),
        use_dct=checkpoint.get("use_dct", True),
        spatial_logit_bias=checkpoint.get("spatial_logit_bias", 0.0),
        frequency_logit_bias=checkpoint.get("frequency_logit_bias", 0.0),
        attention_temperature=checkpoint.get("attention_temperature", 1.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    map_config = FrequencyMapConfig(**checkpoint["map_config"])
    threshold = float(checkpoint.get("threshold", 0.5))
    detector = build_mtcnn(config["preprocessing"].get("detector_device", "cpu"))
    margin = config["preprocessing"].get("face_margin", 24)
    visualization_dir = Path(config["output"].get("visualization_dir", "outputs/visualizations"))

    media_path = Path(args.media)
    scores: list[float] = []
    attention_scores: list[np.ndarray] = []
    visualization_artifacts: dict[str, np.ndarray] | None = None

    if args.media_type == "image":
        if not media_path.exists() or not is_image_file(media_path):
            raise FileNotFoundError(f"Unsupported or missing image: {media_path}")
        frame = load_media_frame(media_path)
        face = detect_largest_face(frame, detector, margin)
        if face is None:
            raise ValueError("No face detected in the input image.")
        visualization_artifacts = build_frequency_artifacts(face, map_config)
        score, attention = predict_tensor(model, face, map_config, device)
        scores.append(score)
        if attention is not None:
            attention_scores.append(attention)
    else:
        if not media_path.exists() or not is_video_file(media_path):
            raise FileNotFoundError(f"Unsupported or missing video: {media_path}")

        for frame in iter_video_frames(
            media_path,
            frame_stride=config["preprocessing"].get("frame_stride", 10),
            max_frames=config["preprocessing"].get("frames_per_video", 12),
        ):
            face = detect_largest_face(frame, detector, margin)
            if face is None:
                continue
            frame_artifacts = build_frequency_artifacts(face, map_config)
            score, attention = predict_tensor(model, face, map_config, device)
            scores.append(score)
            if attention is not None:
                attention_scores.append(attention)
            if visualization_artifacts is None:
                visualization_artifacts = frame_artifacts

        if not scores:
            raise ValueError("No faces were detected in the sampled video frames.")

    fake_probability = float(np.mean(scores))
    label = "fake" if fake_probability >= threshold else "real"

    print(f"Media: {media_path}")
    print(f"Prediction: {label}")
    print(f"Fake probability: {fake_probability:.4f}")
    if len(scores) > 1:
        print(f"Frames used: {len(scores)}")
    mean_attention = None
    if attention_scores:
        mean_attention = np.mean(np.stack(attention_scores, axis=0), axis=0)
        print(f"Spatial attention: {mean_attention[0]:.4f}")
        print(f"Frequency attention: {mean_attention[1]:.4f}")
    if visualization_artifacts is not None:
        visual_path = visualization_dir / f"{media_path.stem}_prediction_panel.png"
        save_prediction_visual(
            visual_path,
            media_path.name,
            label,
            fake_probability,
            visualization_artifacts,
            mean_attention,
        )
        print(f"Visualization: {visual_path}")


if __name__ == "__main__":
    main()
