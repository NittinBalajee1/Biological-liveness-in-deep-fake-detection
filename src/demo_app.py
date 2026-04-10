from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import load_config
    from src.face_processing import (
        build_mtcnn,
        detect_largest_face,
        detect_largest_face_context,
        iter_video_frames,
        load_media_frame,
    )
    from src.frequency_maps import (
        FrequencyMapConfig,
        build_frequency_artifacts,
        build_frequency_tensor,
        build_spatial_tensor,
    )
    from src.models_cnn import DualBranchDeepfakeDetector
else:
    from .config import load_config
    from .face_processing import (
        build_mtcnn,
        detect_largest_face,
        detect_largest_face_context,
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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(36, 116, 156, 0.18), transparent 32%),
                radial-gradient(circle at top right, rgba(232, 122, 65, 0.16), transparent 30%),
                linear-gradient(180deg, #eef4f8 0%, #f6f2ec 48%, #f2eee8 100%);
            color: #16242d;
        }
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        h1, h2, h3 {
            color: #122430;
            letter-spacing: -0.02em;
        }
        .hero-card, .info-card, .metric-card, .explain-card {
            border-radius: 22px;
            padding: 1.2rem 1.3rem;
            background: rgba(252, 250, 246, 0.86);
            border: 1px solid rgba(18, 36, 48, 0.08);
            box-shadow: 0 18px 40px rgba(30, 48, 62, 0.08);
            backdrop-filter: blur(10px);
        }
        .hero-card {
            padding: 1.5rem 1.6rem;
            background: linear-gradient(135deg, rgba(243, 249, 255, 0.94), rgba(255, 245, 237, 0.96));
        }
        .hero-kicker {
            color: #c0612b;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.08em;
        }
        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.05;
            margin-top: 0.2rem;
            margin-bottom: 0.8rem;
        }
        .hero-text {
            font-size: 1.03rem;
            color: #39505d;
            line-height: 1.65;
            margin-bottom: 0;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }
        .pill {
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(25, 72, 102, 0.08);
            color: #214255;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .metric-label {
            color: #5f7079;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
            color: #14303d;
        }
        .metric-sub {
            margin-top: 0.5rem;
            color: #566972;
            font-size: 0.95rem;
        }
        .status-real {
            color: #1e7f67;
        }
        .status-fake {
            color: #bf5530;
        }
        .section-note {
            color: #47606d;
            font-size: 0.96rem;
            line-height: 1.65;
        }
        .attention-wrap {
            display: grid;
            gap: 0.8rem;
        }
        .attention-item label {
            display: block;
            font-weight: 700;
            margin-bottom: 0.35rem;
            color: #1f3642;
        }
        .attention-bar {
            width: 100%;
            height: 14px;
            border-radius: 999px;
            background: rgba(20, 48, 61, 0.09);
            overflow: hidden;
        }
        .attention-fill-spatial {
            height: 100%;
            background: linear-gradient(90deg, #1f6f9c, #5e9fc6);
        }
        .attention-fill-frequency {
            height: 100%;
            background: linear-gradient(90deg, #d76938, #ef9b56);
        }
        .upload-card {
            border-radius: 20px;
            border: 1px dashed rgba(21, 52, 70, 0.2);
            background: rgba(255, 255, 255, 0.56);
            padding: 1rem 1rem 0.4rem 1rem;
        }
        .freq-hero {
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            background: linear-gradient(135deg, rgba(17, 57, 82, 0.96), rgba(208, 97, 47, 0.92));
            color: #fff7f1;
            box-shadow: 0 20px 42px rgba(19, 44, 60, 0.22);
            margin-bottom: 1rem;
        }
        .freq-hero h3 {
            color: #fffdf9;
            margin-bottom: 0.35rem;
        }
        .freq-hero p {
            margin: 0;
            font-size: 0.98rem;
            line-height: 1.6;
            color: rgba(255, 248, 240, 0.92);
        }
        .freq-caption {
            border-radius: 18px;
            padding: 0.85rem 1rem;
            background: rgba(19, 51, 69, 0.92);
            color: #f4f8fb;
            margin-top: 0.55rem;
            margin-bottom: 0.85rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .freq-caption strong {
            color: #ffd8bf;
        }
        .freq-caption span {
            display: block;
            margin-top: 0.3rem;
            color: rgba(244, 248, 251, 0.86);
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_bundle(config_path: str):
    config = load_config(config_path)
    checkpoint = torch.load(config["output"]["checkpoint_path"], map_location="cpu")
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
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    map_config = FrequencyMapConfig(**checkpoint["map_config"])
    detector = build_mtcnn(config["preprocessing"].get("detector_device", "cpu"))
    return config, checkpoint, model, map_config, detector


@torch.no_grad()
def score_face(model, face_bgr: np.ndarray, map_config: FrequencyMapConfig):
    spatial = torch.from_numpy(build_spatial_tensor(face_bgr, map_config.image_size)).float().unsqueeze(0)
    frequency = torch.from_numpy(build_frequency_tensor(face_bgr, map_config)).float().unsqueeze(0)
    output = model({"spatial": spatial, "frequency": frequency}, return_attention=True)
    if isinstance(output, tuple):
        logits, attention = output
        attention_np = attention.squeeze(0).cpu().numpy() if attention is not None else None
    else:
        logits = output
        attention_np = None
    probability = float(torch.sigmoid(logits).item())
    return probability, attention_np


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Deepfake Detection Using Frequency Domain Analysis</div>
            <div class="hero-title">Spatial + FFT + DCT + Attention Fusion</div>
            <p class="hero-text">
                This demo analyzes the face in both spatial and frequency domains. The model studies
                visible facial structure, FFT energy patterns, and DCT compression cues, then uses
                attention fusion to decide whether the input looks naturally captured or synthetically manipulated.
            </p>
            <div class="pill-row">
                <div class="pill">Face-Centered Analysis</div>
                <div class="pill">FFT Heatmap</div>
                <div class="pill">DCT Spectrum</div>
                <div class="pill">Attention-Based Fusion</div>
                <div class="pill">Explainable Output</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str, is_status: bool = False) -> None:
    status_class = ""
    if is_status:
        status_class = "status-fake" if value.upper() == "FAKE" else "status-real"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {status_class}">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_label(probability: float, threshold: float, uncertainty_margin: float = 0.08) -> tuple[str, str]:
    if probability >= threshold:
        label = "fake"
    else:
        label = "real"

    if abs(probability - threshold) <= uncertainty_margin:
        note = "Borderline decision. This sample sits close to the threshold."
    elif label == "fake":
        note = "Model sees enough suspicious evidence to cross the current fake threshold."
    else:
        note = "Model sees stronger natural evidence than suspicious evidence under the current threshold."
    return label, note


def render_attention(attention: np.ndarray | None) -> None:
    if attention is None or len(attention) != 2:
        st.info("Attention weights are only available when both spatial and frequency branches are active.")
        return

    spatial_pct = float(attention[0]) * 100.0
    frequency_pct = float(attention[1]) * 100.0
    st.markdown('<div class="info-card"><h3>Branch Attention</h3>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="attention-wrap">
            <div class="attention-item">
                <label>Spatial branch: {spatial_pct:.1f}%</label>
                <div class="attention-bar">
                    <div class="attention-fill-spatial" style="width:{spatial_pct:.1f}%"></div>
                </div>
            </div>
            <div class="attention-item">
                <label>Frequency branch: {frequency_pct:.1f}%</label>
                <div class="attention-bar">
                    <div class="attention-fill-frequency" style="width:{frequency_pct:.1f}%"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p class="section-note">
            Higher spatial weight means the model trusted visible facial appearance more.
            Higher frequency weight means the model leaned more on spectral irregularities like
            unnatural texture, noise, or compression behavior.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_frequency_insights(artifacts: dict[str, np.ndarray]) -> dict[str, float]:
    fft_map = artifacts["normalized_fft"]
    dct_map = artifacts["normalized_dct"]
    height, width = fft_map.shape
    cy, cx = height // 2, width // 2

    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_mask = radius <= min(height, width) * 0.18
    high_mask = radius >= min(height, width) * 0.35

    fft_low = float(fft_map[low_mask].mean())
    fft_high = float(fft_map[high_mask].mean())
    fft_ratio = fft_high / (fft_low + 1e-6)

    dct_h, dct_w = dct_map.shape
    low_block = dct_map[: dct_h // 4, : dct_w // 4]
    high_block = dct_map[dct_h // 3 :, dct_w // 3 :]
    dct_low = float(low_block.mean())
    dct_high = float(high_block.mean())
    dct_ratio = dct_high / (dct_low + 1e-6)

    high_fft = fft_map.copy()
    high_fft[~high_mask] = 0.0
    suspicious_fft_index = np.unravel_index(int(np.argmax(high_fft)), high_fft.shape)

    suspicious_dct_region = dct_map[dct_h // 3 :, dct_w // 3 :]
    region_index = np.unravel_index(int(np.argmax(suspicious_dct_region)), suspicious_dct_region.shape)
    suspicious_dct_index = (region_index[0] + dct_h // 3, region_index[1] + dct_w // 3)

    return {
        "fft_low_mean": fft_low,
        "fft_high_mean": fft_high,
        "fft_high_low_ratio": fft_ratio,
        "dct_low_mean": dct_low,
        "dct_high_mean": dct_high,
        "dct_high_low_ratio": dct_ratio,
        "fft_suspicious_y": float(suspicious_fft_index[0]),
        "fft_suspicious_x": float(suspicious_fft_index[1]),
        "dct_suspicious_y": float(suspicious_dct_index[0]),
        "dct_suspicious_x": float(suspicious_dct_index[1]),
    }


def compute_radial_profile(freq_map: np.ndarray, bins: int = 32) -> tuple[np.ndarray, np.ndarray]:
    height, width = freq_map.shape
    cy, cx = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius_norm = radius / (radius.max() + 1e-6)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile = []
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (radius_norm >= start) & (radius_norm < end)
        profile.append(float(freq_map[mask].mean()) if np.any(mask) else 0.0)
    return centers, np.array(profile, dtype=np.float32)


def aggregate_artifacts(artifacts_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not artifacts_list:
        raise ValueError("No artifacts to aggregate.")
    middle = len(artifacts_list) // 2
    raw_fft = np.mean(np.stack([item["raw_fft"] for item in artifacts_list], axis=0), axis=0)
    raw_dct = np.mean(np.stack([item["raw_dct"] for item in artifacts_list], axis=0), axis=0)
    normalized_fft = np.mean(
        np.stack([item["normalized_fft"] for item in artifacts_list], axis=0), axis=0
    )
    normalized_dct = np.mean(
        np.stack([item["normalized_dct"] for item in artifacts_list], axis=0), axis=0
    )
    dct_display = np.mean(
        np.stack([item["dct_display"] for item in artifacts_list], axis=0), axis=0
    )
    heatmap_source = (normalized_fft * 255.0).clip(0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_source, cv2.COLORMAP_INFERNO)
    dct_heatmap_source = (dct_display * 255.0).clip(0, 255).astype(np.uint8)
    dct_heatmap_bgr = cv2.applyColorMap(dct_heatmap_source, cv2.COLORMAP_INFERNO)
    return {
        "frame_bgr": artifacts_list[middle]["frame_bgr"],
        "face_bgr": artifacts_list[middle]["face_bgr"],
        "gray_uint8": artifacts_list[middle]["gray_uint8"],
        "raw_fft": raw_fft.astype(np.float32),
        "raw_dct": raw_dct.astype(np.float32),
        "normalized_fft": normalized_fft.astype(np.float32),
        "normalized_dct": normalized_dct.astype(np.float32),
        "heatmap_bgr": heatmap_bgr,
        "dct_display": dct_display.astype(np.float32),
        "dct_heatmap_bgr": dct_heatmap_bgr,
    }


def draw_corner_brackets(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    length: int = 18,
    thickness: int = 2,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    corners = [
        ((x1, y1), (x1 + length, y1), (x1, y1 + length)),
        ((x2, y1), (x2 - length, y1), (x2, y1 + length)),
        ((x1, y2), (x1 + length, y2), (x1, y2 - length)),
        ((x2, y2), (x2 - length, y2), (x2, y2 - length)),
    ]
    for origin, horizontal, vertical in corners:
        cv2.line(image, origin, horizontal, color, thickness, cv2.LINE_AA)
        cv2.line(image, origin, vertical, color, thickness, cv2.LINE_AA)


def build_annotated_frequency_views(
    artifacts: dict[str, np.ndarray], insights: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    fft_rgb = cv2.cvtColor(artifacts["heatmap_bgr"], cv2.COLOR_BGR2RGB)
    dct_rgb = cv2.cvtColor(artifacts["dct_heatmap_bgr"], cv2.COLOR_BGR2RGB)

    h, w = fft_rgb.shape[:2]
    center = (w // 2, h // 2)
    low_radius = int(min(h, w) * 0.18)
    high_radius = int(min(h, w) * 0.35)

    fft_overlay = fft_rgb.copy()
    cv2.circle(fft_overlay, center, low_radius, (116, 216, 230), 2, cv2.LINE_AA)
    cv2.circle(fft_overlay, center, high_radius, (237, 187, 92), 1, cv2.LINE_AA)
    cv2.addWeighted(fft_overlay, 0.75, fft_rgb, 0.25, 0, fft_rgb)
    fft_point = (int(insights["fft_suspicious_x"]), int(insights["fft_suspicious_y"]))
    cv2.circle(fft_rgb, fft_point, 4, (242, 107, 90), -1, cv2.LINE_AA)
    cv2.circle(fft_rgb, fft_point, 10, (255, 214, 208), 1, cv2.LINE_AA)

    dh, dw = dct_rgb.shape[:2]
    dct_overlay = dct_rgb.copy()
    draw_corner_brackets(
        dct_overlay,
        (0, 0),
        (dw // 4, dh // 4),
        (116, 216, 230),
        length=max(12, dw // 14),
        thickness=2,
    )
    cv2.addWeighted(dct_overlay, 0.75, dct_rgb, 0.25, 0, dct_rgb)
    dct_point = (int(insights["dct_suspicious_x"]), int(insights["dct_suspicious_y"]))
    cv2.circle(dct_rgb, dct_point, 4, (242, 107, 90), -1, cv2.LINE_AA)
    cv2.circle(dct_rgb, dct_point, 10, (255, 214, 208), 1, cv2.LINE_AA)
    return fft_rgb, dct_rgb


def render_frequency_interpretation(insights: dict[str, float]) -> None:
    fft_ratio = insights["fft_high_low_ratio"]
    dct_ratio = insights["dct_high_low_ratio"]
    fft_status = "more natural" if fft_ratio < 0.78 else "more suspicious"
    dct_status = "more natural" if dct_ratio < 0.82 else "more suspicious"
    st.markdown(
        f"""
        <div class="info-card">
            <h3>Why FFT And DCT Are Used Together</h3>
            <p class="section-note">
                <strong>FFT</strong> tells us how image energy spreads from low-frequency structure to high-frequency detail.
                Real faces usually show a smoother decay from center to edge. If the outer ring becomes too strong, it can suggest
                synthetic texture or unnatural sharpening.
            </p>
            <p class="section-note">
                <strong>DCT</strong> is useful because manipulated media is often re-encoded or generated with artifacts that show up
                as unusual high-order coefficients. A normal image keeps more energy in the top-left low-order region, while suspicious
                samples can show stronger spikes deeper in the spectrum.
            </p>
            <p class="section-note">
                Visual guide:
                the <strong>cyan circle/box</strong> marks the region where natural low-frequency or low-order energy is expected to dominate.
                the <strong>small red dot</strong> marks a stronger high-frequency or high-order response that may deserve attention.
            </p>
            <p class="section-note">
                Current sample summary:
                FFT high/low ratio = <strong>{fft_ratio:.3f}</strong> ({fft_status}),
                DCT high/low ratio = <strong>{dct_ratio:.3f}</strong> ({dct_status}).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_dct_diagonal_profile(dct_map: np.ndarray, bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    height, width = dct_map.shape
    yy, xx = np.ogrid[:height, :width]
    diag = (yy + xx).astype(np.float32)
    diag_norm = diag / (diag.max() + 1e-6)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile = []
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (diag_norm >= start) & (diag_norm < end)
        profile.append(float(dct_map[mask].mean()) if np.any(mask) else 0.0)
    return centers, np.array(profile, dtype=np.float32)


def render_visual_grid(artifacts: dict[str, np.ndarray]) -> None:
    insights = compute_frequency_insights(artifacts)
    fft_annotated, dct_annotated = build_annotated_frequency_views(artifacts, insights)
    radii, fft_profile = compute_radial_profile(artifacts["normalized_fft"])
    _, dct_profile = compute_radial_profile(artifacts["normalized_dct"])
    dct_diag_x, dct_diag_profile = compute_dct_diagonal_profile(artifacts["dct_display"])
    st.markdown(
        """
        <div class="freq-hero">
            <h3>Frequency Domain Spotlight</h3>
            <p>
                This is the core of the project. FFT highlights how visual energy spreads across low and high frequencies,
                while DCT emphasizes block-style compression and structural irregularities. These two views help expose
                artifacts that may be hard to see in the normal image alone.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    image_col_1, image_col_2 = st.columns(2)
    image_col_1.image(
        cv2.cvtColor(artifacts["frame_bgr"], cv2.COLOR_BGR2RGB),
        caption="Original Frame With Detected Face",
        use_container_width=True,
    )
    image_col_2.image(
        cv2.cvtColor(artifacts["face_bgr"], cv2.COLOR_BGR2RGB),
        caption="Face Crop",
        use_container_width=True,
    )
    image_col_1, image_col_2 = st.columns(2)
    image_col_1.image(
        artifacts["gray_uint8"],
        caption="Grayscale Structure",
        use_container_width=True,
    )

    image_col_2.image(
        fft_annotated,
        caption="Annotated FFT Heatmap",
        use_container_width=True,
    )
    image_col_2.markdown(
        """
        <div class="freq-caption">
            <strong>FFT View</strong>
            <span>
                The center naturally carries coarse facial structure and lighting. The small red point marks a stronger
                high-frequency response that can indicate extra sharpening, synthetic texture, or unusual noise.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    image_col_3, image_col_4 = st.columns(2)
    image_col_3.image(
        dct_annotated,
        caption="Annotated DCT Spectrum",
        use_container_width=True,
    )
    image_col_3.markdown(
        """
        <div class="freq-caption">
            <strong>DCT View</strong>
            <span>
                The boxed top-left area is where normal low-order energy should dominate. The small red point marks a
                stronger high-order response, which can be associated with unusual compression or generative artifacts.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    image_col_4.markdown(
        """
        <div class="info-card">
            <h3>Why The Face Is Cropped</h3>
            <p class="section-note">
                Deepfake artifacts are usually concentrated in the face region, so the model analyzes the cropped face for prediction.
                The full frame is still shown here so you can see exactly where the detector focused.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_frequency_interpretation(insights)
    st.markdown(
        """
        <div class="info-card">
            <h3>Frequency Decay Profile</h3>
            <p class="section-note">
                This chart shows how spectral energy changes as we move from the center toward higher-frequency regions.
                Real content usually decays more smoothly, while suspicious samples may keep unusually strong energy farther out.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.plot(radii, fft_profile, color="#1f6f9c", linewidth=2.4, label="FFT")
    ax.plot(radii, dct_profile, color="#d76938", linewidth=2.4, label="DCT")
    ax.set_xlabel("Normalized Frequency Radius")
    ax.set_ylabel("Average Energy")
    ax.set_title("Energy From Low To High Frequency")
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(frameon=False)
    ax.set_facecolor("#fbf8f3")
    fig.patch.set_facecolor("#fbf8f3")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown(
        """
        <div class="info-card">
            <h3>DCT Coefficient Profile</h3>
            <p class="section-note">
                This view tracks DCT energy from low-order coefficients near the top-left toward higher-order coefficients deeper in the spectrum.
                It is more directly tied to compression-style behavior than the FFT radial plot.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.plot(dct_diag_x, dct_diag_profile, color="#d76938", linewidth=2.5)
    ax.fill_between(dct_diag_x, dct_diag_profile, alpha=0.16, color="#d76938")
    ax.set_xlabel("Normalized DCT Order")
    ax.set_ylabel("Average DCT Energy")
    ax.set_title("DCT Low-Order To High-Order Energy")
    ax.grid(alpha=0.22, linestyle="--")
    ax.set_facecolor("#fbf8f3")
    fig.patch.set_facecolor("#fbf8f3")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_frame_strip(face_frames: list[np.ndarray]) -> None:
    if len(face_frames) <= 1:
        return
    st.markdown(
        """
        <div class="info-card">
            <h3>Frames Used For Video Analysis</h3>
            <p class="section-note">
                The detector averages evidence across multiple sampled frames, then visualizes the combined frequency response.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    preview_frames = face_frames[: min(5, len(face_frames))]
    cols = st.columns(len(preview_frames))
    for idx, (col, face) in enumerate(zip(cols, preview_frames, strict=False), start=1):
        col.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption=f"Frame {idx}", use_container_width=True)


def render_frame_score_chart(frame_scores: list[float]) -> None:
    if len(frame_scores) <= 1:
        st.warning("Only one valid face frame was analyzed for this video. Increase the frame budget or use a clearer face clip for stronger temporal evidence.")
        return
    st.markdown(
        """
        <div class="info-card">
            <h3>Per-Frame Fake Scores</h3>
            <p class="section-note">
                This chart proves the video prediction is being formed from multiple sampled frames. The final decision uses the average of these scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    x = np.arange(1, len(frame_scores) + 1)
    y = np.array(frame_scores, dtype=np.float32)
    ax.plot(x, y, marker="o", linewidth=2.2, color="#bf5530")
    ax.fill_between(x, y, alpha=0.16, color="#bf5530")
    ax.set_xlabel("Analyzed Frame")
    ax.set_ylabel("Fake Probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Frame-by-Frame Suspicion")
    ax.grid(alpha=0.22, linestyle="--")
    ax.set_facecolor("#fbf8f3")
    fig.patch.set_facecolor("#fbf8f3")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_explanation(label: str, probability: float, attention: np.ndarray | None) -> None:
    confidence = probability if label == "fake" else 1.0 - probability
    branch_line = "The current model is blending both branches."
    if attention is not None and len(attention) == 2:
        dominant = "spatial" if attention[0] >= attention[1] else "frequency"
        branch_line = f"The {dominant} branch contributed slightly more to this decision."

    st.markdown(
        f"""
        <div class="explain-card">
            <h3>How To Explain This Result</h3>
            <p class="section-note">
                The detector first isolates the face, then checks the same content in two ways:
                normal visual appearance and frequency-domain behavior. Real media usually has a
                smoother spectral decay and more natural sensor/compression traces. Manipulated media
                often leaves inconsistent high-frequency patterns or unusual concentration in parts of the spectrum.
            </p>
            <p class="section-note">
                Final label: <strong>{label.upper()}</strong>. Confidence for this decision is approximately
                <strong>{confidence:.2%}</strong>. {branch_line}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def analyze_media(
    temp_path: Path,
    media_type: str,
    config: dict,
    map_config: FrequencyMapConfig,
    detector,
    model,
    frame_stride: int | None = None,
    max_frames: int | None = None,
) -> dict:
    margin = config["preprocessing"].get("face_margin", 24)
    if media_type == "image":
        frame = load_media_frame(temp_path)
        detection = detect_largest_face_context(frame, detector, margin)
        if detection is None:
            raise ValueError("No face detected in the uploaded image.")
        face, frame_annotated = detection
        probability, attention = score_face(model, face, map_config)
        artifacts = build_frequency_artifacts(face, map_config)
        artifacts["frame_bgr"] = frame_annotated
        frames_used = 1
        face_frames = [face]
        frame_scores = [probability]
    else:
        scores = []
        attentions = []
        artifacts_list = []
        face_frames = []
        frame_indices = []
        used_stride = frame_stride or config["preprocessing"].get("frame_stride", 10)
        used_max_frames = max_frames or config["preprocessing"].get("frames_per_video", 12)
        for frame in iter_video_frames(
            temp_path,
            frame_stride=used_stride,
            max_frames=used_max_frames,
        ):
            detection = detect_largest_face_context(frame, detector, margin)
            if detection is None:
                continue
            face, frame_annotated = detection
            probability_frame, attention_frame = score_face(model, face, map_config)
            scores.append(probability_frame)
            face_frames.append(face)
            frame_artifacts = build_frequency_artifacts(face, map_config)
            frame_artifacts["frame_bgr"] = frame_annotated
            artifacts_list.append(frame_artifacts)
            frame_indices.append(len(frame_indices))
            if attention_frame is not None:
                attentions.append(attention_frame)

        if not scores or not artifacts_list:
            raise ValueError("No faces were detected in the sampled video frames.")

        probability = float(np.mean(scores))
        attention = np.mean(np.stack(attentions, axis=0), axis=0) if attentions else None
        artifacts = aggregate_artifacts(artifacts_list)
        frames_used = len(scores)
        frame_scores = list(scores)

    return {
        "probability": probability,
        "attention": attention,
        "artifacts": artifacts,
        "frames_used": frames_used,
        "face_frames": face_frames,
        "frame_scores": frame_scores,
        "frame_indices": frame_indices if media_type == "video" else [0],
    }


def render_single_result(result: dict, threshold: float) -> None:
    probability = result["probability"]
    attention = result["attention"]
    artifacts = result["artifacts"]
    frames_used = result["frames_used"]
    face_frames = result.get("face_frames", [])
    frame_scores = result.get("frame_scores", [])
    label, threshold_note = resolve_label(probability, threshold)
    confidence = probability if label == "fake" else 1.0 - probability

    result_col_1, result_col_2, result_col_3 = st.columns(3)
    with result_col_1:
        render_metric_card("Prediction", label.upper(), "Model decision on the uploaded media", is_status=True)
    with result_col_2:
        render_metric_card("Fake Probability", f"{probability:.4f}", "Sigmoid score after fusion")
    with result_col_3:
        render_metric_card("Frames Used", str(frames_used), "Sampled frames analyzed from the file")

    st.progress(min(max(float(confidence), 0.0), 1.0), text=f"Decision confidence: {confidence:.2%}")
    st.caption(threshold_note)

    tab_overview, tab_visuals, tab_reasoning, tab_project = st.tabs(
        ["Overview", "Frequency Views", "Decision Reasoning", "Project Edge"]
    )

    with tab_overview:
        overview_left, overview_right = st.columns([1.25, 1])
        with overview_left:
            render_explanation(label, probability, attention)
        with overview_right:
            st.markdown(
                """
                <div class="info-card">
                    <h3>Pipeline</h3>
                    <p class="section-note">
                        Input media -> face detection -> spatial tensor + FFT map + DCT map ->
                        dual-branch network -> attention fusion -> real/fake prediction.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="info-card" style="margin-top: 1rem;">
                    <h3>Why A Deepfake Can Still Look Real</h3>
                    <p class="section-note">
                        Some manipulations are subtle, heavily compressed, or visually clean enough that the
                        model's score stays below the fake threshold. In practice, threshold choice, dataset bias,
                        and weak artifacts can all cause false negatives.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_visuals:
        render_frame_strip(face_frames)
        render_frame_score_chart(frame_scores)
        render_visual_grid(artifacts)

    with tab_reasoning:
        render_attention(attention)
        st.markdown(
            """
            <div class="info-card">
                <h3>How To Reduce False Negatives</h3>
                <p class="section-note">
                    Lower the decision threshold slightly, test on harder external datasets, and compare branch behavior.
                    If a deepfake is being called real, it often means the fake probability is close to the threshold rather
                    than strongly natural.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_project:
        st.markdown(
            """
            <div class="info-card">
                <h3>Why This Project Stands Out</h3>
                <p class="section-note">
                    This is not a plain image classifier. The system combines spatial evidence with two frequency
                    transforms, uses attention-based fusion, produces explainable visuals, and supports ablation-based
                    analysis to show what each branch contributes.
                </p>
                <p class="section-note">
                    Strong presentation angle: frequency-only branches are weaker on their own, but fusion performs
                    best. That proves the project is solving the task in a more thoughtful way than a single-input model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_comparison_card(title: str, result: dict, threshold: float) -> None:
    probability = result["probability"]
    attention = result["attention"]
    label, threshold_note = resolve_label(probability, threshold)
    confidence = probability if label == "fake" else 1.0 - probability
    st.markdown(f"### {title}")
    render_metric_card("Prediction", label.upper(), threshold_note, is_status=True)
    metric_col_1, metric_col_2 = st.columns(2)
    with metric_col_1:
        st.metric("Fake Probability", f"{probability:.4f}")
    with metric_col_2:
        st.metric("Confidence", f"{confidence:.2%}")
    render_frame_strip(result.get("face_frames", []))
    render_frame_score_chart(result.get("frame_scores", []))
    render_visual_grid(result["artifacts"])
    render_attention(attention)


def main() -> None:
    st.set_page_config(
        page_title="Deepfake Frequency Fusion Demo",
        page_icon=":camera:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    render_hero()

    st.sidebar.header("Demo Controls")
    config_path = st.sidebar.text_input("Config Path", "configs/frequency_cnn.yaml")
    media_type = st.sidebar.selectbox("Input Type", ["image", "video"])
    mode = st.sidebar.selectbox("Mode", ["single analysis", "comparison"])
    st.sidebar.markdown(
        """
        **Model summary**

        - Spatial RGB branch
        - FFT frequency branch
        - DCT frequency branch
        - Attention fusion
        """
    )
    config, checkpoint, model, map_config, detector = load_model_bundle(config_path)
    threshold_default = float(checkpoint.get("threshold", 0.5))
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.20,
        max_value=0.80,
        value=float(threshold_default),
        step=0.01,
        help="Lower values catch more suspicious samples but may increase false positives.",
    )
    frame_budget = st.sidebar.slider(
        "Frames Per Video",
        min_value=4,
        max_value=24,
        value=int(config["preprocessing"].get("frames_per_video", 12)),
        step=2,
        help="Maximum number of sampled video frames to analyze.",
    )
    frame_stride = st.sidebar.slider(
        "Frame Stride",
        min_value=1,
        max_value=20,
        value=int(config["preprocessing"].get("frame_stride", 10)),
        step=1,
        help="Analyze every Nth frame in the video.",
    )
    st.sidebar.caption("If a deepfake is being predicted as real, try a slightly lower threshold and compare its frequency profile.")

    uploaded = None
    uploaded_a = None
    uploaded_b = None
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    if mode == "single analysis":
        uploaded = st.file_uploader(
            "Upload an image or video",
            type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
            help="Use a face-focused image or a short video clip for the cleanest demo.",
            key="single_upload",
        )
    else:
        compare_col_1, compare_col_2 = st.columns(2)
        with compare_col_1:
            uploaded_a = st.file_uploader(
                "Upload sample A",
                type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
                key="compare_upload_a",
            )
        with compare_col_2:
            uploaded_b = st.file_uploader(
                "Upload sample B",
                type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
                key="compare_upload_b",
            )
    st.markdown("</div>", unsafe_allow_html=True)

    if mode == "single analysis" and not uploaded:
        info_left, info_right = st.columns([1.2, 1])
        with info_left:
            st.markdown(
                """
                <div class="info-card">
                    <h3>What This Demo Shows</h3>
                    <p class="section-note">
                        Upload media and the webpage will detect the main face, compute FFT and DCT maps,
                        run the attention-fusion detector, and explain whether the content appears real or fake.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with info_right:
            st.markdown(
                """
                <div class="info-card">
                    <h3>Best Demo Flow</h3>
                    <p class="section-note">
                        1. Upload a face image.<br>
                        2. Show the FFT and DCT panels.<br>
                        3. Point out branch attention.<br>
                        4. Explain why frequency cues help catch synthetic artifacts.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return
    if mode == "comparison" and (uploaded_a is None or uploaded_b is None):
        st.info("Upload two samples to compare how the model scores them and how their frequency behavior differs.")
        return

    if mode == "single analysis":
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            temp_path = Path(tmp.name)
        with st.spinner("Analyzing media across spatial and frequency domains..."):
            try:
                result = analyze_media(
                    temp_path,
                    media_type,
                    config,
                    map_config,
                    detector,
                    model,
                    frame_stride=frame_stride,
                    max_frames=frame_budget,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
        render_single_result(result, threshold)
    else:
        results = []
        for uploaded_file in (uploaded_a, uploaded_b):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = Path(tmp.name)
            try:
                result = analyze_media(
                    temp_path,
                    media_type,
                    config,
                    map_config,
                    detector,
                    model,
                    frame_stride=frame_stride,
                    max_frames=frame_budget,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            results.append(result)

        score_gap = abs(results[0]["probability"] - results[1]["probability"])
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Comparison Summary</h3>
                <p class="section-note">
                    Score gap between the two samples: <strong>{score_gap:.4f}</strong>.
                    This is useful when one sample is being predicted as real even though it feels suspicious.
                    Side-by-side comparison often reveals that the "real" prediction is only slightly under the fake threshold.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        compare_left, compare_right = st.columns(2)
        with compare_left:
            render_comparison_card("Sample A", results[0], threshold)
        with compare_right:
            render_comparison_card("Sample B", results[1], threshold)


if __name__ == "__main__":
    main()
