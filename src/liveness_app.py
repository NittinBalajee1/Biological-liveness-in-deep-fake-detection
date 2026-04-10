from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import load_config
    from src.liveness_core import (
        ClassificationConfig,
        ROIConfig,
        SignalConfig,
        analyze_video_for_liveness,
        build_face_mesh,
    )
else:
    from .config import load_config
    from .liveness_core import (
        ClassificationConfig,
        ROIConfig,
        SignalConfig,
        analyze_video_for_liveness,
        build_face_mesh,
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(43, 136, 121, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(219, 114, 67, 0.15), transparent 28%),
                linear-gradient(180deg, #eef5f4 0%, #f7f2ea 50%, #f4eee7 100%);
            color: #173038;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero, .card {
            background: rgba(255, 252, 247, 0.88);
            border: 1px solid rgba(23, 48, 56, 0.08);
            box-shadow: 0 18px 38px rgba(26, 44, 52, 0.08);
            border-radius: 22px;
            padding: 1.25rem 1.35rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.65rem;
            color: #12323a;
        }
        .hero-kicker {
            color: #1d7b6b;
            font-size: 0.84rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .note {
            color: #49616a;
            line-height: 1.65;
            font-size: 0.98rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_liveness_config(config_path: str):
    config = load_config(config_path)
    roi_config = ROIConfig(**config["roi"])
    signal_config = SignalConfig(**config["signal"])
    classification_config = ClassificationConfig(**config["classification"])
    return config, roi_config, signal_config, classification_config


def render_summary(result: dict[str, object]) -> None:
    status = str(result["status"])
    bpm = result.get("peak_bpm")
    snr = result.get("snr")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", "AUTHENTICATED" if status == "authenticated" else "SYNTHETIC / INCONCLUSIVE")
    with col2:
        st.metric("Estimated BPM", f"{bpm:.1f}" if bpm is not None else "N/A")
    with col3:
        st.metric("Pulse SNR", f"{snr:.2f}" if snr is not None else "N/A")


def render_frames(frames: list[np.ndarray]) -> None:
    if not frames:
        return
    st.markdown("### ROI Tracking")
    cols = st.columns(len(frames))
    for idx, (col, frame) in enumerate(zip(cols, frames, strict=False), start=1):
        col.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {idx}", use_container_width=True)


def render_signals(result: dict[str, object]) -> None:
    raw_signal = np.asarray(result.get("raw_signal", []), dtype=np.float32)
    filtered_signal = np.asarray(result.get("filtered_signal", []), dtype=np.float32)
    freqs = np.asarray(result.get("freqs", []), dtype=np.float32)
    power = np.asarray(result.get("power", []), dtype=np.float32)
    if raw_signal.size == 0:
        return

    st.markdown("### Pulse Signal")
    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.plot(raw_signal, color="#7a8d95", linewidth=1.6, label="Raw green signal")
    if filtered_signal.size:
        ax.plot(filtered_signal, color="#1d7b6b", linewidth=2.0, label="Filtered pulse")
    ax.set_title("Remote PPG Signal Over Time")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Intensity")
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(frameon=False)
    fig.patch.set_facecolor("#fbf8f3")
    ax.set_facecolor("#fbf8f3")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if freqs.size and power.size:
        st.markdown("### Frequency Spectrum")
        fig, ax = plt.subplots(figsize=(9, 3.1))
        ax.plot(freqs, power, color="#d56a3b", linewidth=2.0)
        peak_freq = result.get("peak_freq_hz")
        if peak_freq is not None:
            ax.axvline(float(peak_freq), color="#1d7b6b", linestyle="--", linewidth=1.6)
        ax.set_xlim(0.0, 3.2)
        ax.set_title("FFT Power Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.grid(alpha=0.22, linestyle="--")
        fig.patch.set_facecolor("#fbf8f3")
        ax.set_facecolor("#fbf8f3")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="Biological Liveness Detection", layout="wide")
    inject_styles()
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Biological Liveness Detection</div>
            <div class="hero-title">Remote PPG From Facial Skin Regions</div>
            <p class="note">
                This module tracks forehead and cheek regions, extracts the green-channel pulse signal,
                filters it into the physiological heart-rate band, and uses FFT peak strength to decide
                whether the media shows a plausible biological pulse.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config_path = st.sidebar.text_input("Config Path", "configs/liveness.yaml")
    uploaded = st.file_uploader("Upload a face video", type=["mp4", "avi", "mov", "mkv"])
    if not uploaded:
        st.markdown(
            """
            <div class="card">
                <h3>How This Works</h3>
                <p class="note">
                    1. Track the face with MediaPipe Face Mesh.<br>
                    2. Extract forehead and cheek ROIs.<br>
                    3. Build a green-channel temporal signal.<br>
                    4. Bandpass filter it to human pulse frequencies.<br>
                    5. Use FFT to measure periodic biological energy.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    config, roi_config, signal_config, classification_config = load_liveness_config(config_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        temp_path = Path(tmp.name)

    with st.spinner("Tracking facial skin regions and extracting pulse signal..."):
        with build_face_mesh(**config["face_mesh"]) as face_mesh:
            result = analyze_video_for_liveness(
                temp_path,
                face_mesh,
                roi_config,
                signal_config,
                classification_config,
                fps_hint=float(config["input"].get("fps_hint", 30.0)),
                max_seconds=int(config["input"].get("max_seconds", 20)),
            )

    render_summary(result)
    st.caption(
        f"Frames read: {result.get('total_frames', 0)} | Valid face frames: {result.get('valid_face_frames', 0)}"
    )
    if result["status"] == "insufficient_data":
        st.warning(str(result.get("reason", "Not enough usable pulse information.")))
    else:
        st.markdown(
            f"""
            <div class="card">
                <h3>Decision Logic</h3>
                <p class="note">
                    Estimated pulse peak: <strong>{result['peak_freq_hz']:.3f} Hz</strong>
                    ({result['peak_bpm']:.1f} BPM). Signal-to-noise ratio:
                    <strong>{result['snr']:.2f}</strong>. A strong peak in the
                    physiological band supports <strong>{result['status'].replace('_', ' ').upper()}</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    render_frames(result.get("annotated_frames", []))
    render_signals(result)


if __name__ == "__main__":
    main()
