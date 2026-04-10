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
    from src.liveness_v2_core import (
        ClassificationConfig,
        ROIConfig,
        SignalConfig,
        analyze_video_for_liveness_v2,
        build_face_mesh,
    )
else:
    from .config import load_config
    from .liveness_v2_core import (
        ClassificationConfig,
        ROIConfig,
        SignalConfig,
        analyze_video_for_liveness_v2,
        build_face_mesh,
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --sun-red: #F3000E;
            --daybreak-orange: #F25016;
            --sapphire: #6596F3;
            --emerald: #83B366;
            --orchid: #D3A4EA;
            --daybreak-gold: #EAD094;
            --mist: #B2DCE2;
            --soft-leaf: #D7EAAC;
            --ink: #152b33;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(101, 150, 243, 0.22), transparent 30%),
                radial-gradient(circle at top right, rgba(234, 208, 148, 0.34), transparent 30%),
                radial-gradient(circle at bottom left, rgba(131, 179, 102, 0.18), transparent 32%),
                linear-gradient(180deg, #f5fbfc 0%, #edf7f8 44%, #fff7e3 100%);
            color: var(--ink);
        }
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero, .card, .metric-card {
            background: rgba(255, 255, 251, 0.92);
            border: 1px solid rgba(101, 150, 243, 0.18);
            box-shadow: 0 18px 42px rgba(36, 70, 90, 0.10);
            border-radius: 22px;
            padding: 1.2rem 1.3rem;
        }
        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1.08;
            margin-bottom: 0.6rem;
            color: var(--ink);
        }
        .hero-kicker {
            color: var(--sapphire);
            font-size: 0.84rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .note {
            color: #435c65;
            line-height: 1.62;
            font-size: 0.98rem;
        }
        .metric-label {
            font-size: 0.82rem;
            font-weight: 700;
            color: #4f7280;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--ink);
        }
        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.28rem 0.72rem;
            font-size: 0.82rem;
            font-weight: 700;
            background: rgba(178, 220, 226, 0.42);
            color: #3a6f78;
            margin-right: 0.4rem;
            margin-top: 0.4rem;
        }
        .warning-pill {
            background: rgba(242, 80, 22, 0.12);
            color: var(--daybreak-orange);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_liveness_v2_config(config_path: str):
    config = load_config(config_path)
    roi_config = ROIConfig(**config["roi"])
    signal_config = SignalConfig(**config["signal"])
    classification_config = ClassificationConfig(**config["classification"])
    return config, roi_config, signal_config, classification_config


def render_metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="note">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary(result: dict[str, object]) -> None:
    status = str(result["status"])
    cols = st.columns(5)
    with cols[0]:
        render_metric_card("Status", "AUTHENTICATED" if status == "authenticated" else "SYNTHETIC / INCONCLUSIVE", "Final liveness verdict")
    with cols[1]:
        render_metric_card("Estimated BPM", f"{float(result.get('peak_bpm', 0.0)):.1f}" if result.get("peak_bpm") is not None else "N/A", "Dominant physiological peak")
    with cols[2]:
        render_metric_card("Pulse Score", f"{float(result.get('live_score', 0.0)):.2f}", "Strength of the biological rhythm")
    with cols[3]:
        render_metric_card("Motion Score", f"{float(result.get('motion_quality', 0.0)):.2f}", "Stability of face tracking")
    with cols[4]:
        render_metric_card("Signal Grade", qualitative_level(float(result.get("live_score", 0.0))), "Audience-friendly evidence level")


def qualitative_level(value: float) -> str:
    if value >= 0.72:
        return "Strong"
    if value >= 0.50:
        return "Moderate"
    return "Weak"


def render_frames(frames: list[np.ndarray]) -> None:
    if not frames:
        return
    st.markdown("### Dynamic ROI Tracking")
    cols = st.columns(min(len(frames), 5))
    for idx, (col, frame) in enumerate(zip(cols, frames[:5], strict=False), start=1):
        col.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Preview {idx}", use_container_width=True)


def render_frame_inspector(result: dict[str, object]) -> None:
    frames = result.get("annotated_frames", [])
    green_frames = result.get("green_heatmap_frames", [])
    frequency_frames = result.get("frequency_heatmap_frames", [])
    frame_indices = result.get("annotated_frame_indices", [])
    if not frames:
        return

    st.markdown("### Frame-by-Frame ROI Inspector")
    if len(frames) == 1:
        selected = 1
        st.caption("Only one tracked ROI frame is available for inspection.")
    else:
        selected = st.slider(
            "Move through tracked ROI frames",
            min_value=1,
            max_value=len(frames),
            value=1,
            step=1,
        )
    frame = frames[selected - 1]
    green_frame = green_frames[selected - 1] if selected - 1 < len(green_frames) else frame
    frequency_frame = frequency_frames[selected - 1] if selected - 1 < len(frequency_frames) else frame
    original_index = frame_indices[selected - 1] if selected - 1 < len(frame_indices) else selected
    view = st.radio(
        "Overlay view",
        ["ROI outline", "Green intensity on face", "Frequency evidence on face"],
        horizontal=True,
    )
    selected_frame = {
        "ROI outline": frame,
        "Green intensity on face": green_frame,
        "Frequency evidence on face": frequency_frame,
    }[view]
    st.image(
        cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB),
        caption=f"{view} | Tracked frame {selected} | Original video frame {original_index}",
        use_container_width=True,
    )


def render_roi_traces(roi_green_traces: dict[str, np.ndarray]) -> None:
    if not roi_green_traces:
        return
    fig, ax = plt.subplots(figsize=(9, 3.3))
    for name, values in roi_green_traces.items():
        ax.plot(values, linewidth=1.5, label=name.replace("_", " ").title())
    ax.set_title("ROI Green-Channel Temporal Signals")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Normalized Intensity")
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.patch.set_facecolor("#fbf8f3")
    ax.set_facecolor("#fbf8f3")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_green_dashboard(result: dict[str, object]) -> None:
    st.markdown("### Green-Channel Dashboard")
    col1, col2 = st.columns([1.2, 1.0])
    with col1:
        render_roi_traces(result.get("roi_green_traces", {}))
    with col2:
        green_trace = np.asarray(result.get("green_trace", []), dtype=np.float32)
        if green_trace.size:
            fig, ax = plt.subplots(figsize=(5.4, 3.3))
            ax.plot(green_trace, color="#83B366", linewidth=2.0)
            ax.fill_between(np.arange(green_trace.size), green_trace, color="#83B366", alpha=0.16)
            ax.set_title("Combined Green Trace")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Intensity")
            ax.grid(alpha=0.22, linestyle="--")
            fig.patch.set_facecolor("#fffdf4")
            ax.set_facecolor("#fffdf4")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def render_frequency_dashboard(result: dict[str, object]) -> None:
    st.markdown("### Frequency Evidence Dashboard")
    components = result.get("components", {})
    if not components:
        return

    primary_key = str(result.get("primary_component", "combined"))
    primary = components.get(primary_key) or components.get("combined")
    col1, col2 = st.columns([1.2, 1.0])
    with col1:
        render_component_spectra(result)
    with col2:
        if primary is None:
            return
        freqs = np.asarray(primary["band_freqs"], dtype=np.float32)
        power = np.asarray(primary["band_power"], dtype=np.float32)
        if freqs.size and power.size:
            peak_idx = int(np.argmax(power))
            fig, ax = plt.subplots(figsize=(5.4, 3.3))
            colors = ["#83B366" if idx == peak_idx else "#B2DCE2" for idx in range(freqs.size)]
            ax.bar(freqs, power, width=0.035, color=colors, edgecolor="none")
            ax.set_title(f"Dominant Spectrum: {primary['name']}")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power")
            ax.grid(alpha=0.18, linestyle="--", axis="y")
            fig.patch.set_facecolor("#fffdf4")
            ax.set_facecolor("#fffdf4")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def render_component_signals(result: dict[str, object]) -> None:
    components = result.get("components", {})
    if not components:
        return

    fig, ax = plt.subplots(figsize=(9, 3.3))
    for key, color in [("green", "#83B366"), ("pos", "#6596F3"), ("chrom", "#D3A4EA"), ("ica", "#F25016"), ("combined", "#152b33")]:
        component = components.get(key)
        if component is None:
            continue
        signal = np.asarray(component["filtered_signal"], dtype=np.float32)
        if signal.size == 0:
            continue
        ax.plot(signal, linewidth=1.8 if key == "combined" else 1.25, label=component["name"], color=color, alpha=0.95)
    ax.set_title("Processed Pulse Signals")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(frameon=False)
    fig.patch.set_facecolor("#fffdf4")
    ax.set_facecolor("#fffdf4")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_component_spectra(result: dict[str, object]) -> None:
    components = result.get("components", {})
    if not components:
        return

    fig, ax = plt.subplots(figsize=(9, 3.3))
    for key, color in [("green", "#83B366"), ("pos", "#6596F3"), ("chrom", "#D3A4EA"), ("ica", "#F25016"), ("combined", "#152b33")]:
        component = components.get(key)
        if component is None:
            continue
        freqs = np.asarray(component["band_freqs"], dtype=np.float32)
        power = np.asarray(component["band_power"], dtype=np.float32)
        if freqs.size == 0 or power.size == 0:
            continue
        ax.plot(freqs, power, linewidth=2.0 if key == "combined" else 1.35, label=component["name"], color=color, alpha=0.95)
    ax.set_xlim(0.65, 2.55)
    ax.axvspan(1.0, 1.5, color="#83B366", alpha=0.14)
    ax.set_title("FFT Spectrum Inside Physiological Band")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(frameon=False)
    fig.patch.set_facecolor("#fffdf4")
    ax.set_facecolor("#fffdf4")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_component_table(result: dict[str, object]) -> None:
    components = result.get("components", {})
    if not components:
        return

    st.markdown("### Component Quality")
    primary = str(result.get("primary_component", "combined"))
    for key in ("green", "pos", "chrom", "ica", "combined"):
        component = components.get(key)
        if component is None:
            continue
        st.markdown(
            f"""
            <div class="card">
                <strong>{component['name']}</strong>
                {'<span class="pill">Primary decision signal</span>' if key == primary else ''}
                <div class="note">
                    Peak BPM: <strong>{float(component['peak_bpm']):.1f}</strong> |
                    Pulse Score: <strong>{float(component['live_score']):.2f}</strong> |
                    Evidence Grade: <strong>{qualitative_level(float(component['live_score']))}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_motion_quality(result: dict[str, object]) -> None:
    st.markdown("### Motion Quality")
    quality = float(result.get("motion_quality", 0.0))
    mean_motion = float(result.get("mean_motion_px", 0.0))
    jitter = float(result.get("motion_jitter_px", 0.0))
    st.markdown(
        f"""
        <div class="card">
            <strong>Face Motion Stability</strong>
            <div class="note">
                Motion score: <strong>{quality:.2f}</strong> |
                Mean frame movement: <strong>{mean_motion:.2f}px</strong> |
                Motion jitter: <strong>{jitter:.2f}px</strong>
            </div>
            <div class="note">
                Higher is better. Strong head motion can create fake pulse-like brightness changes, so this score
                makes authentication stricter when tracking is unstable.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_roi_quality(result: dict[str, object]) -> None:
    roi_quality = result.get("roi_quality", {})
    roi_weights = result.get("roi_weights", {})
    if not roi_quality:
        return
    st.markdown("### ROI Quality Weighting")
    for name, quality in roi_quality.items():
        weight = float(roi_weights.get(name, 0.0))
        st.markdown(
            f"""
            <div class="card">
                <strong>{name.replace('_', ' ').title()}</strong>
                <div class="note">
                    Region pulse score: <strong>{float(quality):.2f}</strong> |
                    Contribution: <strong>masked</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def analyze_uploaded_video(uploaded, config, roi_config, signal_config, classification_config) -> dict[str, object]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        temp_path = Path(tmp.name)

    with build_face_mesh(**config["face_mesh"]) as face_mesh:
        return analyze_video_for_liveness_v2(
            temp_path,
            face_mesh,
            roi_config,
            signal_config,
            classification_config,
            fps_hint=float(config["input"].get("fps_hint", 30.0)),
            max_seconds=int(config["input"].get("max_seconds", 20)),
            inspect_frame_limit=int(config["input"].get("inspect_frame_limit", 120)),
        )


def render_calibration_mode(config_path: str) -> None:
    config, roi_config, signal_config, classification_config = load_liveness_v2_config(config_path)
    st.markdown(
        """
        <div class="card">
            <h3>Calibration Mode</h3>
            <p class="note">
                Upload a few known real clips and known synthetic clips. The app will compare their pulse evidence,
                pulse evidence, BPM, and motion quality, then recommend a safer operating profile for your camera and lighting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    real_uploads = st.file_uploader("Known real videos", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True, key="cal_real")
    fake_uploads = st.file_uploader("Known synthetic/deepfake videos", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True, key="cal_fake")
    if not st.button("Run Calibration", type="primary"):
        return
    if not real_uploads or not fake_uploads:
        st.warning("Upload at least one known real and one known synthetic video.")
        return

    rows = []
    with st.spinner("Calibrating the operating profile from uploaded clips..."):
        for label, uploads in [("real", real_uploads), ("synthetic", fake_uploads)]:
            for uploaded in uploads:
                result = analyze_uploaded_video(uploaded, config, roi_config, signal_config, classification_config)
                rows.append(
                    {
                        "name": uploaded.name,
                        "label": label,
                        "status": result.get("status", "n/a"),
                        "bpm": float(result.get("peak_bpm", 0.0) or 0.0),
                        "pulse_score": round(float(result.get("live_score", 0.0) or 0.0), 3),
                        "motion_score": round(float(result.get("motion_quality", 0.0) or 0.0), 3),
                        "pulse_grade": qualitative_level(float(result.get("live_score", 0.0) or 0.0)),
                        "valid_frames": int(result.get("valid_face_frames", 0) or 0),
                        "_snr": float(result.get("snr", 0.0) or 0.0),
                        "_live_score": float(result.get("live_score", 0.0) or 0.0),
                        "_motion_quality": float(result.get("motion_quality", 0.0) or 0.0),
                    }
                )

    visible_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    st.dataframe(visible_rows, use_container_width=True)
    real_rows = [row for row in rows if row["label"] == "real"]
    fake_rows = [row for row in rows if row["label"] == "synthetic"]
    if not real_rows or not fake_rows:
        return

    min_real_snr = min(row["_snr"] for row in real_rows)
    max_fake_snr = max(row["_snr"] for row in fake_rows)
    min_real_score = min(row["_live_score"] for row in real_rows)
    max_fake_score = max(row["_live_score"] for row in fake_rows)
    min_real_motion = min(row["_motion_quality"] for row in real_rows)

    suggested_snr = max(2.8, (min_real_snr + max_fake_snr) / 2.0)
    suggested_score = max(0.60, (min_real_score + max_fake_score) / 2.0)
    suggested_motion = max(0.25, min(0.55, min_real_motion - 0.05))

    st.markdown(
        f"""
        <div class="card">
            <h3>Recommended Operating Profile</h3>
            <p class="note">
                Camera/lighting profile: <strong>{'Strict' if suggested_score >= 0.68 else 'Balanced'}</strong><br>
                Pulse separation target: <strong>{qualitative_level(suggested_score)}</strong><br>
                Motion tolerance target: <strong>{qualitative_level(suggested_motion)}</strong><br>
                Heart-rate gate: <strong>normal adult range enabled</strong>
            </p>
            <p class="note">
                Use this profile to tune the hidden config after testing on your own known real and synthetic clips.
                The audience sees the evidence quality, not the exact internal decision settings.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Biological Liveness Detection v2", layout="wide")
    inject_styles()
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Biological Liveness Detection v2</div>
            <div class="hero-title">Dense ROI rPPG With Green, POS, ICA, and FFT</div>
            <p class="note">
                This upgraded copy follows your presentation workflow closely: 468-point face mapping,
                denser forehead and upper-cheek ROIs, green-channel temporal averaging, POS and ICA
                signal separation, bandpass filtering, FFT peak analysis, and final liveness classification.
            </p>
            <div class="pill">Separate copy from the current model</div>
            <div class="pill">More ROI points</div>
            <div class="pill">Better low-brightness robustness</div>
            <div class="pill">Emerald Sapphire Daybreak palette</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config_path = st.sidebar.text_input("Config Path", "configs/liveness_v2.yaml")
    mode = st.sidebar.radio("Mode", ["Analyze Video", "Calibration Mode"])
    if mode == "Calibration Mode":
        render_calibration_mode(config_path)
        return

    uploaded = st.file_uploader("Upload a face video", type=["mp4", "avi", "mov", "mkv"])

    if not uploaded:
        st.markdown(
            """
            <div class="card">
                <h3>System Pipeline</h3>
                <p class="note">
                    Face Detection and 3D mesh mapping -> Dynamic ROI extraction -> Green-channel temporal averaging ->
                    POS, CHROM, and ICA signal separation -> Detrending and bandpass filtering -> FFT peak detection ->
                    liveness decision.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    config, roi_config, signal_config, classification_config = load_liveness_v2_config(config_path)
    with st.spinner("Running dense ROI pulse analysis..."):
        result = analyze_uploaded_video(uploaded, config, roi_config, signal_config, classification_config)

    render_summary(result)
    st.caption(
        f"Frames read: {result.get('total_frames', 0)} | Valid face frames: {result.get('valid_face_frames', 0)} | Primary signal: {str(result.get('primary_component', 'n/a')).upper()}"
    )

    if result["status"] == "insufficient_data":
        st.warning(str(result.get("reason", "Not enough usable pulse information.")))
        return

    st.markdown(
        f"""
        <div class="card">
            <h3>Why This Version Is More Robust</h3>
            <p class="note">
                This copy does not rely on raw green brightness alone. It first normalizes RGB traces,
                then compares four physiological candidates: the normalized green signal, POS, CHROM, and ICA.
                That makes the system less likely to reject darker-skin videos simply because the absolute
                green intensity is lower. The final decision uses relative spectral quality, support from
                multiple signal branches, ROI quality weighting, and adaptive low-brightness handling.
            </p>
            <p class="note">
                Dominant pulse estimate: <strong>{float(result.get('peak_bpm', 0.0)):.1f} BPM</strong>.
                Pulse score: <strong>{float(result.get('live_score', 0.0)):.2f}</strong>.
                Motion score: <strong>{float(result.get('motion_quality', 0.0)):.2f}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["ROI Tracking", "Signals", "FFT Analysis", "Decision Logic"])
    with tabs[0]:
        render_frames(result.get("annotated_frames", []))
        render_frame_inspector(result)
        render_green_dashboard(result)
    with tabs[1]:
        render_component_signals(result)
    with tabs[2]:
        render_frequency_dashboard(result)
    with tabs[3]:
        render_component_table(result)
        render_motion_quality(result)
        render_roi_quality(result)


if __name__ == "__main__":
    main()
