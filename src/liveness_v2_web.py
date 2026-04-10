from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, send_from_directory, url_for

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


ROOT = Path(__file__).resolve().parent.parent
WEB_OUTPUT = ROOT / "outputs" / "liveness_v2_web"
UPLOAD_DIR = WEB_OUTPUT / "uploads"
RENDER_DIR = WEB_OUTPUT / "renders"
TEMPLATE_DIR = ROOT / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))


def qualitative_level(value: float) -> str:
    if value >= 0.72:
        return "Strong"
    if value >= 0.50:
        return "Moderate"
    return "Weak"


def safe_status(status: str) -> str:
    return "AUTHENTICATED" if status == "authenticated" else "SYNTHETIC / INCONCLUSIVE"


def load_bundle(config_path: str):
    config = load_config(config_path)
    roi_config = ROIConfig(**config["roi"])
    signal_config = SignalConfig(**config["signal"])
    classification_config = ClassificationConfig(**config["classification"])
    return config, roi_config, signal_config, classification_config


def write_video(frames: list[np.ndarray], output_path: Path, fps: float) -> bool:
    if not frames:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(8.0, min(float(fps), 30.0)),
        (width, height),
    )
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def write_frame_sequence(frames: list[np.ndarray], output_dir: Path, max_width: int = 960) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = []
    if not frames:
        return urls
    for index, frame in enumerate(frames):
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / float(width)
            frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
        filename = f"frame_{index:04d}.jpg"
        cv2.imwrite(str(output_dir / filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        urls.append(filename)
    return urls


def save_plot(fig, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path.name


def style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#152B33", pad=12)
    ax.set_xlabel(xlabel, color="#60737A", labelpad=8)
    ax.set_ylabel(ylabel, color="#60737A", labelpad=8)
    ax.grid(alpha=0.18, linestyle="-", color="#9EB9C0")
    ax.tick_params(colors="#60737A", labelsize=8)
    ax.set_facecolor("#FFFDF4")
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_plot_signature(fig, text: str) -> None:
    fig.text(
        0.015,
        0.018,
        text,
        color="#789096",
        fontsize=7,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def build_graphs(result: dict[str, object], output_dir: Path) -> dict[str, str]:
    graphs: dict[str, str] = {}
    roi_green_traces = result.get("roi_green_traces", {})
    if roi_green_traces:
        fig, ax = plt.subplots(figsize=(8.4, 3.35))
        palette = ["#83B366", "#6596F3", "#D3A4EA", "#F25016", "#B2DCE2", "#D7EAAC", "#EAD094"]
        for name, values in roi_green_traces.items():
            color = palette[len(ax.lines) % len(palette)]
            values = np.asarray(values, dtype=np.float32)
            ax.plot(values, linewidth=2.0, label=name.replace("_", " ").title(), color=color, alpha=0.92)
            ax.fill_between(np.arange(values.size), values, values.min(), color=color, alpha=0.045)
        style_axis(ax, "ROI Green-Channel Signals", "Frame", "Green intensity")
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
        fig.patch.set_facecolor("#FFFDF4")
        add_plot_signature(fig, "DYNAMIC ROI TEMPORAL TRACE")
        graphs["green"] = save_plot(fig, output_dir / "green_signals.png")

    components = result.get("components", {})
    if components:
        fig, ax = plt.subplots(figsize=(8.4, 3.35))
        colors = {"green": "#83B366", "pos": "#6596F3", "chrom": "#D3A4EA", "ica": "#F25016", "combined": "#152B33"}
        for key in ("green", "pos", "chrom", "ica", "combined"):
            component = components.get(key)
            if not component:
                continue
            signal = np.asarray(component["filtered_signal"], dtype=np.float32)
            if signal.size:
                ax.plot(
                    signal,
                    linewidth=2.6 if key == "combined" else 1.35,
                    label=component["name"],
                    color=colors[key],
                    alpha=0.96 if key == "combined" else 0.72,
                )
        style_axis(ax, "Processed Pulse Signals", "Frame", "Amplitude")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.patch.set_facecolor("#FFFDF4")
        add_plot_signature(fig, "GREEN + POS + CHROM + ICA FUSION")
        graphs["signals"] = save_plot(fig, output_dir / "processed_signals.png")

        fig, ax = plt.subplots(figsize=(8.4, 3.35))
        for key in ("green", "pos", "chrom", "ica", "combined"):
            component = components.get(key)
            if not component:
                continue
            freqs = np.asarray(component["band_freqs"], dtype=np.float32)
            power = np.asarray(component["band_power"], dtype=np.float32)
            if freqs.size and power.size:
                if key == "combined":
                    ax.fill_between(freqs, power, color="#152B33", alpha=0.08)
                ax.plot(
                    freqs,
                    power,
                    linewidth=2.7 if key == "combined" else 1.35,
                    label=component["name"],
                    color=colors[key],
                    alpha=0.96 if key == "combined" else 0.72,
                )
        ax.axvspan(1.0, 1.5, color="#83B366", alpha=0.16, lw=0)
        ax.text(1.02, ax.get_ylim()[1] * 0.92, "60-90 BPM focus", fontsize=8, color="#54733A", fontweight="bold")
        ax.set_xlim(0.65, 2.55)
        style_axis(ax, "FFT Frequency Evidence", "Frequency (Hz)", "Power")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.patch.set_facecolor("#FFFDF4")
        add_plot_signature(fig, "FREQUENCY DOMAIN PEAK ANALYSIS")
        graphs["fft"] = save_plot(fig, output_dir / "fft_evidence.png")
    return graphs


def downsample(values: np.ndarray, limit: int = 240) -> list[float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size <= limit:
        return [round(float(value), 6) for value in values]
    indices = np.linspace(0, values.size - 1, limit).astype(np.int32)
    return [round(float(values[index]), 6) for index in indices]


def build_chart_data(result: dict[str, object]) -> dict[str, object]:
    roi_series = []
    for name, values in result.get("roi_green_traces", {}).items():
        roi_series.append(
            {
                "name": name.replace("_", " ").title(),
                "values": downsample(np.asarray(values, dtype=np.float32)),
            }
        )

    signal_series = []
    spectrum_series = []
    components = result.get("components", {})
    for key in ("green", "pos", "chrom", "ica", "combined"):
        component = components.get(key)
        if not component:
            continue
        signal_series.append(
            {
                "name": component["name"],
                "values": downsample(np.asarray(component["filtered_signal"], dtype=np.float32)),
            }
        )
        spectrum_series.append(
            {
                "name": component["name"],
                "x": downsample(np.asarray(component["band_freqs"], dtype=np.float32)),
                "y": downsample(np.asarray(component["band_power"], dtype=np.float32)),
            }
        )

    return {
        "roi": roi_series,
        "signals": signal_series,
        "spectrum": spectrum_series,
    }


def result_for_template(
    result: dict[str, object],
    render_id: str,
    frames: dict[str, list[str]],
    graphs: dict[str, str],
    chart_data: dict[str, object],
) -> dict[str, object]:
    components = result.get("components", {})
    component_rows = []
    for key in ("green", "pos", "chrom", "ica", "combined"):
        component = components.get(key)
        if component is None:
            continue
        component_rows.append(
            {
                "name": component["name"],
                "bpm": round(float(component["peak_bpm"]), 1),
                "score": round(float(component["live_score"]), 2),
                "grade": qualitative_level(float(component["live_score"])),
                "primary": key == result.get("primary_component"),
            }
        )

    roi_rows = []
    for name, quality in result.get("roi_quality", {}).items():
        roi_rows.append(
            {
                "name": name.replace("_", " ").title(),
                "score": round(float(quality), 2),
                "grade": qualitative_level(float(quality)),
                "contribution": "masked",
            }
        )

    return {
        "status": safe_status(str(result.get("status", ""))),
        "status_raw": str(result.get("status", "")),
        "bpm": round(float(result.get("peak_bpm", 0.0) or 0.0), 1),
        "pulse_score": round(float(result.get("live_score", 0.0) or 0.0), 2),
        "motion_score": round(float(result.get("motion_quality", 0.0) or 0.0), 2),
        "signal_grade": qualitative_level(float(result.get("live_score", 0.0) or 0.0)),
        "frames_read": int(result.get("total_frames", 0) or 0),
        "valid_frames": int(result.get("valid_face_frames", 0) or 0),
        "primary_signal": str(result.get("primary_component", "n/a")).upper(),
        "component_rows": component_rows,
        "roi_rows": roi_rows,
        "videos": {
            "outline": url_for("render_file", render_id=render_id, filename="outline.mp4"),
            "green": url_for("render_file", render_id=render_id, filename="green.mp4"),
            "frequency": url_for("render_file", render_id=render_id, filename="frequency.mp4"),
        },
        "frames": {
            "outline": [url_for("render_file", render_id=render_id, filename=f"outline_frames/{name}") for name in frames.get("outline", [])],
            "green": [url_for("render_file", render_id=render_id, filename=f"green_frames/{name}") for name in frames.get("green", [])],
            "frequency": [url_for("render_file", render_id=render_id, filename=f"frequency_frames/{name}") for name in frames.get("frequency", [])],
        },
        "graphs": {
            key: url_for("render_file", render_id=render_id, filename=f"graphs/{filename}")
            for key, filename in graphs.items()
        },
        "chart_data": chart_data,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result_data = None
    error = None
    config_path = request.form.get("config_path", "configs/liveness_v2.yaml")

    if request.method == "POST":
        uploaded = request.files.get("video")
        if not uploaded or uploaded.filename == "":
            error = "Upload a face video first."
        else:
            try:
                WEB_OUTPUT.mkdir(parents=True, exist_ok=True)
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                RENDER_DIR.mkdir(parents=True, exist_ok=True)
                render_id = uuid4().hex
                suffix = Path(uploaded.filename).suffix or ".mp4"
                upload_path = UPLOAD_DIR / f"{render_id}{suffix}"
                uploaded.save(upload_path)

                config, roi_config, signal_config, classification_config = load_bundle(config_path)
                with build_face_mesh(**config["face_mesh"]) as face_mesh:
                    result = analyze_video_for_liveness_v2(
                        upload_path,
                        face_mesh,
                        roi_config,
                        signal_config,
                        classification_config,
                        fps_hint=float(config["input"].get("fps_hint", 30.0)),
                        max_seconds=int(config["input"].get("max_seconds", 20)),
                        inspect_frame_limit=int(config["input"].get("inspect_frame_limit", 120)),
                    )

                render_path = RENDER_DIR / render_id
                write_video(result.get("annotated_frames", []), render_path / "outline.mp4", float(result.get("fps", 30.0)))
                write_video(result.get("green_heatmap_frames", []), render_path / "green.mp4", float(result.get("fps", 30.0)))
                write_video(result.get("frequency_heatmap_frames", []), render_path / "frequency.mp4", float(result.get("fps", 30.0)))
                frame_sets = {
                    "outline": write_frame_sequence(result.get("annotated_frames", []), render_path / "outline_frames"),
                    "green": write_frame_sequence(result.get("green_heatmap_frames", []), render_path / "green_frames"),
                    "frequency": write_frame_sequence(result.get("frequency_heatmap_frames", []), render_path / "frequency_frames"),
                }
                graph_files = build_graphs(result, render_path / "graphs")

                with (render_path / "result.json").open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "status": str(result.get("status", "")),
                            "bpm": float(result.get("peak_bpm", 0.0) or 0.0),
                            "pulse_score": float(result.get("live_score", 0.0) or 0.0),
                            "motion_score": float(result.get("motion_quality", 0.0) or 0.0),
                            "frames_read": int(result.get("total_frames", 0) or 0),
                            "valid_frames": int(result.get("valid_face_frames", 0) or 0),
                        },
                        handle,
                        indent=2,
                    )

                chart_data = build_chart_data(result)
                result_data = result_for_template(result, render_id, frame_sets, graph_files, chart_data)
            except Exception as exc:
                error = str(exc)

    return render_template("liveness_v2_web.html", result=result_data, error=error, config_path=config_path)


@app.route("/renders/<render_id>/<path:filename>")
def render_file(render_id: str, filename: str):
    return send_from_directory(RENDER_DIR / render_id, filename)


def main() -> None:
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
