from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning


FOREHEAD_CENTER_IDS = [10, 67, 69, 104, 108, 109, 151, 337, 338, 299, 296, 336]
FOREHEAD_LEFT_IDS = [54, 68, 71, 103, 104, 105, 66, 107, 9, 108]
FOREHEAD_RIGHT_IDS = [284, 298, 301, 333, 334, 335, 296, 336, 9, 107]
LEFT_CHEEK_UPPER_IDS = [50, 101, 118, 119, 120, 121, 123, 147, 187, 205, 206, 207]
RIGHT_CHEEK_UPPER_IDS = [280, 330, 347, 348, 349, 350, 352, 376, 411, 425, 426, 427]
LEFT_TEMPLE_IDS = [34, 139, 127, 116, 117, 118, 100, 47]
RIGHT_TEMPLE_IDS = [264, 368, 356, 345, 346, 347, 329, 277]

ROI_INDEX_MAP = {
    "forehead_center": FOREHEAD_CENTER_IDS,
    "forehead_left": FOREHEAD_LEFT_IDS,
    "forehead_right": FOREHEAD_RIGHT_IDS,
    "left_cheek_upper": LEFT_CHEEK_UPPER_IDS,
    "right_cheek_upper": RIGHT_CHEEK_UPPER_IDS,
    "left_temple": LEFT_TEMPLE_IDS,
    "right_temple": RIGHT_TEMPLE_IDS,
}


@dataclass
class ROIConfig:
    forehead_center: bool = True
    forehead_left: bool = True
    forehead_right: bool = True
    left_cheek_upper: bool = True
    right_cheek_upper: bool = True
    left_temple: bool = False
    right_temple: bool = False


@dataclass
class SignalConfig:
    min_samples: int = 120
    low_hz: float = 0.7
    high_hz: float = 2.5
    resting_low_hz: float = 1.0
    resting_high_hz: float = 1.5
    detrend_window_seconds: float = 1.4
    pos_window_seconds: float = 1.6
    smoothing_window: int = 5
    roi_trim_percent: float = 0.1
    ica_components: int = 3
    roi_quality_floor: float = 0.05


@dataclass
class ClassificationConfig:
    snr_threshold: float = 3.2
    live_score_threshold: float = 0.68
    peak_prominence: float = 0.015
    min_valid_frames: int = 75
    min_bpm: float = 60.0
    max_bpm: float = 150.0
    min_support_count: int = 2
    min_motion_quality: float = 0.35
    low_brightness_relaxation: float = 0.1
    low_brightness_cutoff: float = 0.34


def build_face_mesh(
    max_num_faces: int = 1,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for liveness detection. Install requirements first."
        ) from exc

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    except ImportError as exc:
        raise ImportError(
            "Installed mediapipe package does not expose Face Mesh in the expected locations."
        ) from exc

    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def _landmarks_to_points(face_landmarks, width: int, height: int, indices: list[int]) -> np.ndarray:
    points = []
    for idx in indices:
        landmark = face_landmarks.landmark[idx]
        x = int(np.clip(landmark.x * width, 0, width - 1))
        y = int(np.clip(landmark.y * height, 0, height - 1))
        points.append((x, y))
    return np.array(points, dtype=np.int32)


def extract_dense_rois(frame_bgr: np.ndarray, face_landmarks, roi_config: ROIConfig) -> dict[str, np.ndarray]:
    height, width = frame_bgr.shape[:2]
    flags = roi_config.__dict__
    rois: dict[str, np.ndarray] = {}
    for name, indices in ROI_INDEX_MAP.items():
        if flags.get(name, False):
            rois[name] = _landmarks_to_points(face_landmarks, width, height, indices)
    return rois


def estimate_face_center(face_landmarks, width: int, height: int) -> np.ndarray:
    stable_ids = [1, 4, 6, 33, 133, 263, 362, 152]
    points = _landmarks_to_points(face_landmarks, width, height, stable_ids)
    return points.mean(axis=0).astype(np.float32)


def compute_motion_quality(face_centers: list[np.ndarray], frame_shape: tuple[int, int]) -> dict[str, float]:
    if len(face_centers) < 3:
        return {
            "motion_quality": 0.0,
            "mean_motion_px": 0.0,
            "motion_jitter_px": 0.0,
            "motion_penalty": 1.0,
        }

    centers = np.stack(face_centers, axis=0)
    deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    height, width = frame_shape
    diagonal = float(np.hypot(width, height) + 1e-6)
    normalized_mean = float(np.mean(deltas) / diagonal)
    normalized_jitter = float(np.std(deltas) / diagonal)
    motion_penalty = np.clip((normalized_mean * 26.0) + (normalized_jitter * 34.0), 0.0, 1.0)
    motion_quality = float(1.0 - motion_penalty)
    return {
        "motion_quality": motion_quality,
        "mean_motion_px": float(np.mean(deltas)),
        "motion_jitter_px": float(np.std(deltas)),
        "motion_penalty": float(motion_penalty),
    }


def _robust_rgb_mean(frame_bgr: np.ndarray, polygon: np.ndarray, trim_percent: float) -> np.ndarray:
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    pixels = frame_bgr[mask == 255].astype(np.float32) / 255.0
    if pixels.size == 0:
        return np.zeros(3, dtype=np.float32)

    trim_percent = float(np.clip(trim_percent, 0.0, 0.45))
    if trim_percent > 0.0 and pixels.shape[0] > 12:
        low = np.quantile(pixels, trim_percent, axis=0)
        high = np.quantile(pixels, 1.0 - trim_percent, axis=0)
        pixels = np.clip(pixels, low, high)
    return pixels.mean(axis=0)


def draw_rois(
    frame_bgr: np.ndarray,
    rois: dict[str, np.ndarray],
    roi_values: dict[str, float] | None = None,
    mode: str = "outline",
) -> np.ndarray:
    overlay = frame_bgr.copy()
    colors = {
        "forehead_center": (226, 220, 178),
        "forehead_left": (243, 150, 101),
        "forehead_right": (243, 150, 101),
        "left_cheek_upper": (102, 179, 131),
        "right_cheek_upper": (102, 179, 131),
        "left_temple": (234, 164, 211),
        "right_temple": (234, 164, 211),
    }
    if mode == "heatmap" and roi_values:
        heat_layer = frame_bgr.copy()
        values = np.asarray(list(roi_values.values()), dtype=np.float32)
        min_value = float(values.min()) if values.size else 0.0
        max_value = float(values.max()) if values.size else 1.0
        for name, polygon in rois.items():
            value = float(roi_values.get(name, min_value))
            normalized = (value - min_value) / (max_value - min_value + 1e-6)
            low = np.array([226, 220, 178], dtype=np.float32)
            mid = np.array([102, 179, 131], dtype=np.float32)
            high = np.array([243, 150, 101], dtype=np.float32)
            if normalized < 0.5:
                color = low * (1.0 - normalized * 2.0) + mid * (normalized * 2.0)
            else:
                color = mid * (2.0 - normalized * 2.0) + high * (normalized * 2.0 - 1.0)
            color = color.astype(np.uint8).tolist()
            cv2.fillConvexPoly(heat_layer, polygon, color)
        overlay = cv2.addWeighted(heat_layer, 0.34, frame_bgr, 0.66, 0)

    for name, polygon in rois.items():
        smooth_polygon = cv2.convexHull(polygon)
        cv2.polylines(
            overlay,
            [smooth_polygon],
            isClosed=True,
            color=colors.get(name, (86, 210, 240)),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return overlay


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.copy()
    padded = np.pad(signal, (window // 2, window - 1 - window // 2), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def detrend_signal(signal: np.ndarray, fps: float, detrend_window_seconds: float) -> np.ndarray:
    window = max(3, int(round(fps * detrend_window_seconds)))
    trend = moving_average(signal, window)
    return signal - trend


def bandpass_filter(signal: np.ndarray, fps: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = fps / 2.0
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.99)
    if low >= high or signal.size < 16:
        return signal.copy()
    b, a = butter(3, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)


def _zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    return (signal - signal.mean()) / (signal.std() + 1e-6)


def _normalize_rgb_trace(rgb_trace: np.ndarray) -> np.ndarray:
    channel_mean = np.mean(rgb_trace, axis=0, keepdims=True) + 1e-6
    return rgb_trace / channel_mean - 1.0


def extract_green_signal(rgb_trace: np.ndarray, fps: float, signal_config: SignalConfig) -> np.ndarray:
    green = rgb_trace[:, 1]
    total = np.clip(np.sum(rgb_trace, axis=1), 1e-6, None)
    normalized_green = green / total
    detrended = detrend_signal(normalized_green, fps, signal_config.detrend_window_seconds)
    smoothed = moving_average(detrended, signal_config.smoothing_window)
    return bandpass_filter(smoothed, fps, signal_config.low_hz, signal_config.high_hz)


def extract_pos_signal(rgb_trace: np.ndarray, fps: float, signal_config: SignalConfig) -> np.ndarray:
    trace = np.asarray(rgb_trace, dtype=np.float32)
    if trace.shape[0] < 3:
        return np.zeros(trace.shape[0], dtype=np.float32)

    normalized = _normalize_rgb_trace(trace) + 1.0
    window = max(8, int(round(signal_config.pos_window_seconds * fps)))
    pulse = np.zeros(trace.shape[0], dtype=np.float32)
    overlap = np.zeros(trace.shape[0], dtype=np.float32)

    for start in range(0, trace.shape[0] - window + 1):
        segment = normalized[start : start + window].T
        mean_color = np.mean(segment, axis=1, keepdims=True) + 1e-6
        segment = segment / mean_color
        s1 = segment[1] - segment[2]
        s2 = segment[1] + segment[2] - 2.0 * segment[0]
        alpha = np.std(s1) / (np.std(s2) + 1e-6)
        h = s1 + alpha * s2
        h = _zscore(h)
        pulse[start : start + window] += h
        overlap[start : start + window] += 1.0

    overlap = np.maximum(overlap, 1.0)
    pulse = pulse / overlap
    pulse = detrend_signal(pulse, fps, signal_config.detrend_window_seconds)
    pulse = moving_average(pulse, signal_config.smoothing_window)
    return bandpass_filter(pulse, fps, signal_config.low_hz, signal_config.high_hz)


def extract_chrom_signal(rgb_trace: np.ndarray, fps: float, signal_config: SignalConfig) -> np.ndarray:
    trace = _normalize_rgb_trace(np.asarray(rgb_trace, dtype=np.float32)) + 1.0
    if trace.shape[0] < 3:
        return np.zeros(trace.shape[0], dtype=np.float32)

    r = trace[:, 2]
    g = trace[:, 1]
    b = trace[:, 0]
    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b
    alpha = np.std(x) / (np.std(y) + 1e-6)
    pulse = x - alpha * y
    pulse = detrend_signal(pulse, fps, signal_config.detrend_window_seconds)
    pulse = moving_average(pulse, signal_config.smoothing_window)
    return bandpass_filter(pulse, fps, signal_config.low_hz, signal_config.high_hz)


def extract_ica_signal(rgb_trace: np.ndarray, fps: float, signal_config: SignalConfig) -> np.ndarray:
    trace = _normalize_rgb_trace(np.asarray(rgb_trace, dtype=np.float32))
    if trace.shape[0] < 16:
        return np.zeros(trace.shape[0], dtype=np.float32)

    n_components = min(signal_config.ica_components, trace.shape[1], trace.shape[0])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            components = FastICA(
                n_components=n_components,
                random_state=7,
                whiten="unit-variance",
                max_iter=2000,
                tol=0.01,
            ).fit_transform(trace)
    except Exception:
        return np.zeros(trace.shape[0], dtype=np.float32)

    best_signal = np.zeros(trace.shape[0], dtype=np.float32)
    best_score = -np.inf
    for index in range(components.shape[1]):
        candidate = moving_average(components[:, index], signal_config.smoothing_window)
        candidate = bandpass_filter(candidate, fps, signal_config.low_hz, signal_config.high_hz)
        quality = analyze_frequency_component(
            candidate,
            fps,
            signal_config,
            focus_low_hz=signal_config.resting_low_hz,
            focus_high_hz=signal_config.resting_high_hz,
        )
        if quality["live_score"] > best_score:
            best_score = quality["live_score"]
            best_signal = candidate
    return best_signal


def analyze_frequency_component(
    signal: np.ndarray,
    fps: float,
    signal_config: SignalConfig,
    focus_low_hz: float | None = None,
    focus_high_hz: float | None = None,
) -> dict[str, float | np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal))

    valid_band = (freqs >= signal_config.low_hz) & (freqs <= signal_config.high_hz)
    band_freqs = freqs[valid_band]
    band_power = power[valid_band]
    if band_power.size == 0:
        return {
            "freqs": freqs,
            "power": power,
            "band_freqs": band_freqs,
            "band_power": band_power,
            "peak_freq_hz": 0.0,
            "peak_bpm": 0.0,
            "snr": 0.0,
            "peak_prominence_ratio": 0.0,
            "focus_ratio": 0.0,
            "live_score": 0.0,
        }

    peak_idx = int(np.argmax(band_power))
    peak_power = float(band_power[peak_idx])
    peak_freq = float(band_freqs[peak_idx])
    noise_floor = float(np.median(band_power) + 1e-6)
    snr = peak_power / noise_floor

    peaks, props = find_peaks(band_power, prominence=max(1e-6, float(np.std(band_power) * 0.3)))
    prominence = float(np.max(props["prominences"])) if peaks.size and "prominences" in props else 0.0
    peak_prominence_ratio = prominence / (peak_power + 1e-6)

    focus_low_hz = signal_config.resting_low_hz if focus_low_hz is None else focus_low_hz
    focus_high_hz = signal_config.resting_high_hz if focus_high_hz is None else focus_high_hz
    focus_band = (band_freqs >= focus_low_hz) & (band_freqs <= focus_high_hz)
    focus_ratio = float(band_power[focus_band].sum() / (band_power.sum() + 1e-6)) if np.any(focus_band) else 0.0

    snr_term = snr / (snr + 1.0)
    live_score = 0.55 * snr_term + 0.25 * focus_ratio + 0.20 * peak_prominence_ratio
    return {
        "freqs": freqs,
        "power": power,
        "band_freqs": band_freqs,
        "band_power": band_power,
        "peak_freq_hz": peak_freq,
        "peak_bpm": peak_freq * 60.0,
        "snr": snr,
        "peak_prominence_ratio": peak_prominence_ratio,
        "focus_ratio": focus_ratio,
        "live_score": live_score,
    }


def _component_dict(name: str, filtered_signal: np.ndarray, component_stats: dict[str, float | np.ndarray]) -> dict[str, object]:
    return {
        "name": name,
        "filtered_signal": filtered_signal,
        "freqs": component_stats["freqs"],
        "power": component_stats["power"],
        "band_freqs": component_stats["band_freqs"],
        "band_power": component_stats["band_power"],
        "peak_freq_hz": component_stats["peak_freq_hz"],
        "peak_bpm": component_stats["peak_bpm"],
        "snr": component_stats["snr"],
        "peak_prominence_ratio": component_stats["peak_prominence_ratio"],
        "focus_ratio": component_stats["focus_ratio"],
        "live_score": component_stats["live_score"],
    }


def analyze_pulse_pipeline(
    rgb_trace: np.ndarray,
    fps: float,
    signal_config: SignalConfig,
    classification_config: ClassificationConfig,
    motion_quality: float = 1.0,
) -> dict[str, object]:
    if rgb_trace.shape[0] < signal_config.min_samples:
        return {
            "status": "insufficient_data",
            "reason": "Not enough valid face frames for pulse analysis.",
            "rgb_trace": rgb_trace,
        }

    green_signal = extract_green_signal(rgb_trace, fps, signal_config)
    pos_signal = extract_pos_signal(rgb_trace, fps, signal_config)
    chrom_signal = extract_chrom_signal(rgb_trace, fps, signal_config)
    ica_signal = extract_ica_signal(rgb_trace, fps, signal_config)

    green_stats = analyze_frequency_component(green_signal, fps, signal_config)
    pos_stats = analyze_frequency_component(pos_signal, fps, signal_config)
    chrom_stats = analyze_frequency_component(chrom_signal, fps, signal_config)
    ica_stats = analyze_frequency_component(ica_signal, fps, signal_config)

    combined_signal = np.mean(
        np.stack([_zscore(green_signal), _zscore(pos_signal), _zscore(chrom_signal), _zscore(ica_signal)], axis=0),
        axis=0,
    )
    combined_signal = bandpass_filter(combined_signal, fps, signal_config.low_hz, signal_config.high_hz)
    combined_stats = analyze_frequency_component(combined_signal, fps, signal_config)

    components = {
        "green": _component_dict("Green", green_signal, green_stats),
        "pos": _component_dict("POS", pos_signal, pos_stats),
        "chrom": _component_dict("CHROM", chrom_signal, chrom_stats),
        "ica": _component_dict("ICA", ica_signal, ica_stats),
        "combined": _component_dict("Combined", combined_signal, combined_stats),
    }

    primary_key = max(components, key=lambda key: float(components[key]["live_score"]))
    primary = components[primary_key]

    brightness = float(np.mean(rgb_trace))
    adaptive_snr_threshold = classification_config.snr_threshold
    adaptive_score_threshold = classification_config.live_score_threshold
    if brightness < classification_config.low_brightness_cutoff:
        adaptive_snr_threshold = max(2.8, adaptive_snr_threshold - classification_config.low_brightness_relaxation)
        adaptive_score_threshold = max(0.60, adaptive_score_threshold - classification_config.low_brightness_relaxation * 0.25)

    if motion_quality < 0.65:
        adaptive_snr_threshold += (0.65 - motion_quality) * 1.2
        adaptive_score_threshold += (0.65 - motion_quality) * 0.12

    bpm = float(primary["peak_bpm"])
    in_band = classification_config.min_bpm <= bpm <= classification_config.max_bpm
    resting_band = signal_config.resting_low_hz * 60.0 <= bpm <= signal_config.resting_high_hz * 60.0
    support_count = sum(
        1
        for key in ("green", "pos", "chrom", "ica")
        if float(components[key]["live_score"]) >= adaptive_score_threshold * 0.9
        and classification_config.min_bpm <= float(components[key]["peak_bpm"]) <= classification_config.max_bpm
    )

    strong_primary = (
        float(primary["snr"]) >= adaptive_snr_threshold
        and float(primary["live_score"]) >= adaptive_score_threshold
        and in_band
    )
    stable_high_confidence_primary = (
        float(primary["live_score"]) >= adaptive_score_threshold
        and motion_quality >= 0.8
        and resting_band
        and in_band
    )
    stable_biological_signal = (
        strong_primary
        and (
            (
                support_count >= classification_config.min_support_count
                and (resting_band or float(primary["live_score"]) >= adaptive_score_threshold + 0.06)
            )
            or stable_high_confidence_primary
        )
    )
    is_live = (
        (stable_biological_signal or stable_high_confidence_primary)
        and rgb_trace.shape[0] >= classification_config.min_valid_frames
        and motion_quality >= classification_config.min_motion_quality
    )

    return {
        "status": "authenticated" if is_live else "synthetic_media_detected",
        "rgb_trace": rgb_trace,
        "green_trace": rgb_trace[:, 1],
        "components": components,
        "primary_component": primary_key,
        "peak_freq_hz": primary["peak_freq_hz"],
        "peak_bpm": primary["peak_bpm"],
        "snr": primary["snr"],
        "live_score": primary["live_score"],
        "support_count": support_count,
        "brightness": brightness,
        "adaptive_snr_threshold": adaptive_snr_threshold,
        "adaptive_score_threshold": adaptive_score_threshold,
    }


def analyze_video_for_liveness_v2(
    video_path: str | Path,
    face_mesh,
    roi_config: ROIConfig,
    signal_config: SignalConfig,
    classification_config: ClassificationConfig,
    fps_hint: float = 30.0,
    max_seconds: int = 20,
    inspect_frame_limit: int = 120,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or fps_hint or 30.0)
    max_frames = int(max_seconds * fps) if max_seconds > 0 else None

    roi_rgb_history: list[np.ndarray] = []
    annotated_frames: list[np.ndarray] = []
    green_heatmap_frames: list[np.ndarray] = []
    frequency_heatmap_frames: list[np.ndarray] = []
    annotated_frame_indices: list[int] = []
    inspected_raw_frames: list[np.ndarray] = []
    inspected_rois: list[dict[str, np.ndarray]] = []
    valid_faces = 0
    total_frames = 0
    roi_names: list[str] = []
    face_centers: list[np.ndarray] = []
    frame_shape: tuple[int, int] | None = None

    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            total_frames += 1
            frame_shape = frame.shape[:2]
            if max_frames is not None and total_frames > max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                continue

            face_landmarks = results.multi_face_landmarks[0]
            face_centers.append(estimate_face_center(face_landmarks, frame.shape[1], frame.shape[0]))
            rois = extract_dense_rois(frame, face_landmarks, roi_config)
            if not rois:
                continue

            rgb_values = {
                name: _robust_rgb_mean(frame, polygon, signal_config.roi_trim_percent)
                for name, polygon in rois.items()
            }
            valid_values = [value for value in rgb_values.values() if np.any(value > 0)]
            if not valid_values:
                continue

            roi_names = list(rgb_values.keys())
            roi_rgb_history.append(np.stack([rgb_values[name] for name in roi_names], axis=0))
            valid_faces += 1

            if len(annotated_frames) < inspect_frame_limit:
                annotated_frames.append(draw_rois(frame, rois))
                green_values = {name: float(value[1]) for name, value in rgb_values.items()}
                green_heatmap_frames.append(draw_rois(frame, rois, green_values, mode="heatmap"))
                inspected_raw_frames.append(frame.copy())
                inspected_rois.append({name: polygon.copy() for name, polygon in rois.items()})
                annotated_frame_indices.append(total_frames)
    finally:
        capture.release()

    if not roi_rgb_history:
        return {
            "status": "insufficient_data",
            "reason": "No valid face ROIs were extracted from the video.",
            "fps": fps,
            "total_frames": total_frames,
            "valid_face_frames": valid_faces,
            "annotated_frames": annotated_frames,
        }

    roi_rgb_array = np.stack(roi_rgb_history, axis=0)
    roi_green_traces = {
        name: roi_rgb_array[:, idx, 1]
        for idx, name in enumerate(roi_names)
    }
    roi_quality: dict[str, float] = {}
    quality_scores = []
    for idx, name in enumerate(roi_names):
        roi_trace = roi_rgb_array[:, idx, :]
        roi_green_signal = extract_green_signal(roi_trace, fps, signal_config)
        roi_stats = analyze_frequency_component(roi_green_signal, fps, signal_config)
        score = max(float(roi_stats["live_score"]), signal_config.roi_quality_floor)
        roi_quality[name] = score
        quality_scores.append(score)

    weights = np.asarray(quality_scores, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-6)
    rgb_trace = np.tensordot(roi_rgb_array, weights, axes=([1], [0]))
    motion_stats = compute_motion_quality(face_centers, frame_shape or (1, 1))

    if annotated_frames and roi_quality:
        frequency_values = {name: float(score) for name, score in roi_quality.items()}
        for raw_frame, rois in zip(inspected_raw_frames, inspected_rois, strict=False):
            frequency_heatmap_frames.append(draw_rois(raw_frame, rois, frequency_values, mode="heatmap"))

    analysis = analyze_pulse_pipeline(
        rgb_trace,
        fps,
        signal_config,
        classification_config,
        motion_quality=motion_stats["motion_quality"],
    )
    analysis.update(
        {
            "fps": fps,
            "total_frames": total_frames,
            "valid_face_frames": valid_faces,
            "annotated_frames": annotated_frames,
            "green_heatmap_frames": green_heatmap_frames,
            "frequency_heatmap_frames": frequency_heatmap_frames,
            "annotated_frame_indices": annotated_frame_indices,
            "roi_names": roi_names,
            "roi_green_traces": roi_green_traces,
            "roi_quality": roi_quality,
            "roi_weights": {name: float(weights[idx]) for idx, name in enumerate(roi_names)},
            **motion_stats,
        }
    )
    return analysis
