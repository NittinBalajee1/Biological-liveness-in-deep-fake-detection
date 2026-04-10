from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


FOREHEAD_IDS = [10, 67, 69, 104, 108, 109, 151, 338, 337, 299, 296, 336]
LEFT_CHEEK_IDS = [50, 101, 118, 119, 120, 121, 123, 147, 187, 205, 206]
RIGHT_CHEEK_IDS = [280, 330, 347, 348, 349, 350, 352, 376, 411, 425, 426]


@dataclass
class ROIConfig:
    use_forehead: bool = True
    use_left_cheek: bool = True
    use_right_cheek: bool = True


@dataclass
class SignalConfig:
    min_samples: int = 90
    low_hz: float = 0.7
    high_hz: float = 2.5
    detrend_window_seconds: float = 1.2
    smoothing_window: int = 5


@dataclass
class ClassificationConfig:
    snr_threshold: float = 4.5
    peak_prominence: float = 0.015
    min_valid_frames: int = 45


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


def extract_rois(frame_bgr: np.ndarray, face_landmarks, roi_config: ROIConfig) -> dict[str, np.ndarray]:
    height, width = frame_bgr.shape[:2]
    rois: dict[str, np.ndarray] = {}
    if roi_config.use_forehead:
        rois["forehead"] = _landmarks_to_points(face_landmarks, width, height, FOREHEAD_IDS)
    if roi_config.use_left_cheek:
        rois["left_cheek"] = _landmarks_to_points(face_landmarks, width, height, LEFT_CHEEK_IDS)
    if roi_config.use_right_cheek:
        rois["right_cheek"] = _landmarks_to_points(face_landmarks, width, height, RIGHT_CHEEK_IDS)
    return rois


def _mean_green_in_polygon(frame_bgr: np.ndarray, polygon: np.ndarray) -> float:
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    green = frame_bgr[:, :, 1]
    values = green[mask == 255]
    return float(values.mean()) if values.size else 0.0


def draw_rois(frame_bgr: np.ndarray, rois: dict[str, np.ndarray]) -> np.ndarray:
    overlay = frame_bgr.copy()
    colors = {
        "forehead": (86, 210, 240),
        "left_cheek": (108, 196, 112),
        "right_cheek": (108, 196, 112),
    }
    for name, polygon in rois.items():
        cv2.polylines(overlay, [polygon], isClosed=True, color=colors.get(name, (86, 210, 240)), thickness=2, lineType=cv2.LINE_AA)
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
    if low >= high:
        return signal.copy()
    b, a = butter(3, [low, high], btype="bandpass")
    if signal.size < 16:
        return signal.copy()
    return filtfilt(b, a, signal)


def analyze_pulse_signal(
    raw_signal: list[float],
    fps: float,
    signal_config: SignalConfig,
    classification_config: ClassificationConfig,
) -> dict[str, object]:
    signal = np.asarray(raw_signal, dtype=np.float32)
    if signal.size < signal_config.min_samples:
        return {
            "status": "insufficient_data",
            "reason": "Not enough valid frames for pulse analysis.",
            "raw_signal": signal,
        }

    detrended = detrend_signal(signal, fps, signal_config.detrend_window_seconds)
    smoothed = moving_average(detrended, signal_config.smoothing_window)
    filtered = bandpass_filter(smoothed, fps, signal_config.low_hz, signal_config.high_hz)

    spectrum = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(filtered.size, d=1.0 / fps)
    power = np.abs(spectrum)
    valid_band = (freqs >= signal_config.low_hz) & (freqs <= signal_config.high_hz)
    band_freqs = freqs[valid_band]
    band_power = power[valid_band]
    if band_power.size == 0:
        return {
            "status": "insufficient_data",
            "reason": "No usable power inside the physiological band.",
            "raw_signal": signal,
            "filtered_signal": filtered,
            "freqs": freqs,
            "power": power,
        }

    peak_idx = int(np.argmax(band_power))
    peak_freq = float(band_freqs[peak_idx])
    peak_power = float(band_power[peak_idx])
    noise_floor = float(np.median(band_power) + 1e-6)
    snr = peak_power / noise_floor
    bpm = peak_freq * 60.0

    peaks, peak_props = find_peaks(band_power, prominence=classification_config.peak_prominence)
    peak_count = int(peaks.size)
    is_live = (
        snr >= classification_config.snr_threshold
        and classification_config.min_valid_frames <= signal.size
        and 42.0 <= bpm <= 150.0
    )
    return {
        "status": "authenticated" if is_live else "synthetic_media_detected",
        "raw_signal": signal,
        "filtered_signal": filtered,
        "freqs": freqs,
        "power": power,
        "band_freqs": band_freqs,
        "band_power": band_power,
        "peak_freq_hz": peak_freq,
        "peak_bpm": bpm,
        "peak_power": peak_power,
        "snr": snr,
        "peak_count": peak_count,
    }


def analyze_video_for_liveness(
    video_path: str | Path,
    face_mesh,
    roi_config: ROIConfig,
    signal_config: SignalConfig,
    classification_config: ClassificationConfig,
    fps_hint: float = 30.0,
    max_seconds: int = 20,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or fps_hint or 30.0)
    max_frames = int(max_seconds * fps) if max_seconds > 0 else None

    raw_signal: list[float] = []
    annotated_frames: list[np.ndarray] = []
    valid_faces = 0
    total_frames = 0

    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            total_frames += 1
            if max_frames is not None and total_frames > max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                continue

            face_landmarks = results.multi_face_landmarks[0]
            rois = extract_rois(frame, face_landmarks, roi_config)
            if not rois:
                continue

            values = [_mean_green_in_polygon(frame, polygon) for polygon in rois.values()]
            if not values:
                continue
            raw_signal.append(float(np.mean(values)))
            valid_faces += 1

            if len(annotated_frames) < 5:
                annotated_frames.append(draw_rois(frame, rois))
    finally:
        capture.release()

    analysis = analyze_pulse_signal(raw_signal, fps, signal_config, classification_config)
    analysis.update(
        {
            "fps": fps,
            "total_frames": total_frames,
            "valid_face_frames": valid_faces,
            "annotated_frames": annotated_frames,
        }
    )
    return analysis
