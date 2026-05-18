
from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, sr: int = 48000, mono: bool = False) -> tuple[np.ndarray, int]:
    """Extract audio from a video file via ffmpeg. Returns (audio, sr).

    If mono=False (default), returns shape (samples, channels) for stereo.
    If mono=True, returns shape (samples,).
    """
    channels = 1 if mono else 2
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",                     # drop video
        "-acodec", "pcm_f32le",    # 32-bit float PCM
        "-ar", str(sr),
        "-ac", str(channels),
        "-f", "wav",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:500]}")
    audio, out_sr = sf.read(io.BytesIO(result.stdout), dtype="float32")
    return audio, out_sr


def extract_audio_mono(video_path: str, sr: int = 48000) -> tuple[np.ndarray, int]:
    """Convenience: extract mono audio."""
    return extract_audio(video_path, sr=sr, mono=True)


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------

def detect_onsets(audio_mono: np.ndarray, sr: int,
                  units: str = "time", backtrack: bool = True) -> dict:
    """Detect audio onset times using librosa.

    Returns:
        {onsets: [float], count: int, method: str}
    """
    onsets = librosa.onset.onset_detect(
        y=audio_mono, sr=sr, units=units, backtrack=backtrack,
    )
    return {
        "onsets": [round(float(t), 4) for t in onsets],
        "count": len(onsets),
        "method": "librosa.onset_detect",
    }


# ---------------------------------------------------------------------------
# Pitch / F0 — using Praat (parselmouth) for speed and robustness
# ---------------------------------------------------------------------------

def _praat_pitch(audio_mono: np.ndarray, sr: int,
                 time_step: float = 0.01,
                 pitch_floor: float = 75.0,
                 pitch_ceiling: float = 1500.0):
    """Internal: create Praat Pitch object."""
    import parselmouth
    snd = parselmouth.Sound(audio_mono, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(
        time_step=time_step,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
    )
    return pitch


def pitch_contour(audio_mono: np.ndarray, sr: int,
                  time_step: float = 0.01) -> dict:
    """Extract full pitch contour using Praat.

    Returns:
        {times: [float], frequencies: [float|null], voiced_fraction: float,
         mean_hz: float|null, median_hz: float|null, method: str}
    """
    pitch = _praat_pitch(audio_mono, sr, time_step=time_step)
    times = pitch.xs()
    freqs = [pitch.get_value_at_time(t) for t in times]

    voiced = [f for f in freqs if f is not None and f > 0 and not np.isnan(f)]
    voiced_frac = len(voiced) / max(len(freqs), 1)

    return {
        "times": [round(float(t), 4) for t in times],
        "frequencies": [round(float(f), 2) if (f is not None and not np.isnan(f) and f > 0) else None for f in freqs],
        "voiced_fraction": round(voiced_frac, 3),
        "mean_hz": round(float(np.mean(voiced)), 2) if voiced else None,
        "median_hz": round(float(np.median(voiced)), 2) if voiced else None,
        "method": "praat_ac",
    }


def extract_pitch_at(audio_mono: np.ndarray, sr: int,
                     time_s: float, window_s: float = 0.15) -> dict:
    """Extract F0 at a specific timestamp (+/- window_s/2).

    Returns:
        {time_s: float, frequency_hz: float|null, confidence: str, method: str}
    """
    half = window_s / 2
    start = max(0, int((time_s - half) * sr))
    end = min(len(audio_mono), int((time_s + half) * sr))
    segment = audio_mono[start:end]

    if len(segment) < int(0.03 * sr):  # too short for pitch
        return {"time_s": time_s, "frequency_hz": None, "confidence": "insufficient_audio", "method": "praat_ac"}

    pitch = _praat_pitch(segment, sr)
    f0 = pitch.get_value_at_time(len(segment) / sr / 2)

    if f0 is None or np.isnan(f0) or f0 <= 0:
        return {"time_s": time_s, "frequency_hz": None, "confidence": "unvoiced", "method": "praat_ac"}

    return {
        "time_s": round(time_s, 4),
        "frequency_hz": round(float(f0), 2),
        "confidence": "voiced",
        "method": "praat_ac",
    }


def pitch_at_onsets(audio_mono: np.ndarray, sr: int,
                    onset_times: list[float], window_s: float = 0.2) -> dict:
    """Extract pitch at each detected onset. Returns list of {time, hz}."""
    results = []
    for t in onset_times:
        p = extract_pitch_at(audio_mono, sr, t, window_s)
        results.append({"time_s": p["time_s"], "hz": p["frequency_hz"]})
    valid = [r["hz"] for r in results if r["hz"] is not None]
    direction = None
    if len(valid) >= 2:
        diffs = np.diff(valid)
        if np.all(diffs > 0):
            direction = "ascending"
        elif np.all(diffs < 0):
            direction = "descending"
        else:
            direction = "non_monotonic"
    return {
        "pitches": results,
        "valid_count": len(valid),
        "direction": direction,
    }


# ---------------------------------------------------------------------------
# Loudness — LUFS (EBU R 128)
# ---------------------------------------------------------------------------

def extract_loudness(audio_mono: np.ndarray, sr: int,
                     start_s: float | None = None,
                     end_s: float | None = None) -> dict:
    """Measure integrated LUFS loudness of a segment.

    Returns:
        {lufs: float, start_s: float, end_s: float, method: str}
    """
    import pyloudnorm as pyln

    s = int((start_s or 0) * sr)
    e = int((end_s or len(audio_mono) / sr) * sr)
    segment = audio_mono[s:e]

    if len(segment) < int(0.4 * sr):
        # pyloudnorm needs >= 400ms
        return {"lufs": None, "start_s": start_s, "end_s": end_s, "method": "pyloudnorm", "error": "segment_too_short"}

    meter = pyln.Meter(sr, block_size=0.4)
    lufs = meter.integrated_loudness(segment)
    # pyloudnorm returns -inf for fully-silent segments. -inf is not valid
    # JSON (RFC 8259) and Gemini's API rejects it; coerce to None.
    lufs_val = float(lufs)
    if not np.isfinite(lufs_val):
        lufs_clean = None
    else:
        lufs_clean = round(lufs_val, 2)
    return {
        "lufs": lufs_clean,
        "start_s": round(start_s or 0, 4),
        "end_s": round((end_s or len(audio_mono) / sr), 4),
        "method": "pyloudnorm",
    }


def loudness_contour(audio_mono: np.ndarray, sr: int,
                     window_s: float = 0.4, hop_s: float = 0.1) -> dict:
    """Compute windowed LUFS contour over time.

    Returns:
        {times: [float], lufs_values: [float], method: str}
    """
    import pyloudnorm as pyln

    meter = pyln.Meter(sr, block_size=window_s)
    window_samples = int(window_s * sr)
    hop_samples = int(hop_s * sr)
    times, values = [], []

    for i in range(0, len(audio_mono) - window_samples, hop_samples):
        segment = audio_mono[i:i + window_samples]
        lufs = meter.integrated_loudness(segment)
        t = (i + window_samples / 2) / sr
        times.append(round(t, 3))
        # pyloudnorm returns -inf for silent windows; coerce to None so the
        # tool result is RFC-8259 valid JSON (Gemini's function_response
        # parser rejects -Infinity / Infinity / NaN literals).
        lufs_val = float(lufs)
        if not np.isfinite(lufs_val):
            values.append(None)
        else:
            values.append(round(lufs_val, 2))

    return {
        "times": times,
        "lufs_values": values,
        "method": "pyloudnorm_windowed",
    }


def rms_contour(audio_mono: np.ndarray, sr: int,
                frame_length: int = 2048, hop_length: int = 512) -> dict:
    """Compute RMS energy contour in dB.

    Returns:
        {times: [float], rms_db: [float]}
    """
    rms = librosa.feature.rms(y=audio_mono, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return {
        "times": [round(float(t), 4) for t in times],
        "rms_db": [round(float(v), 2) for v in rms_db],
    }


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------

def spectral_features(audio_mono: np.ndarray, sr: int,
                      start_s: float | None = None,
                      end_s: float | None = None) -> dict:
    """Extract spectral centroid, rolloff, bandwidth, and zero-crossing rate.

    Returns dict with mean and std for each feature.
    """
    s = int((start_s or 0) * sr)
    e = int((end_s or len(audio_mono) / sr) * sr)
    seg = audio_mono[s:e]

    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(seg)[0]

    def stat(arr):
        return {"mean": round(float(np.mean(arr)), 2), "std": round(float(np.std(arr)), 2)}

    return {
        "start_s": round(start_s or 0, 4),
        "end_s": round(end_s or len(audio_mono) / sr, 4),
        "centroid_hz": stat(centroid),
        "rolloff_hz": stat(rolloff),
        "bandwidth_hz": stat(bandwidth),
        "zcr": stat(zcr),
    }


# ---------------------------------------------------------------------------
# Stereo / spatial analysis
# ---------------------------------------------------------------------------

def stereo_balance(audio_stereo: np.ndarray, sr: int,
                   window_s: float = 0.5) -> dict:
    """Analyze left-right channel energy balance over time for spatial eval.

    Input audio must be stereo (samples, 2).

    Returns:
        {times: [float], balance: [float (-1=left, +1=right)],
         mean_balance: float, dominant_side: str}
    """
    if audio_stereo.ndim != 2 or audio_stereo.shape[1] != 2:
        return {"error": "input must be stereo (samples, 2)"}

    left = audio_stereo[:, 0]
    right = audio_stereo[:, 1]
    window = int(window_s * sr)
    hop = window // 2
    times, balances = [], []

    for i in range(0, len(left) - window, hop):
        l_rms = np.sqrt(np.mean(left[i:i + window] ** 2)) + 1e-10
        r_rms = np.sqrt(np.mean(right[i:i + window] ** 2)) + 1e-10
        # Balance: (R - L) / (R + L), range [-1, 1]
        bal = (r_rms - l_rms) / (r_rms + l_rms)
        t = (i + window / 2) / sr
        times.append(round(t, 3))
        balances.append(round(float(bal), 4))

    mean_bal = float(np.mean(balances)) if balances else 0.0
    dominant = "left" if mean_bal < -0.05 else ("right" if mean_bal > 0.05 else "center")

    return {
        "times": times,
        "balance": balances,
        "mean_balance": round(mean_bal, 4),
        "dominant_side": dominant,
        "method": "rms_balance",
    }


def ild_over_time(audio_stereo: np.ndarray, sr: int,
                  window_s: float = 0.5) -> dict:
    """Compute Interaural Level Difference (ILD) contour in dB.

    Positive ILD = louder in right channel.
    """
    if audio_stereo.ndim != 2 or audio_stereo.shape[1] != 2:
        return {"error": "input must be stereo"}

    left = audio_stereo[:, 0]
    right = audio_stereo[:, 1]
    window = int(window_s * sr)
    hop = window // 2
    times, ilds = [], []

    for i in range(0, len(left) - window, hop):
        l_rms = np.sqrt(np.mean(left[i:i + window] ** 2)) + 1e-10
        r_rms = np.sqrt(np.mean(right[i:i + window] ** 2)) + 1e-10
        ild_db = 20 * np.log10(r_rms / l_rms)
        times.append(round((i + window / 2) / sr, 3))
        ilds.append(round(float(ild_db), 2))

    return {
        "times": times,
        "ild_db": ilds,
        "mean_ild_db": round(float(np.mean(ilds)), 2) if ilds else 0.0,
    }


# ---------------------------------------------------------------------------
# Reverberation (RT60 estimation via Schroeder backward integration)
# ---------------------------------------------------------------------------

def estimate_rt60(audio_mono: np.ndarray, sr: int,
                  decay_db: float = 30) -> dict:
    """Estimate RT60 from audio using Schroeder backward integration.

    Uses T30 (30 dB decay) doubled to estimate RT60 by default.

    Returns:
        {rt60_s: float|null, method: str, decay_used_db: float}
    """
    # Find strongest onset and use the decay after it
    energy = audio_mono ** 2
    # Smooth with 10ms window
    win = int(0.01 * sr)
    if win < 1:
        win = 1
    kernel = np.ones(win) / win
    smoothed = np.convolve(energy, kernel, mode="same")

    peak_idx = np.argmax(smoothed)
    decay_signal = energy[peak_idx:]

    if len(decay_signal) < int(0.1 * sr):
        return {"rt60_s": None, "method": "schroeder", "error": "insufficient_decay"}

    # Schroeder backward integration
    schroeder = np.cumsum(decay_signal[::-1])[::-1]
    schroeder_db = 10 * np.log10(schroeder / (schroeder[0] + 1e-20) + 1e-20)

    # Find where decay crosses -5 dB and -(5+decay_db) dB
    start_db = -5
    end_db = start_db - decay_db

    start_idx = np.where(schroeder_db <= start_db)[0]
    end_idx = np.where(schroeder_db <= end_db)[0]

    if len(start_idx) == 0 or len(end_idx) == 0:
        return {"rt60_s": None, "method": "schroeder", "error": "decay_not_reached"}

    t_start = start_idx[0] / sr
    t_end = end_idx[0] / sr
    t_decay = t_end - t_start

    # Scale to 60 dB
    rt60 = t_decay * (60.0 / decay_db)

    return {
        "rt60_s": round(float(rt60), 3),
        "method": "schroeder",
        "decay_used_db": decay_db,
    }


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------

def silence_analysis(audio_mono: np.ndarray, sr: int,
                     threshold_db: float = -50) -> dict:
    """Analyze silence in audio.

    Returns:
        {mean_rms_db: float, silent_fraction: float, is_mostly_silent: bool}
    """
    rms = librosa.feature.rms(y=audio_mono, frame_length=2048, hop_length=512)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    mean_db = float(np.mean(rms_db))
    silent_frac = float(np.mean(rms_db < threshold_db))

    return {
        "mean_rms_db": round(mean_db, 2),
        "silent_fraction": round(silent_frac, 3),
        "is_mostly_silent": silent_frac > 0.8,
        "threshold_db": threshold_db,
    }


# ---------------------------------------------------------------------------
# Segment comparison (for comparative_test template)
# ---------------------------------------------------------------------------

def compare_segments(audio_mono: np.ndarray, sr: int,
                     seg1: tuple[float, float],
                     seg2: tuple[float, float]) -> dict:
    """Compare two audio segments on pitch, loudness, and spectral features.

    Args:
        seg1: (start_s, end_s) for first segment
        seg2: (start_s, end_s) for second segment

    Returns dict with delta measurements.
    """
    s1 = audio_mono[int(seg1[0] * sr):int(seg1[1] * sr)]
    s2 = audio_mono[int(seg2[0] * sr):int(seg2[1] * sr)]

    # Pitch
    p1 = _praat_pitch(s1, sr)
    p2 = _praat_pitch(s2, sr)

    def median_pitch(pitch_obj, seg_audio, sr_val):
        times = pitch_obj.xs()
        freqs = [pitch_obj.get_value_at_time(t) for t in times]
        valid = [f for f in freqs if f is not None and not np.isnan(f) and f > 0]
        return float(np.median(valid)) if valid else None

    pitch1 = median_pitch(p1, s1, sr)
    pitch2 = median_pitch(p2, s2, sr)

    # Loudness
    loud1 = extract_loudness(s1, sr)
    loud2 = extract_loudness(s2, sr)

    # Spectral centroid
    cent1 = float(np.mean(librosa.feature.spectral_centroid(y=s1, sr=sr)))
    cent2 = float(np.mean(librosa.feature.spectral_centroid(y=s2, sr=sr)))

    # extract_loudness already coerces -inf -> None. We just need to guard
    # the delta computation when either operand is None.
    l1 = loud1.get("lufs")
    l2 = loud2.get("lufs")
    delta_lufs = (round(l2 - l1, 2)
                  if (l1 is not None and l2 is not None) else None)

    result = {
        "seg1": {"start_s": seg1[0], "end_s": seg1[1]},
        "seg2": {"start_s": seg2[0], "end_s": seg2[1]},
        "pitch": {
            "seg1_median_hz": round(pitch1, 2) if pitch1 else None,
            "seg2_median_hz": round(pitch2, 2) if pitch2 else None,
            "delta_hz": round(pitch2 - pitch1, 2) if (pitch1 and pitch2) else None,
            "direction": ("higher" if pitch2 > pitch1 else "lower") if (pitch1 and pitch2) else None,
        },
        "loudness": {
            "seg1_lufs": l1,
            "seg2_lufs": l2,
            "delta_lufs": delta_lufs,
        },
        "spectral_centroid": {
            "seg1_hz": round(cent1, 2),
            "seg2_hz": round(cent2, 2),
            "delta_hz": round(cent2 - cent1, 2),
            "direction": "brighter" if cent2 > cent1 else "darker",
        },
    }
    return result


# ---------------------------------------------------------------------------
# Audio-visual alignment (AV PC: sync and causal ordering)
# ---------------------------------------------------------------------------

def av_align(audio_mono: np.ndarray, sr: int,
             visible_event_times_s: list[float],
             expected_delay_s: float = 0.0,
             tolerance_ms: float = 100.0) -> dict:
    """Cross-reference visible event times (from the agent's observation of the
    video) with detected audio onsets. Core primitive for AV PC sync / causal
    verdicts.

    Methodology:
        - Run onset detection on the audio track.
        - For each visible event time t_v, compute target audio time
          t_target = t_v + expected_delay_s.
        - Find the nearest audio onset t_a to t_target.
        - Report delta_ms = (t_a - t_target) * 1000.
        - `within_tolerance` = |delta_ms| <= tolerance_ms.
        - `causal_ok` = audio onset is not earlier than the visible cause
          (t_a >= t_v - tolerance) for any expected_delay_s >= 0.

    JND-calibrated default: 100 ms tolerance follows SonicBench's Weber-fraction
    analysis (60-90 ms reliable discrimination floor). Use a larger tolerance
    for long-delay causal chains (e.g., thunder after lightning).
    """
    onset_result = detect_onsets(audio_mono, sr)
    onset_times = onset_result["onsets"]

    alignments = []
    for t_v in visible_event_times_s:
        target = t_v + expected_delay_s
        if not onset_times:
            alignments.append({
                "t_v": round(float(t_v), 3),
                "target_t_a": round(float(target), 3),
                "nearest_onset_s": None,
                "delta_ms": None,
                "within_tolerance": False,
                "causal_ok": False,
            })
            continue
        nearest = min(onset_times, key=lambda t: abs(t - target))
        delta_ms = (nearest - target) * 1000.0
        within = abs(delta_ms) <= tolerance_ms
        causal_ok = (expected_delay_s <= 0) or (nearest >= t_v - tolerance_ms / 1000.0)
        alignments.append({
            "t_v": round(float(t_v), 3),
            "target_t_a": round(float(target), 3),
            "nearest_onset_s": round(float(nearest), 3),
            "delta_ms": round(float(delta_ms), 1),
            "within_tolerance": bool(within),
            "causal_ok": bool(causal_ok),
        })

    passed = sum(1 for a in alignments if a["within_tolerance"] and a["causal_ok"])
    return {
        "alignments": alignments,
        "tolerance_ms": float(tolerance_ms),
        "expected_delay_s": float(expected_delay_s),
        "passed_count": int(passed),
        "total": len(alignments),
        "all_pass": bool(passed == len(alignments) and len(alignments) > 0),
        "audio_onset_count": len(onset_times),
        "method": "librosa_onsets_vs_agent_visible_times",
    }


# ---------------------------------------------------------------------------
# High-level measurement dispatcher (used by orchestrator)
# ---------------------------------------------------------------------------

def measure_for_category(audio_mono: np.ndarray, audio_stereo: np.ndarray | None,
                         sr: int, category: str, onset_times: list[float],
                         transition_time: float | None = None) -> dict:
    """Select and run appropriate DSP measurements based on physics category.

    Args:
        category: e.g. "1" (pitch), "2" (loudness), "10" (spatial)
        onset_times: detected audio onsets
        transition_time: for comparative prompts, where the scene changes

    Returns a measurements dict.
    """
    measurements = {}

    # Always include basic stats
    measurements["onsets"] = {"times": onset_times, "count": len(onset_times)}
    measurements["silence"] = silence_analysis(audio_mono, sr)

    cat_num = int(category) if category.isdigit() else int(category.rstrip("abcdefgh"))

    if cat_num in (1,):  # Pitch & Frequency
        measurements["pitch_contour"] = pitch_contour(audio_mono, sr)
        if onset_times:
            measurements["pitch_at_onsets"] = pitch_at_onsets(audio_mono, sr, onset_times)

    elif cat_num in (2,):  # Loudness & Amplitude
        measurements["loudness_contour"] = loudness_contour(audio_mono, sr)
        measurements["rms_contour"] = rms_contour(audio_mono, sr)

    elif cat_num in (3,):  # Timbre & Material
        measurements["spectral"] = spectral_features(audio_mono, sr)

    elif cat_num in (4,):  # Medium & Propagation
        measurements["silence"] = silence_analysis(audio_mono, sr, threshold_db=-45)

    elif cat_num in (5,):  # Distance & Attenuation
        measurements["loudness_contour"] = loudness_contour(audio_mono, sr)
        measurements["spectral"] = spectral_features(audio_mono, sr)

    elif cat_num in (6,):  # Reverb
        measurements["rt60"] = estimate_rt60(audio_mono, sr)

    elif cat_num in (7,):  # Resonance
        measurements["pitch_contour"] = pitch_contour(audio_mono, sr)
        measurements["spectral"] = spectral_features(audio_mono, sr)

    elif cat_num in (8,):  # Absorption & Obstruction
        measurements["spectral"] = spectral_features(audio_mono, sr)

    elif cat_num in (9,):  # Temporal
        measurements["onsets_detailed"] = detect_onsets(audio_mono, sr)

    elif cat_num in (10,):  # Spatial
        if audio_stereo is not None:
            measurements["stereo_balance"] = stereo_balance(audio_stereo, sr)
            measurements["ild"] = ild_over_time(audio_stereo, sr)

    elif cat_num in (11,):  # Acoustic Source Fidelity
        measurements["spectral"] = spectral_features(audio_mono, sr)

    # For comparative prompts with a transition point, compare before/after
    if transition_time is not None and transition_time > 0.3:
        duration = len(audio_mono) / sr
        if transition_time < duration - 0.3:
            measurements["segment_comparison"] = compare_segments(
                audio_mono, sr,
                seg1=(0.0, transition_time),
                seg2=(transition_time, duration),
            )

    return measurements
