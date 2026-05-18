
from __future__ import annotations

import functools

from google.genai import types

from . import dsp_tools
from . import vis_tools

# ---------------------------------------------------------------------------
# Audio cache — avoid re-extracting audio on every tool call.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _audio_mono(video_path: str) -> tuple:
    audio, sr = dsp_tools.extract_audio_mono(video_path, sr=48000)
    return audio, sr


@functools.lru_cache(maxsize=16)
def _audio_stereo(video_path: str) -> tuple:
    audio, sr = dsp_tools.extract_audio(video_path, sr=48000, mono=False)
    return audio, sr


# ---------------------------------------------------------------------------
# Tool implementations (thin wrappers over dsp_tools).
# ---------------------------------------------------------------------------

def dsp_detect_onsets(video_path: str) -> dict:
    audio, sr = _audio_mono(video_path)
    return dsp_tools.detect_onsets(audio, sr)


def dsp_pitch_at_onsets(video_path: str) -> dict:
    audio, sr = _audio_mono(video_path)
    onset_result = dsp_tools.detect_onsets(audio, sr)
    return dsp_tools.pitch_at_onsets(audio, sr, onset_result["onsets"])


def dsp_pitch_contour(video_path: str) -> dict:
    audio, sr = _audio_mono(video_path)
    return dsp_tools.pitch_contour(audio, sr)


def dsp_loudness_contour(video_path: str) -> dict:
    audio, sr = _audio_mono(video_path)
    return dsp_tools.loudness_contour(audio, sr)


def dsp_spectral_features(video_path: str, start_s: float = 0.0,
                          end_s: float = -1.0) -> dict:
    audio, sr = _audio_mono(video_path)
    end = None if end_s < 0 else end_s
    start = None if start_s <= 0 else start_s
    return dsp_tools.spectral_features(audio, sr, start, end)


def dsp_compare_segments(video_path: str, seg1_start: float, seg1_end: float,
                         seg2_start: float, seg2_end: float) -> dict:
    audio, sr = _audio_mono(video_path)
    return dsp_tools.compare_segments(audio, sr, (seg1_start, seg1_end),
                                       (seg2_start, seg2_end))


def dsp_silence_analysis(video_path: str) -> dict:
    audio, sr = _audio_mono(video_path)
    return dsp_tools.silence_analysis(audio, sr)


def dsp_estimate_rt60(video_path: str) -> dict:
    audio, sr = _audio_mono(video_path)
    return dsp_tools.estimate_rt60(audio, sr)


def dsp_stereo_balance(video_path: str) -> dict:
    audio, sr = _audio_stereo(video_path)
    return dsp_tools.stereo_balance(audio, sr)


def dsp_av_align(video_path: str, visible_event_times_s: list[float],
                 expected_delay_s: float = 0.0,
                 tolerance_ms: float = 100.0) -> dict:
    audio, sr = _audio_mono(video_path)
    return dsp_tools.av_align(
        audio, sr,
        list(visible_event_times_s),
        float(expected_delay_s),
        float(tolerance_ms),
    )


# ---------------------------------------------------------------------------
# Visual tool implementations. These call vis_tools.py and return paths;
# the React loop in react_agent.py detects `saved_path` + image mime and
# uploads the image to the Files API so the model can see it on the next
# turn.
# ---------------------------------------------------------------------------

def vis_frame_at_time(video_path: str, time_s: float) -> dict:
    return vis_tools.extract_frame_at_time(video_path, float(time_s))


def vis_zoom_crop(video_path: str, frame_path: str,
                  x: int, y: int, width: int, height: int) -> dict:
    # video_path is required by the dispatcher contract (auto-injected by
    # _execute_tool) but is unused here — the crop operates on frame_path.
    return vis_tools.crop_frame(str(frame_path), int(x), int(y),
                                 int(width), int(height))


# ---------------------------------------------------------------------------
# Dispatch table: name -> callable
# ---------------------------------------------------------------------------

TOOL_DISPATCH: dict[str, callable] = {
    "dsp_detect_onsets":      dsp_detect_onsets,
    "dsp_pitch_at_onsets":    dsp_pitch_at_onsets,
    "dsp_pitch_contour":      dsp_pitch_contour,
    "dsp_loudness_contour":   dsp_loudness_contour,
    "dsp_spectral_features":  dsp_spectral_features,
    "dsp_compare_segments":   dsp_compare_segments,
    "dsp_silence_analysis":   dsp_silence_analysis,
    "dsp_estimate_rt60":      dsp_estimate_rt60,
    "dsp_stereo_balance":     dsp_stereo_balance,
    "dsp_av_align":           dsp_av_align,
    "vis_frame_at_time":      vis_frame_at_time,
    "vis_zoom_crop":          vis_zoom_crop,
}

# ---------------------------------------------------------------------------
# Gemini FunctionDeclarations
# ---------------------------------------------------------------------------

_S = types.Schema  # shorthand


def _str(desc: str) -> _S:
    return _S(type="STRING", description=desc)


def _num(desc: str) -> _S:
    return _S(type="NUMBER", description=desc)


TOOL_DECLARATIONS = types.Tool(function_declarations=[

    types.FunctionDeclaration(
        name="dsp_detect_onsets",
        description="Detect audio onset timestamps. Returns the list of "
                    "onset times in seconds and the total count.",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
        }, required=["video_path"]),
    ),

    types.FunctionDeclaration(
        name="dsp_pitch_at_onsets",
        description="Detect onsets, then extract the fundamental frequency "
                    "(F0) in Hz at each onset. Returns the list of "
                    "(time, Hz) pairs and the overall direction "
                    "(ascending / descending / non_monotonic) when at least "
                    "two voiced onsets are found.",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
        }, required=["video_path"]),
    ),

    types.FunctionDeclaration(
        name="dsp_pitch_contour",
        description="Extract the full pitch contour (F0 over time) for the "
                    "entire audio track. Returns per-frame times and "
                    "frequencies in Hz (null for unvoiced frames), the "
                    "voiced fraction, and the mean and median Hz over voiced "
                    "frames.",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
        }, required=["video_path"]),
    ),

    types.FunctionDeclaration(
        name="dsp_loudness_contour",
        description="Compute integrated LUFS loudness over time using a "
                    "sliding window. Returns the time and LUFS value at each "
                    "window centre.",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
        }, required=["video_path"]),
    ),

    types.FunctionDeclaration(
        name="dsp_spectral_features",
        description="Extract spectral features over an audio segment: "
                    "centroid, rolloff, bandwidth, and zero-crossing rate. "
                    "Returns mean and std (Hz) for each feature. Use "
                    "start_s / end_s to scope to a segment; defaults span "
                    "the whole clip.",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
            "start_s": _num("Segment start time in seconds. Default 0."),
            "end_s": _num("Segment end time in seconds. "
                          "Use -1 (default) for end of clip."),
        }, required=["video_path"]),
    ),

    types.FunctionDeclaration(
        name="dsp_compare_segments",
        description="Compare two audio segments. Returns delta_hz for "
                    "median pitch, delta_lufs for integrated loudness, and "
                    "delta for spectral centroid, with the direction of "
                    "each change.",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
            "seg1_start": _num("First segment start time in seconds."),
            "seg1_end":   _num("First segment end time in seconds."),
            "seg2_start": _num("Second segment start time in seconds."),
            "seg2_end":   _num("Second segment end time in seconds."),
        }, required=["video_path", "seg1_start", "seg1_end",
                     "seg2_start", "seg2_end"]),
    ),

    types.FunctionDeclaration(
        name="dsp_silence_analysis",
        description="Analyse silence in the audio track. Returns mean RMS "
                    "in dB, the fraction of frames below the silence "
                    "threshold, and a boolean flag for whether the clip is "
                    "mostly silent.",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
        }, required=["video_path"]),
    ),

    types.FunctionDeclaration(
        name="dsp_estimate_rt60",
        description="Estimate RT60 (reverberation time) in seconds via "
                    "Schroeder backward integration on the loudest portion "
                    "of the clip. May return null if the decay is too "
                    "short to estimate.",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
        }, required=["video_path"]),
    ),

    types.FunctionDeclaration(
        name="dsp_stereo_balance",
        description="Analyse left-right stereo channel energy balance. "
                    "Returns the windowed balance trace over time, the mean "
                    "balance (-1 = entirely left, +1 = entirely right), and "
                    "a dominant_side label (left / center / right).",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
        }, required=["video_path"]),
    ),

    types.FunctionDeclaration(
        name="dsp_av_align",
        description="Audio-visual temporal alignment check. You provide "
                    "the timestamps of visible events you observed in the "
                    "clip; this tool detects audio onsets and, for each "
                    "visible event, returns the nearest audio onset, the "
                    "offset in milliseconds, whether it falls within the "
                    "tolerance, and whether causal ordering holds (audio "
                    "not earlier than the visible cause). Use "
                    "expected_delay_s = 0 for synchronous events; > 0 for "
                    "causal chains.",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
            "visible_event_times_s": _S(
                type="ARRAY",
                items=_num("Seconds"),
                description="Timestamps (seconds) of visible events you "
                            "observed in the clip. Provide one entry per "
                            "event you want to check.",
            ),
            "expected_delay_s": _num("Expected audio-after-video delay in "
                                     "seconds. 0 for synchronous, > 0 for "
                                     "causal chains. Default 0."),
            "tolerance_ms": _num("Tolerance in milliseconds for the "
                                  "within-tolerance flag. Default 100."),
        }, required=["video_path", "visible_event_times_s"]),
    ),

    types.FunctionDeclaration(
        name="vis_frame_at_time",
        description="Extract a single full-resolution frame from the video "
                    "at the given time in seconds, save to disk, and return "
                    "its path together with width and height. Use this when "
                    "you need to inspect a specific moment more closely than "
                    "the embedded video allows. The returned image is "
                    "automatically attached to the next conversation turn so "
                    "you can see it directly.",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
            "time_s":     _num("Timestamp in seconds at which to extract "
                                "the frame."),
        }, required=["video_path", "time_s"]),
    ),

    types.FunctionDeclaration(
        name="vis_zoom_crop",
        description="Crop a rectangular region from a frame previously "
                    "extracted by vis_frame_at_time. Provide the frame_path "
                    "you got back from that tool, plus the bounding box in "
                    "pixel coordinates of that frame: x and y are the "
                    "top-left corner, width and height are the size of the "
                    "crop. The cropped image is automatically attached to "
                    "the next conversation turn so you can see it.",
        parameters=_S(type="OBJECT", properties={
            "video_path": _str("Path to the video file"),
            "frame_path": _str("Path of the frame previously extracted by "
                                "vis_frame_at_time."),
            "x":          _num("Left edge of the crop, in pixels."),
            "y":          _num("Top edge of the crop, in pixels."),
            "width":      _num("Crop width, in pixels."),
            "height":     _num("Crop height, in pixels."),
        }, required=["video_path", "frame_path",
                     "x", "y", "width", "height"]),
    ),

])
