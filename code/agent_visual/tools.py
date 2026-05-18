
from __future__ import annotations

from google.genai import types

from . import vis_tools


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
