from __future__ import annotations

from . import vis_tools


# ---------------------------------------------------------------------------
# Visual tool implementations. These call vis_tools.py and return paths;
# the React loop in react_agent.py detects `saved_path` + image mime and
# re-injects the image as an image_url part so the model can see it on the
# next turn.
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
# OpenAI function-calling schemas.
# ---------------------------------------------------------------------------


def _str(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _num(desc: str) -> dict:
    return {"type": "number", "description": desc}


def _fn(name: str, description: str, properties: dict,
        required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_DECLARATIONS = [

    _fn("vis_frame_at_time",
        "Extract a single full-resolution frame from the video at the "
        "given time in seconds, save to disk, and return its path together "
        "with width and height. Use this when you need to inspect a "
        "specific moment more closely than the embedded video allows. The "
        "returned image is automatically attached to the next conversation "
        "turn so you can see it directly.",
        {
            "video_path": _str("Path to the video file"),
            "time_s":     _num("Timestamp in seconds at which to extract "
                               "the frame."),
        },
        ["video_path", "time_s"]),

    _fn("vis_zoom_crop",
        "Crop a rectangular region from a frame previously extracted by "
        "vis_frame_at_time. Provide the frame_path you got back from that "
        "tool, plus the bounding box in pixel coordinates of that frame: "
        "x and y are the top-left corner, width and height are the size "
        "of the crop. The cropped image is automatically attached to the "
        "next conversation turn so you can see it.",
        {
            "video_path": _str("Path to the video file"),
            "frame_path": _str("Path of the frame previously extracted by "
                               "vis_frame_at_time."),
            "x":          _num("Left edge of the crop, in pixels."),
            "y":          _num("Top edge of the crop, in pixels."),
            "width":      _num("Crop width, in pixels."),
            "height":     _num("Crop height, in pixels."),
        },
        ["video_path", "frame_path", "x", "y", "width", "height"]),

]
