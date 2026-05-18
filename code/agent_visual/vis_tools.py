
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image


_TMP_ROOT = Path(tempfile.gettempdir()) / "phyavbench_vis"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _frame_dir_for(video_path: str) -> Path:
    stem = Path(video_path).stem
    d = _TMP_ROOT / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_frame_at_time(video_path: str, time_s: float) -> dict:
    """Extract a single full-resolution frame at time_s.

    Returns saved_path, mime_type, width, height, and the requested /
    actual timestamps. Cached by output filename so a second call at the
    same timestamp returns instantly.
    """
    safe_t = max(0.0, float(time_s))
    out_dir = _frame_dir_for(video_path)
    out_path = out_dir / f"frame_t{safe_t:07.3f}.png"
    if not out_path.exists():
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{safe_t}", "-i", str(video_path),
            "-frames:v", "1", str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    with Image.open(out_path) as img:
        w, h = img.size
    return {
        "saved_path":       str(out_path),
        "mime_type":        "image/png",
        "width":            w,
        "height":           h,
        "time_s_requested": safe_t,
        "time_s_actual":    safe_t,
    }


def crop_frame(frame_path: str, x: int, y: int,
               width: int, height: int) -> dict:
    """Crop [x, y, x+width, y+height] from frame_path.

    Bounding box is clamped to the frame; an empty crop returns an
    error dict instead of crashing the React loop.
    """
    with Image.open(frame_path) as img:
        W, H = img.size
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(W, int(x) + int(width))
        y1 = min(H, int(y) + int(height))
        if x1 <= x0 or y1 <= y0:
            return {"error":
                    f"empty crop: bbox=[{x},{y},{width},{height}], "
                    f"frame={W}x{H}"}
        crop = img.crop((x0, y0, x1, y1))
        out_dir = Path(frame_path).parent
        stem = Path(frame_path).stem
        crop_path = out_dir / f"{stem}_crop_{x0}_{y0}_{x1-x0}_{y1-y0}.png"
        crop.save(crop_path)
        cw, ch = crop.size
    return {
        "saved_path":   str(crop_path),
        "mime_type":    "image/png",
        "width":        cw,
        "height":       ch,
        "bbox_applied": [x0, y0, x1 - x0, y1 - y0],
        "source_frame": frame_path,
    }
