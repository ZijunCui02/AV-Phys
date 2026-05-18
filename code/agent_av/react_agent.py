"""Agent that watches a clip, calls audio (DSP) and visual tools in a ReAct
loop, then writes a structured per-statement verdict.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from .tools import TOOL_DECLARATIONS, TOOL_DISPATCH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict schema and helpers (kept here so the agent dir is self-contained).
# ---------------------------------------------------------------------------

class PerStatement(BaseModel):
    statement_id: str = Field(
        description='Exactly one of: "video_sa.objects", "video_sa.event", '
                    '"audio_sa.objects", "audio_sa.sound", or '
                    '"<video_pc|audio_pc|av_pc>.Statement_<n>".'
    )
    observation: str = Field(
        description="1-3 sentences of evidence drawn from the clip and any "
                    "DSP / visual measurements that support or refute the "
                    "statement."
    )
    verdict: Literal["Yes", "No"] = Field(
        description="Strictly 'Yes' or 'No'."
    )


class Verdict(BaseModel):
    object: str = Field(description="Short description of visible objects.")
    event: str = Field(description="Short description of the main event.")
    observations: str = Field(
        description="2-4 sentences noting any physics anomalies in either "
                    "modality."
    )
    per_statement: List[PerStatement] = Field(min_length=1)


VERDICT_SCHEMA = Verdict.model_json_schema()


def _log(msg: str) -> None:
    print(msg)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("429" in msg or "rate limit" in msg or "quota" in msg
            or "resource_exhausted" in msg or "resource exhausted" in msg)


def _expected_statement_ids(rubric_data: dict) -> list[str]:
    ids = ["video_sa.objects", "video_sa.event",
           "audio_sa.objects", "audio_sa.sound"]
    ks = rubric_data.get("key_standards", {})
    for aspect in ("video_pc", "audio_pc", "av_pc"):
        for i in range(1, len(ks.get(aspect, [])) + 1):
            ids.append(f"{aspect}.Statement_{i}")
    return ids


def _coverage_ok(parsed: dict, expected_ids: list[str]) -> bool:
    if not isinstance(parsed, dict):
        return False
    seen = {row.get("statement_id", "") for row in parsed.get("per_statement", [])
            if isinstance(row, dict)}
    return set(expected_ids).issubset(seen)


def aggregate(per_statement: list[dict]) -> dict:
    bits: dict[str, list[int]] = {
        "video_sa": [], "audio_sa": [],
        "video_pc": [], "audio_pc": [], "av_pc": [],
    }
    for row in per_statement:
        sid = row.get("statement_id", "")
        if "." not in sid:
            continue
        aspect = sid.split(".", 1)[0].lower()
        if aspect not in bits:
            continue
        v = 1 if str(row.get("verdict", "No")).strip().lower() == "yes" else 0
        bits[aspect].append(v)

    def AND(xs):
        if not xs:
            return None
        return 1 if all(b == 1 for b in xs) else 0

    video_sa = AND(bits["video_sa"])
    audio_sa = AND(bits["audio_sa"])
    video_pc = AND(bits["video_pc"])
    audio_pc = AND(bits["audio_pc"])
    av_pc    = AND(bits["av_pc"])
    sa  = (None if (video_sa is None or audio_sa is None)
           else (1 if (video_sa == 1 and audio_sa == 1) else 0))
    pcs = [v for v in (video_pc, audio_pc, av_pc) if v is not None]
    pc  = None if not pcs else (1 if all(v == 1 for v in pcs) else 0)
    both = (None if (sa is None or pc is None)
            else (1 if (sa == 1 and pc == 1) else 0))
    return {
        "video_sa": video_sa, "audio_sa": audio_sa,
        "video_pc": video_pc, "audio_pc": audio_pc, "av_pc": av_pc,
        "SA": sa, "PC": pc, "Both": both,
    }


# ---------------------------------------------------------------------------
# CAP frame and prompt builders.
# ---------------------------------------------------------------------------

CAP_FRAME = (
    "Suppose you are an expert in judging and evaluating the quality of "
    "AI-generated audio-video clips. This is a generated clip from a joint "
    "audio-video model rather than a recording of the real world, so it may "
    "be low quality, fuzzy, or inconsistent, and may not obey real-world "
    "physics. Do not rationalise artefacts as stylistic choices — treat any "
    "deviation from physical plausibility as a potential failure to report."
)


def _fmt_objects(objs) -> str:
    if not objs:
        return "[none]"
    if isinstance(objs, list):
        return ", ".join(f'"{o}"' for o in objs)
    return f'"{objs}"'


def _fmt_sound(sound) -> str:
    if isinstance(sound, list):
        return ", ".join(f'"{s}"' for s in sound)
    return f'"{sound}"'


def _build_sa_block(rubric_data: dict) -> str:
    bs = rubric_data.get("basic_standards", {})
    video = bs.get("video", {})
    audio = bs.get("audio", {})
    flags = rubric_data.get("flags", {})

    video_objects = _fmt_objects(video.get("objects", []))
    video_event   = video.get("event", "")
    audio_objects = _fmt_objects(audio.get("objects", []))
    audio_sound   = _fmt_sound(audio.get("sound", ""))

    lines = [
        f'  1. **video_sa.objects** — Are all of the following visually '
        f'present in the clip: {video_objects}? Answer Yes or No.',
        f'  2. **video_sa.event** — Is the event "{video_event}" visually '
        f'depicted in the clip? Answer Yes or No.',
    ]
    if flags.get("silence_expected", False):
        lines += [
            f'  3. **audio_sa.objects** — These sound source(s) '
            f'{audio_objects} would normally be audible if real-world '
            f'physics held. Answer Yes if they are appropriately '
            f'represented as such (typically silent here), No otherwise.',
            f'  4. **audio_sa.sound** — The clip is expected to be silent '
            f'during the depicted event. Answer Yes if the clip is '
            f'appropriately silent throughout the event with no audible '
            f'leak-through or incidental sound. Otherwise No.',
        ]
    else:
        lines += [
            f'  3. **audio_sa.objects** — Are the sound source(s) '
            f'{audio_objects} audible in the clip? Answer Yes or No.',
            f'  4. **audio_sa.sound** — Is the sound {audio_sound} clearly '
            f'audible in the clip? Answer Yes or No.',
        ]
    return "\n".join(lines)


def _build_pc_block(rubric_data: dict) -> str:
    ks = rubric_data.get("key_standards", {})
    lines = []
    for aspect in ("video_pc", "audio_pc", "av_pc"):
        for i, s in enumerate(ks.get(aspect, []), 1):
            lines.append(f"  - **{aspect}.Statement_{i}**: {s}")
    return "\n".join(lines)


def build_observation_prompt(rubric_data: dict) -> str:
    return (
        f"{CAP_FRAME}\n\n"
        f"Please tell me what is in this audio-video clip — what is "
        f"**visually depicted** AND what is **audible**. Include the visible "
        f"objects, the visible event, the audible sound source(s), the "
        f"audible signature (timbre, pitch, loudness, reverb, spatial "
        f"location), and any **physics phenomena in either modality** that "
        f"you observe.\n\n"
        f"Please be sure to include:\n"
        f"  - Visible objects in the scene.\n"
        f"  - The main visible event (action / motion / state change).\n"
        f"  - Audible sound source(s).\n"
        f"  - The audible signature.\n"
        f"  - Any physics phenomena in either modality (motion continuity, "
        f"sync between visible and audible events, spatial correspondence, "
        f"reverb, pitch / loudness changes, etc.)."
    )


def build_verdict_prompt(rubric_data: dict, prev_response: str) -> str:
    sa = _build_sa_block(rubric_data)
    pc = _build_pc_block(rubric_data)
    expected_ids = _expected_statement_ids(rubric_data)
    expected_ids_str = ", ".join(f"`{i}`" for i in expected_ids)

    return (
        f"Suppose you are an expert in summarization and finding answers. "
        f"Here is the text description from another large language model "
        f"about an AI-generated audio-video clip:\n\n"
        f"\"{prev_response}\"\n\n"
        f"Based on this description, please answer each of the following "
        f"questions with strictly \"Yes\" or \"No\".\n\n"
        f"### Basic Standards (Semantic Adherence)\n\n"
        f"{sa}\n\n"
        f"### Key Standards (Physical Commonsense)\n\n"
        f"Check whether each of the following physics statements is true of "
        f"the clip. Answer \"Yes\" if the statement is clearly true; \"No\" "
        f"if it is false, ambiguous, or only partially true.\n\n"
        f"{pc}\n\n"
        f"### Output\n\n"
        f"Return JSON with one entry in `per_statement` for every statement "
        f"id listed above ({expected_ids_str}). Each entry has "
        f"`statement_id`, `observation` (1-3 sentences citing the "
        f"description), and `verdict` (\"Yes\" or \"No\")."
    )


# ---------------------------------------------------------------------------
# Tool addendum appended to the observation prompt.
# ---------------------------------------------------------------------------

_TOOL_NAMES_BLOCK = (
    "- `dsp_detect_onsets` — audio onset timestamps\n"
    "- `dsp_pitch_contour` — F0 (Hz) over time\n"
    "- `dsp_pitch_at_onsets` — F0 at each detected onset, with overall direction\n"
    "- `dsp_loudness_contour` — LUFS over time\n"
    "- `dsp_spectral_features` — centroid / rolloff / bandwidth / ZCR (segment-scoped)\n"
    "- `dsp_compare_segments` — A/B comparison on pitch, loudness, centroid\n"
    "- `dsp_silence_analysis` — RMS / silent fraction\n"
    "- `dsp_estimate_rt60` — reverberation time (seconds)\n"
    "- `dsp_stereo_balance` — L/R balance and dominant side\n"
    "- `dsp_av_align` — for AV temporal questions: you supply visible event "
    "times, the tool returns the nearest audio onsets and offsets\n"
    "- `vis_frame_at_time` — extract a single full-resolution frame at a "
    "chosen timestamp; the frame is shown back to you so you can examine "
    "details too small to see in the embedded video\n"
    "- `vis_zoom_crop` — crop a region of a previously extracted frame; the "
    "crop is shown back to you for fine-grained inspection"
)

_TOOL_GUIDE_BLOCK = (
    "- Pitch / frequency → `dsp_pitch_at_onsets`, `dsp_pitch_contour`, "
    "`dsp_compare_segments`\n"
    "- Loudness / amplitude → `dsp_loudness_contour`, `dsp_compare_segments`\n"
    "- Timbre / material → `dsp_spectral_features`\n"
    "- Spatial / stereo → `dsp_stereo_balance`\n"
    "- Temporal sync / causal order → `dsp_av_align` "
    "(you supply the visible event times)\n"
    "- Reverb / room → `dsp_estimate_rt60`\n"
    "- Silence / vacuum → `dsp_silence_analysis`\n"
    "- Before / after comparison → `dsp_compare_segments`\n"
    "- Counting / fine visual attribute (e.g. number of clock hands, "
    "presence of a small element, identity drift across frames) → "
    "`vis_frame_at_time` then `vis_zoom_crop`\n"
    "- Visual event timing for AV sync → `vis_frame_at_time` to localise "
    "the event visually, then feed those times into `dsp_av_align`"
)

_TOOL_USAGE_RULE = (
    "If a Physical Commonsense (PC) statement targets a measurable physical "
    "quantity — pitch in Hz, loudness or decay in dB or seconds, "
    "reverberation time, stereo position, audio-visual onset alignment, "
    "silence in vacuum — you must call the relevant tool before producing "
    "the verdict for that statement. For statements that are purely "
    "qualitative (e.g. timbre matching a real-world source class), tool "
    "use is at your discretion.\n\n"
    "You may call multiple tools across multiple turns. Pass the path "
    "`{video_path}` to all tool calls.\n\n"
    "**Required minimum tool coverage for this clip**: before you produce "
    "any verdict, you must have called **at least one audio tool "
    "(`dsp_*`) and at least one visual tool (`vis_*`)**. The audio call "
    "should target a measurable acoustic quantity informative for the "
    "physical commonsense statements being judged; the visual call should "
    "either localise an event in time (`vis_frame_at_time`) or zoom into a "
    "region whose details matter for the verdict (`vis_zoom_crop` after "
    "`vis_frame_at_time`). Do not call tools whose output you will not "
    "actually use — the requirement is that visual and audio measurement "
    "evidence both contribute to the verdict, not that you fire off "
    "unrelated calls."
)


def _decorate_with_tools(prompt: str, video_path: str,
                         include_tool_guide: bool = True) -> str:
    parts = [
        prompt,
        "\n\n---\n\n## Audio + visual tools available\n\n"
        "You have access to the following tools that extract precise "
        "physical quantities from the audio track and let you inspect "
        "specific moments and regions of the video at full resolution:\n\n"
        + _TOOL_NAMES_BLOCK,
    ]
    if include_tool_guide:
        parts.append("\n\n## Tool selection guide\n\n" + _TOOL_GUIDE_BLOCK)
    parts.append("\n\n## Tool usage rule\n\n"
                 + _TOOL_USAGE_RULE.format(video_path=video_path))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Video bytes cache (one read per process per path).
# ---------------------------------------------------------------------------

_video_cache: dict[str, tuple[bytes, str]] = {}
_cache_lock = threading.Lock()


def upload_video(client: genai.Client, video_path: str) -> tuple[bytes, str]:
    del client  # unused
    with _cache_lock:
        cached = _video_cache.get(video_path)
    if cached is not None:
        return cached
    with open(video_path, "rb") as fh:
        data = fh.read()
    mime = "video/mp4"
    with _cache_lock:
        _video_cache[video_path] = (data, mime)
    return data, mime


# ---------------------------------------------------------------------------
# Retry-hardened generate_content.
# ---------------------------------------------------------------------------

def _generate_with_retry(client: genai.Client, model: str,
                         contents: list,
                         config: types.GenerateContentConfig,
                         context: str = "", max_retries: int = 5):
    last_failure = None
    last_was_rate_limit = False
    for attempt in range(1, max_retries + 1):
        last_was_rate_limit = False
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config,
            )
        except Exception as e:
            last_was_rate_limit = _is_rate_limit_error(e)
            last_failure = f"{'429' if last_was_rate_limit else 'api'}: {e}"
            tag = "RATE-LIMIT" if last_was_rate_limit else "API"
            _log(f"  [{context}] attempt {attempt}: {tag} {e}")
        if attempt < max_retries:
            base = 8.0 if last_was_rate_limit else 2.0
            backoff = base * (2 ** (attempt - 1))
            jitter = random.uniform(0, backoff * 0.25)
            time.sleep(backoff + jitter)
    _log(f"  [{context}] max retries reached ({last_failure})")
    return None


# ---------------------------------------------------------------------------
# ReAct loop.
# ---------------------------------------------------------------------------

def _has_function_calls(response) -> bool:
    if not response.candidates:
        return False
    for part in response.candidates[0].content.parts:
        if part.function_call:
            return True
    return False


def _get_function_calls(response) -> list:
    return [part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call]


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def _maybe_inject_image_part(client, result, context: str = ""):
    if not isinstance(result, dict):
        return None
    saved_path = result.get("saved_path")
    mime = result.get("mime_type", "")
    if not saved_path or not isinstance(mime, str) or not mime.startswith("image/"):
        return None
    if not os.path.exists(saved_path):
        _log(f"  [{context}] image-inject: missing path {saved_path}")
        return None
    try:
        with open(saved_path, "rb") as fh:
            data = fh.read()
    except Exception as e:
        _log(f"  [{context}] image-inject read failed: {e}")
        return None
    return types.Part(inline_data=types.Blob(data=data, mime_type=mime))


def _execute_tool(fc, video_path: str) -> dict:
    name = fc.name
    args = dict(fc.args) if fc.args else {}
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    if "video_path" not in args:
        args["video_path"] = video_path
    try:
        result = fn(**args)
    except Exception as e:
        return {"error": str(e)}
    return _sanitize_for_json(result)


def _run_react_loop(client: genai.Client,
                    agent_model: str,
                    video_bytes: bytes,
                    video_mime: str,
                    video_path: str,
                    system_text: str,
                    max_turns: int,
                    context: str) -> tuple[Optional[str], list, list]:
    contents: list = [
        types.Content(role="user", parts=[
            types.Part(inline_data=types.Blob(
                data=video_bytes,
                mime_type=video_mime)),
            types.Part(text=system_text),
        ]),
    ]
    config = types.GenerateContentConfig(
        tools=[TOOL_DECLARATIONS],
        temperature=0,
        media_resolution="MEDIA_RESOLUTION_HIGH",
        thinking_config=types.ThinkingConfig(thinking_budget=-1,
                                             include_thoughts=False),
        max_output_tokens=8192,
    )

    tool_trace: list = []
    last_text: Optional[str] = None
    response = None

    for turn in range(max_turns):
        response = _generate_with_retry(
            client, agent_model, contents, config,
            context=f"{context}#react.t{turn+1}",
        )
        if response is None or not response.candidates:
            return last_text, tool_trace, contents
        model_content = response.candidates[0].content
        contents.append(model_content)

        if _has_function_calls(response):
            fn_calls = _get_function_calls(response)
            fn_response_parts = []
            image_inject_parts = []
            for fc in fn_calls:
                result = _execute_tool(fc, video_path)
                args_clean = {k: v for k, v in
                              (dict(fc.args) if fc.args else {}).items()
                              if k != "video_path"}
                tool_trace.append({
                    "turn":   turn + 1,
                    "tool":   fc.name,
                    "args":   args_clean,
                    "result": result,
                })
                fn_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name, response=result)
                )
                inj = _maybe_inject_image_part(client, result,
                                                context=context)
                if inj is not None:
                    image_inject_parts.append(inj)
            contents.append(types.Content(role="user",
                                           parts=fn_response_parts))
            if image_inject_parts:
                contents.append(types.Content(role="user",
                                               parts=image_inject_parts))
        else:
            last_text = response.text or ""
            return last_text, tool_trace, contents

    if response is not None:
        try:
            last_text = response.text or ""
        except Exception:
            last_text = None
    _log(f"  [{context}] react loop hit max_turns={max_turns}")
    return last_text, tool_trace, contents


def _verdict_with_video(client: genai.Client,
                        agent_model: str,
                        video_bytes: bytes,
                        video_mime: str,
                        prompt_text: str,
                        context: str
                        ) -> tuple[Optional[str], Optional[dict]]:
    contents = [
        types.Content(role="user", parts=[
            types.Part(inline_data=types.Blob(
                data=video_bytes,
                mime_type=video_mime)),
            types.Part(text=prompt_text),
        ]),
    ]
    config = types.GenerateContentConfig(
        temperature=0,
        media_resolution="MEDIA_RESOLUTION_HIGH",
        thinking_config=types.ThinkingConfig(thinking_budget=-1,
                                             include_thoughts=False),
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_json_schema=VERDICT_SCHEMA,
    )
    response = _generate_with_retry(client, agent_model, contents, config,
                                    context=context)
    if response is None or not response.text:
        return None, None
    text = response.text
    try:
        return text, json.loads(text)
    except json.JSONDecodeError as e:
        _log(f"  [{context}] verdict JSON parse failed: {e}")
        return text, None


# ---------------------------------------------------------------------------
# Per-cell entry point.
# ---------------------------------------------------------------------------

def evaluate_one_cell(client: genai.Client,
                      agent_model: str,
                      prompt_id: str,
                      generator_model: str,
                      video_path: str,
                      rubric_data: dict,
                      output_root: str,
                      max_turns: int = 10,
                      include_tool_guide: bool = True,
                      skip_if_exists: bool = True
                      ) -> Optional[str]:
    ctx = f"{prompt_id}/{generator_model}"
    out_dir = Path(output_root) / agent_model / generator_model
    out_path = out_dir / f"{prompt_id}.json"
    if skip_if_exists and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("per_statement"):
                return str(out_path)
        except Exception:
            pass

    expected_ids = _expected_statement_ids(rubric_data)
    t0 = time.time()
    video_bytes, video_mime = upload_video(client, video_path)

    obs_prompt = build_observation_prompt(rubric_data)
    decorated = _decorate_with_tools(obs_prompt, video_path,
                                      include_tool_guide=include_tool_guide)
    last_text, tool_trace, _ = _run_react_loop(
        client, agent_model, video_bytes, video_mime, video_path, decorated,
        max_turns=max_turns, context=ctx,
    )
    description = last_text or ""
    if not last_text:
        _log(f"[{ctx}] react loop produced no final description")

    verdict_prompt = build_verdict_prompt(rubric_data, description)
    raw_text, parsed = _verdict_with_video(
        client, agent_model, video_bytes, video_mime, verdict_prompt,
        context=f"{ctx}#verdict",
    )
    if not _coverage_ok(parsed, expected_ids):
        _log(f"[{ctx}] coverage incomplete, retrying once")
        stricter = verdict_prompt + (
            "\n\nIMPORTANT: per_statement must contain exactly one entry "
            "for EACH of these ids: " + ", ".join(expected_ids) + "."
        )
        raw_text, parsed = _verdict_with_video(
            client, agent_model, video_bytes, video_mime, stricter,
            context=f"{ctx}#verdict-retry",
        )

    parse_error: Optional[str]
    if parsed is None:
        _log(f"[{ctx}] verdict parse failed after retries")
        parsed = {"object": "", "event": "", "observations": "",
                  "per_statement": []}
        parse_error = "verdict_parse_failed"
    elif not _coverage_ok(parsed, expected_ids):
        parse_error = "incomplete_statement_coverage"
    else:
        parse_error = None

    per_statement = parsed.get("per_statement", [])
    aggregated = aggregate(per_statement)
    elapsed = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "prompt_id":          prompt_id,
        "judge_model":        agent_model,
        "generator_model":    generator_model,
        "rubric":             rubric_data,
        "object":             parsed.get("object", ""),
        "event":              parsed.get("event", ""),
        "observations":       parsed.get("observations", ""),
        "per_statement":      per_statement,
        "aggregated":         aggregated,
        "description":        description,
        "raw_text":           raw_text,
        "tool_trace":         tool_trace,
        "agent_turns":        len(tool_trace) + 1,
        "include_tool_guide": include_tool_guide,
        "elapsed_s":          round(elapsed, 2),
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    }
    if parse_error:
        record["parse_error"] = parse_error

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return str(out_path)
