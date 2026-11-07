"""Agent that watches a clip, calls visual tools in a ReAct loop, then writes
a structured per-statement verdict.
"""

from __future__ import annotations

import base64
import collections
import json
import logging
import math
import os
import random
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
import openai
from openai import OpenAI

from .tools import TOOL_DECLARATIONS, TOOL_DISPATCH

logger = logging.getLogger(__name__)

TEMPERATURE      = 0
MAX_TOKENS       = 8192
SEED             = 42


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
    if isinstance(exc, openai.RateLimitError):
        return True
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
    "- `vis_frame_at_time` — extract a single full-resolution frame at a "
    "chosen timestamp; the frame is shown back to you so you can examine "
    "details too small to see in the embedded video\n"
    "- `vis_zoom_crop` — crop a region of a previously extracted frame; the "
    "crop is shown back to you for fine-grained inspection"
)

_TOOL_GUIDE_BLOCK = (
    "- Counting / fine visual attribute (e.g. number of clock hands, "
    "presence of a small element, identity drift across frames) → "
    "`vis_frame_at_time` then `vis_zoom_crop`\n"
    "- Visual event localisation in time → `vis_frame_at_time` at the "
    "moment(s) you want to inspect more closely\n"
    "- Cross-frame attribute consistency (e.g. an object's appearance "
    "drifts between frames) → `vis_frame_at_time` at multiple timestamps, "
    "compare visually"
)

_TOOL_USAGE_RULE = (
    "If a Physical Commonsense (PC) statement turns on a fine visual "
    "detail — object count, attribute drift across frames, presence of a "
    "small element, geometry that may be hard to see at the embedded "
    "resolution — you must call the relevant visual tool before producing "
    "the verdict for that statement. For statements that are purely "
    "qualitative at scene level, tool use is at your discretion.\n\n"
    "You may call multiple tools across multiple turns. Pass the path "
    "`{video_path}` to all tool calls.\n\n"
    "**Required minimum tool coverage for this clip**: before you produce "
    "any verdict, you must have called **at least one visual tool "
    "(`vis_*`)**. The visual call should either localise an event in time "
    "(`vis_frame_at_time`) or zoom into a region whose details matter for "
    "the verdict (`vis_zoom_crop` after `vis_frame_at_time`). Do not call "
    "tools whose output you will not actually use — the requirement is "
    "that visual measurement evidence contributes to the verdict, not "
    "that you fire off unrelated calls."
)


def _decorate_with_tools(prompt: str, video_path: str,
                         include_tool_guide: bool = True) -> str:
    parts = [
        prompt,
        "\n\n---\n\n## Visual tools available\n\n"
        "You have access to the following tools that let you inspect "
        "specific moments and regions of the video at full resolution:\n\n"
        + _TOOL_NAMES_BLOCK,
    ]
    if include_tool_guide:
        parts.append("\n\n## Tool selection guide\n\n" + _TOOL_GUIDE_BLOCK)
    parts.append("\n\n## Tool usage rule\n\n"
                 + _TOOL_USAGE_RULE.format(video_path=video_path))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Media cache (one encode per process per path, bounded).
# ---------------------------------------------------------------------------

_media_cache: "collections.OrderedDict[str, list]" = collections.OrderedDict()
_cache_lock = threading.Lock()
_MEDIA_CACHE_MAX = 64


def _extract_audio_wav_bytes(video_path: str) -> bytes:
    """Demux the audio track to 16 kHz mono WAV."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
             "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp_path],
            check=True, capture_output=True,
        )
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def build_media_parts(video_path: str) -> list:
    with _cache_lock:
        cached = _media_cache.get(video_path)
        if cached is not None:
            _media_cache.move_to_end(video_path)
            return cached

    with open(video_path, "rb") as fh:
        video_b64 = base64.b64encode(fh.read()).decode("ascii")
    wav_b64 = base64.b64encode(
        _extract_audio_wav_bytes(video_path)).decode("ascii")
    parts = [
        {"type": "video_url",
         "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
        {"type": "audio_url",
         "audio_url": {"url": f"data:audio/wav;base64,{wav_b64}"}},
    ]

    with _cache_lock:
        _media_cache[video_path] = parts
        while len(_media_cache) > _MEDIA_CACHE_MAX:
            _media_cache.popitem(last=False)
    return parts


# ---------------------------------------------------------------------------
# Retry-hardened chat completion.
# ---------------------------------------------------------------------------

def _chat_with_retry(client: OpenAI, model: str, messages: list,
                     tools: Optional[list] = None,
                     response_schema: Optional[dict] = None,
                     context: str = "", max_retries: int = 5):
    kwargs: dict = {
        "model":       model,
        "messages":    messages,
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "seed":        SEED,
    }
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if response_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "Verdict", "schema": response_schema},
        }

    last_failure = None
    last_was_rate_limit = False
    for attempt in range(1, max_retries + 1):
        last_was_rate_limit = False
        try:
            return client.chat.completions.create(**kwargs)
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

def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def _maybe_image_part(result, context: str = ""):
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
            data = base64.b64encode(fh.read()).decode("ascii")
    except Exception as e:
        _log(f"  [{context}] image-inject read failed: {e}")
        return None
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}"}}


def _execute_tool(name: str, args: dict, video_path: str) -> dict:
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


def _run_react_loop(client: OpenAI,
                    agent_model: str,
                    media_parts: list,
                    video_path: str,
                    system_text: str,
                    max_turns: int,
                    context: str) -> tuple[Optional[str], list, list]:
    messages: list = [
        {"role": "user",
         "content": [*media_parts, {"type": "text", "text": system_text}]},
    ]

    tool_trace: list = []
    last_text: Optional[str] = None
    msg = None

    for turn in range(max_turns):
        response = _chat_with_retry(
            client, agent_model, messages, tools=TOOL_DECLARATIONS,
            context=f"{context}#react.t{turn+1}",
        )
        if response is None or not response.choices:
            return last_text, tool_trace, messages
        msg = response.choices[0].message

        assistant_entry: dict = {"role": "assistant",
                                 "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if msg.tool_calls:
            image_inject_parts = []
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments not an object")
                except Exception as e:
                    args = {}
                    result = {"error": f"unparseable tool arguments: {e}"}
                else:
                    result = _execute_tool(name, args, video_path)
                args_clean = {k: v for k, v in args.items()
                              if k != "video_path"}
                tool_trace.append({
                    "turn":   turn + 1,
                    "tool":   name,
                    "args":   args_clean,
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
                inj = _maybe_image_part(result, context=context)
                if inj is not None:
                    image_inject_parts.append(inj)
            if image_inject_parts:
                messages.append({"role": "user",
                                 "content": image_inject_parts})
        else:
            last_text = msg.content or ""
            return last_text, tool_trace, messages

    if msg is not None:
        last_text = msg.content or ""
    _log(f"  [{context}] react loop hit max_turns={max_turns}")
    return last_text, tool_trace, messages


def _verdict_with_video(client: OpenAI,
                        agent_model: str,
                        media_parts: list,
                        prompt_text: str,
                        context: str
                        ) -> tuple[Optional[str], Optional[dict]]:
    messages = [
        {"role": "user",
         "content": [*media_parts, {"type": "text", "text": prompt_text}]},
    ]
    response = _chat_with_retry(client, agent_model, messages,
                                response_schema=VERDICT_SCHEMA,
                                context=context)
    if response is None or not response.choices:
        return None, None
    text = response.choices[0].message.content or ""
    if not text:
        return None, None
    try:
        return text, json.loads(text)
    except json.JSONDecodeError as e:
        _log(f"  [{context}] verdict JSON parse failed: {e}")
        return text, None


# ---------------------------------------------------------------------------
# Per-cell entry point.
# ---------------------------------------------------------------------------

def evaluate_one_cell(client: OpenAI,
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
    media_parts = build_media_parts(video_path)

    obs_prompt = build_observation_prompt(rubric_data)
    decorated = _decorate_with_tools(obs_prompt, video_path,
                                      include_tool_guide=include_tool_guide)
    last_text, tool_trace, _ = _run_react_loop(
        client, agent_model, media_parts, video_path, decorated,
        max_turns=max_turns, context=ctx,
    )
    description = last_text or ""
    if not last_text:
        _log(f"[{ctx}] react loop produced no final description")

    verdict_prompt = build_verdict_prompt(rubric_data, description)
    raw_text, parsed = _verdict_with_video(
        client, agent_model, media_parts, verdict_prompt,
        context=f"{ctx}#verdict",
    )
    if not _coverage_ok(parsed, expected_ids):
        _log(f"[{ctx}] coverage incomplete, retrying once")
        stricter = verdict_prompt + (
            "\n\nIMPORTANT: per_statement must contain exactly one entry "
            "for EACH of these ids: " + ", ".join(expected_ids) + "."
        )
        raw_text, parsed = _verdict_with_video(
            client, agent_model, media_parts, stricter,
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
