"""Single-call multimodal-LLM evaluator for AV-Phys Bench.

Loads each (prompt, video) pair, sends the whole MP4 to the judge LLM
together with the prompt's rubric, and parses a structured per-statement
verdict back out.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional

from tqdm import tqdm
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Defaults (override on the CLI or via environment).
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_DIR    = "videos"
DEFAULT_RUBRIC_DIR   = "rubrics"
DEFAULT_OUTPUT_ROOT  = "results/mllm"
DEFAULT_JUDGE_MODEL  = "gemini-3.1-pro-preview"
DEFAULT_GENERATORS   = ["Seedance-2.0", "Kling-3.0-Omni", "Veo-3.1",
                        "LTX-2.3", "Ovi", "JavisDiT++", "MagiHuman"]


# ---------------------------------------------------------------------------
# Pydantic schema for the structured per-statement verdict.
# ---------------------------------------------------------------------------

class PerStatement(BaseModel):
    statement_id: str = Field(
        description='Exactly one of: "video_sa.objects", "video_sa.event", '
                    '"audio_sa.objects", "audio_sa.sound", or '
                    '"<video_pc|audio_pc|av_pc>.Statement_<n>".'
    )
    verdict: Literal["Yes", "No"] = Field(
        description='Strictly "Yes" or "No".'
    )


class Verdict(BaseModel):
    per_statement: List[PerStatement] = Field(min_length=1)


VERDICT_SCHEMA = Verdict.model_json_schema()


def _log(msg: str) -> None:
    try:
        tqdm.write(msg)
    except Exception:
        print(msg)


# ---------------------------------------------------------------------------
# Rubric loading.
# ---------------------------------------------------------------------------

def load_rubric(prompt_id: str, rubric_dir: str) -> dict:
    path = Path(rubric_dir) / f"{prompt_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No rubric for {prompt_id}")
    return json.loads(path.read_text())


def list_available_prompts(rubric_dir: str) -> list[str]:
    return sorted(p.stem for p in Path(rubric_dir).glob("*.json"))


# ---------------------------------------------------------------------------
# Gemini client + retry-hardened generate.
# ---------------------------------------------------------------------------

_video_cache: dict[str, tuple[bytes, str]] = {}
_cache_lock = threading.Lock()


def upload_video(client: genai.Client, video_path: str) -> tuple[bytes, str]:
    """Read video bytes for inline upload."""
    del client
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


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("429" in msg or "rate limit" in msg or "quota" in msg
            or "resource_exhausted" in msg or "resource exhausted" in msg)


def _generate(client: genai.Client, judge_model: str,
              parts: list[types.Part], response_schema=None,
              context: str = "", max_retries: int = 5):
    config = types.GenerateContentConfig(
        temperature=0,
        media_resolution="MEDIA_RESOLUTION_HIGH",
        thinking_config=types.ThinkingConfig(thinking_budget=-1,
                                             include_thoughts=False),
        max_output_tokens=8192,
    )
    if response_schema is not None:
        config.response_mime_type = "application/json"
        config.response_json_schema = response_schema

    contents = [types.Content(role="user", parts=parts)]
    last_failure = None
    last_was_rate_limit = False

    for attempt in range(1, max_retries + 1):
        last_was_rate_limit = False
        try:
            response = client.models.generate_content(
                model=judge_model, contents=contents, config=config,
            )
            text = response.text or ""
            if not text:
                last_failure = "empty_response"
                _log(f"  [{context}] attempt {attempt}: empty response")
            else:
                if response_schema is None:
                    return text, None
                try:
                    parsed = json.loads(text)
                    return text, parsed
                except json.JSONDecodeError as e:
                    last_failure = f"json_decode: {e}"
                    _log(f"  [{context}] attempt {attempt}: JSON parse failed: {e}")
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
    return None, None


def call_judge_with_video(client, judge_model, video_bytes, video_mime,
                          prompt_text, response_schema=None, context: str = ""):
    parts = [
        types.Part(inline_data=types.Blob(data=video_bytes,
                                          mime_type=video_mime)),
        types.Part(text=prompt_text),
    ]
    return _generate(client, judge_model, parts, response_schema, context=context)


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------

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


def _expected_statement_ids(rubric_data: dict) -> list[str]:
    ids = ["video_sa.objects", "video_sa.event",
           "audio_sa.objects", "audio_sa.sound"]
    ks = rubric_data.get("key_standards", {})
    for aspect in ("video_pc", "audio_pc", "av_pc"):
        for i in range(1, len(ks.get(aspect, [])) + 1):
            ids.append(f"{aspect}.Statement_{i}")
    return ids


def build_prompt(rubric_data: dict) -> str:
    sa = _build_sa_block(rubric_data)
    pc = _build_pc_block(rubric_data)
    expected_ids = _expected_statement_ids(rubric_data)
    expected_ids_str = ", ".join(f"`{i}`" for i in expected_ids)
    statements = sa + "\n" + pc

    return (
        f'Watch and listen to the clip. For each statement below, return '
        f'verdict "Yes" or "No".\n\n'
        f'{statements}\n\n'
        f'Return JSON with one entry in `per_statement` for each statement id '
        f'({expected_ids_str}); each entry has `statement_id` and `verdict`.'
    )


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------

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


def _coverage_ok(parsed: dict, expected_ids: list[str]) -> bool:
    if not isinstance(parsed, dict):
        return False
    seen = {row.get("statement_id", "") for row in parsed.get("per_statement", [])
            if isinstance(row, dict)}
    return set(expected_ids).issubset(seen)


# ---------------------------------------------------------------------------
# Per-prompt processing.
# ---------------------------------------------------------------------------

def process_single_prompt(client: genai.Client,
                          judge_model: str,
                          prompt_id: str,
                          generator_model: str,
                          video_path: str,
                          rubric_data: dict,
                          output_root: str,
                          skip_if_exists: bool = True) -> Optional[str]:
    ctx = f"{prompt_id}/{generator_model}"
    out_dir = Path(output_root) / judge_model / generator_model
    out_path = out_dir / f"{prompt_id}.json"
    if skip_if_exists and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("per_statement"):
                return str(out_path)
        except Exception:
            pass

    expected_ids = _expected_statement_ids(rubric_data)
    video_bytes, video_mime = upload_video(client, video_path)
    prompt = build_prompt(rubric_data)
    raw_text, parsed = call_judge_with_video(
        client, judge_model, video_bytes, video_mime, prompt,
        response_schema=VERDICT_SCHEMA, context=ctx,
    )
    if not _coverage_ok(parsed, expected_ids):
        _log(f"[{ctx}] coverage incomplete, retrying once")
        stricter = prompt + (
            "\n\nIMPORTANT: per_statement must contain exactly one entry "
            "for EACH of these ids: " + ", ".join(expected_ids) + "."
        )
        raw_text, parsed = call_judge_with_video(
            client, judge_model, video_bytes, video_mime, stricter,
            response_schema=VERDICT_SCHEMA, context=f"{ctx}#retry",
        )

    parse_error: Optional[str]
    if parsed is None:
        _log(f"[{ctx}] verdict parse failed after retries")
        parsed = {"per_statement": []}
        parse_error = "verdict_parse_failed"
    elif not _coverage_ok(parsed, expected_ids):
        parse_error = "incomplete_statement_coverage"
    else:
        parse_error = None

    per_statement = parsed.get("per_statement", [])
    aggregated    = aggregate(per_statement)

    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "prompt_id":       prompt_id,
        "judge_model":     judge_model,
        "generator_model": generator_model,
        "rubric":          rubric_data,
        "per_statement":   per_statement,
        "aggregated":      aggregated,
        "raw_text":        raw_text,
    }
    if parse_error:
        record["parse_error"] = parse_error

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return str(out_path)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MLLM evaluator for AV-Phys Bench.")
    p.add_argument("--judge-model",      default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--video-dir",        default=DEFAULT_VIDEO_DIR)
    p.add_argument("--rubric-dir",       default=DEFAULT_RUBRIC_DIR)
    p.add_argument("--output-root",      default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--generator-models", nargs="+", default=DEFAULT_GENERATORS)
    p.add_argument("--prompt-ids",       nargs="+", default=None)
    p.add_argument("--run-in-parallel",  action="store_true")
    p.add_argument("--max-workers",      type=int, default=8)
    return p.parse_args()


def _build_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY for the AI Studio API."
        )
    return genai.Client(api_key=api_key)


def main():
    args = parse_args()
    client = _build_client()

    if args.prompt_ids:
        prompt_ids = list(args.prompt_ids)
    else:
        prompt_ids = list_available_prompts(args.rubric_dir)

    jobs: list[tuple[str, str, str, dict]] = []
    for pid in prompt_ids:
        try:
            rubric_data = load_rubric(pid, args.rubric_dir)
        except FileNotFoundError as e:
            print(f"  skip {pid}: {e}")
            continue
        for gen in args.generator_models:
            vp = Path(args.video_dir) / gen / f"{pid}.mp4"
            if vp.exists():
                jobs.append((pid, gen, str(vp), rubric_data))

    print(f"Judge model:        {args.judge_model}")
    print(f"Generator models:   {', '.join(args.generator_models)}")
    print(f"Total jobs:         {len(jobs)}")
    if not jobs:
        return

    if args.run_in_parallel:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [
                ex.submit(process_single_prompt,
                          client, args.judge_model, pid, gen, vp, rubric,
                          args.output_root)
                for pid, gen, vp, rubric in jobs
            ]
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc="evaluating"):
                try:
                    out = f.result()
                    if out:
                        _log(f"  saved: {out}")
                except Exception as e:
                    _log(f"  error: {e}")
    else:
        for pid, gen, vp, rubric in tqdm(jobs, desc="evaluating"):
            try:
                out = process_single_prompt(
                    client, args.judge_model, pid, gen, vp, rubric,
                    args.output_root,
                )
                if out:
                    _log(f"  saved: {out}")
            except Exception as e:
                _log(f"  error: {e}")


if __name__ == "__main__":
    main()
