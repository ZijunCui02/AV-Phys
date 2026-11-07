"""Single-call multimodal-LLM evaluator for AV-Phys Bench.

Loads each (prompt, video) pair, sends the whole MP4 to the judge LLM
together with the prompt's rubric, and parses a structured per-statement
verdict back out.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import random
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional

from tqdm import tqdm
from pydantic import BaseModel, Field
import openai
from openai import OpenAI


# ---------------------------------------------------------------------------
# Defaults (override on the CLI or via environment).
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_DIR    = "videos"
DEFAULT_RUBRIC_DIR   = "rubrics"
DEFAULT_OUTPUT_ROOT  = "results/mllm"
DEFAULT_JUDGE_MODEL  = "Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_BASE_URL     = "http://localhost:8000/v1"
DEFAULT_GENERATORS   = ["Seedance-2.0", "Kling-3.0-Omni", "Veo-3.1",
                        "LTX-2.3", "Ovi", "JavisDiT++", "MagiHuman"]

TEMPERATURE      = 0
MAX_TOKENS       = 8192
SEED             = 42


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
# OpenAI-compatible client + retry-hardened generate.
# ---------------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    msg = str(exc).lower()
    return ("429" in msg or "rate limit" in msg or "quota" in msg
            or "resource_exhausted" in msg or "resource exhausted" in msg)


def _generate(client: OpenAI, judge_model: str,
              media_parts: list, prompt_text: str, response_schema=None,
              context: str = "", max_retries: int = 5):
    messages = [
        {"role": "user",
         "content": [*media_parts, {"type": "text", "text": prompt_text}]},
    ]
    kwargs: dict = {
        "model":       judge_model,
        "messages":    messages,
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "seed":        SEED,
    }
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
            response = client.chat.completions.create(**kwargs)
            text = ""
            if response.choices:
                text = response.choices[0].message.content or ""
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


def call_judge_with_video(client, judge_model, media_parts,
                          prompt_text, response_schema=None, context: str = ""):
    return _generate(client, judge_model, media_parts, prompt_text,
                     response_schema, context=context)


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

def process_single_prompt(client: OpenAI,
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
    media_parts = build_media_parts(video_path)
    prompt = build_prompt(rubric_data)
    raw_text, parsed = call_judge_with_video(
        client, judge_model, media_parts, prompt,
        response_schema=VERDICT_SCHEMA, context=ctx,
    )
    if not _coverage_ok(parsed, expected_ids):
        _log(f"[{ctx}] coverage incomplete, retrying once")
        stricter = prompt + (
            "\n\nIMPORTANT: per_statement must contain exactly one entry "
            "for EACH of these ids: " + ", ".join(expected_ids) + "."
        )
        raw_text, parsed = call_judge_with_video(
            client, judge_model, media_parts, stricter,
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
    p.add_argument("--base-url",         default=DEFAULT_BASE_URL,
                   help="OpenAI-compatible endpoint (vLLM server).")
    p.add_argument("--video-dir",        default=DEFAULT_VIDEO_DIR)
    p.add_argument("--rubric-dir",       default=DEFAULT_RUBRIC_DIR)
    p.add_argument("--output-root",      default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--generator-models", nargs="+", default=DEFAULT_GENERATORS)
    p.add_argument("--prompt-ids",       nargs="+", default=None)
    p.add_argument("--run-in-parallel",  action="store_true")
    p.add_argument("--max-workers",      type=int, default=8)
    return p.parse_args()


def _build_client(base_url: str) -> OpenAI:
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=api_key,
                  timeout=600.0, max_retries=0)


def main():
    args = parse_args()
    client = _build_client(args.base_url)

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
    print(f"Endpoint:           {args.base_url}")
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
