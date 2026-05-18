"""Sweep driver for the audio + visual tools agent.

Iterates (prompt_id, generator_model) pairs, calls evaluate_one_cell on
each, and writes the resulting JSON under output_root/{agent}/{generator}.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from google import genai

from .react_agent import evaluate_one_cell, _log


DEFAULT_VIDEO_DIR    = "videos"
DEFAULT_RUBRIC_DIR   = "rubrics"
DEFAULT_OUTPUT_ROOT  = "results/agent_av"
DEFAULT_AGENT_MODEL  = "gemini-3.1-pro-preview"
DEFAULT_GENERATORS   = ["Seedance-2.0", "Kling-3.0-Omni", "Veo-3.1",
                        "LTX-2.3", "Ovi", "JavisDiT++", "MagiHuman"]


def list_available_prompts(rubric_dir: str) -> list[str]:
    return sorted(p.stem for p in Path(rubric_dir).glob("*.json"))


def load_rubric(prompt_id: str, rubric_dir: str) -> dict:
    path = Path(rubric_dir) / f"{prompt_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No rubric for {prompt_id}")
    return json.loads(path.read_text())


def parse_args():
    p = argparse.ArgumentParser(description="Audio + visual tools agent evaluator.")
    p.add_argument("--agent-model",      default=DEFAULT_AGENT_MODEL)
    p.add_argument("--max-turns",        type=int, default=10)
    p.add_argument("--no-tool-guide",    action="store_true",
                   help="Drop the category-to-tool selection guide.")
    p.add_argument("--run-in-parallel",  action="store_true")
    p.add_argument("--max-workers",      type=int, default=8)
    p.add_argument("--generator-models", nargs="+", default=DEFAULT_GENERATORS)
    p.add_argument("--prompt-ids",       nargs="+", default=None)
    p.add_argument("--video-dir",        default=DEFAULT_VIDEO_DIR)
    p.add_argument("--rubric-dir",       default=DEFAULT_RUBRIC_DIR)
    p.add_argument("--output-root",      default=DEFAULT_OUTPUT_ROOT)
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

    include_tool_guide = not args.no_tool_guide
    print(f"Agent model:        {args.agent_model}")
    print(f"Max turns:          {args.max_turns}")
    print(f"Tool guide:         {'on' if include_tool_guide else 'off'}")
    print(f"Generator models:   {', '.join(args.generator_models)}")
    print(f"Total jobs:         {len(jobs)}")
    if not jobs:
        return

    def _run(job):
        pid, gen, vp, rubric = job
        return evaluate_one_cell(
            client=client,
            agent_model=args.agent_model,
            prompt_id=pid,
            generator_model=gen,
            video_path=vp,
            rubric_data=rubric,
            output_root=args.output_root,
            max_turns=args.max_turns,
            include_tool_guide=include_tool_guide,
        )

    if args.run_in_parallel:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(_run, job) for job in jobs]
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc="evaluating"):
                try:
                    out = f.result()
                    if out:
                        _log(f"  saved: {out}")
                except Exception as e:
                    _log(f"  error: {e}")
    else:
        for job in tqdm(jobs, desc="evaluating"):
            try:
                out = _run(job)
                if out:
                    _log(f"  saved: {out}")
            except Exception as e:
                _log(f"  error: {e}")


if __name__ == "__main__":
    main()
