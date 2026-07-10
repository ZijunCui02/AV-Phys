#!/usr/bin/env python3
"""
AV-Phys Bench — static site builder.

Reads release data + site sources, emits a self-contained static site:
  build/
    index.html                  -> leaderboard (homepage)
    videos/index.html           -> Apple-style gallery (Seedance, by category)
    videos/<INDEX>/index.html   -> per-prompt page (7 model videos + rubric)
    data/leaderboard.json       -> eval_type x model x partition x metric
    data/prompts.json           -> 321-entry catalog for gallery JS
    style.css, theme.js, fonts/, assets/thumbs/...

Usage:
    python3 build.py
    python3 build.py --release /path/to/data_release --src /path/to/src --out build
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# ---------- Models ----------------------------------------------------------
MODELS = [
    "Seedance-2.0",
    "Kling-3.0-Omni",
    "Veo-3.1",
    "LTX-2.3",
    "Ovi",
    "JavisDiT++",
    "MagiHuman",
]
MODEL_DISPLAY = {
    "Seedance-2.0":   "Seedance 2.0",
    "Kling-3.0-Omni": "Kling 3.0 Omni",
    "Veo-3.1":        "Veo 3.1",
    "LTX-2.3":        "LTX-2.3",
    "Ovi":            "Ovi 1.1",
    "JavisDiT++":     "JavisDiT++",
    "MagiHuman":      "MagiHuman",
}
MODEL_ORG = {
    "Seedance-2.0":   "ByteDance · proprietary",
    "Kling-3.0-Omni": "Kuaishou · proprietary",
    "Veo-3.1":        "Google · proprietary",
    "LTX-2.3":        "Lightricks · open",
    "Ovi":            "character.ai · open",
    "JavisDiT++":     "JavisDiT · open",
    "MagiHuman":      "GAIR-NLP · open",
}

DIMS = ["video_sa", "audio_sa", "video_pc", "audio_pc", "av_pc"]
DIM_LABEL = {
    "video_sa": "V-SA",
    "audio_sa": "A-SA",
    "video_pc": "V-PC",
    "audio_pc": "A-PC",
    "av_pc":    "AV-PC",
    "SA":       "SA",
    "PC":       "PC",
    "Both":     "Both",
}

CATEGORY_LABELS = {
    "C1": "Steady State",
    "C2": "Event Transition",
    "C3": "Environment Transition",
}

HF_BASE = "https://huggingface.co/datasets/ZijunCui/AV-Phys-Bench/resolve/main"
GH_CODE = "https://github.com/ZijunCui02/AV-Phys"
HF_DATASET = "https://huggingface.co/datasets/ZijunCui/AV-Phys-Bench"
ARXIV = "https://arxiv.org/abs/2605.07061"

# Canonical origin of the deployed site. Canonical/OG/sitemap URLs always use
# this, independent of --base-url, so mirrors (e.g. *.github.io) point back here.
CANON = "https://zijuncui.com/AV-Phys"
OG_IMAGE = f"{CANON}/assets/og-card.jpg"
PAPER_TITLE = "Do Joint Audio-Video Generation Models Understand Physics?"
DESC_DEFAULT = (
    "AV-Phys Bench — benchmarking physical commonsense in joint audio-video "
    "generation. Leaderboard, dataset, and per-prompt video gallery."
)

# Base URL prefix for all internal links. Set via --base-url (e.g. "/AV-Phys").
# Default empty → site is served at the host root.
BASE = ""

# ---------- Data loading -----------------------------------------------------
def load_prompts(release: Path):
    """Returns list of dicts ordered by CSV index."""
    rows = []
    with (release / "prompts.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["is_anti"] = r["subcategory_id"].endswith("-4")
            rows.append(r)
    return rows


def load_rubric(release: Path, index: str):
    with (release / "rubrics" / f"{index}.json").open(encoding="utf-8") as f:
        return json.load(f)


def load_human(release: Path, index: str):
    p = release / "human_eval" / f"{index}.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def load_eval(release: Path, eval_kind: str, model: str, index: str):
    """eval_kind in {mllm_eval, agent_av_eval, agent_audio_eval, agent_visual_eval}"""
    p = release / eval_kind / model / f"{index}.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


# ---------- Aggregation ------------------------------------------------------
def majority_vote(verdicts):
    """Given a list of 'yes'/'no'/null verdicts, return 1/0/None.
       Majority = >=ceil(n_nonnull/2)+1? We use strict majority (>50% yes).
       For 3 evaluators: 2+ yes -> pass."""
    nonnull = [v for v in verdicts if v is not None]
    if not nonnull:
        return None
    yes = sum(1 for v in nonnull if str(v).lower() == "yes")
    return 1 if yes * 2 > len(nonnull) else 0


def dimension_pass(statements):
    """A dim passes IFF every non-None statement is 1. None statements are skipped.
       If all statements are None, the dimension is None (doesn't apply)."""
    nonnull = [s for s in statements if s is not None]
    if not nonnull:
        return None
    return 1 if all(s == 1 for s in nonnull) else 0


def aggregate_human_prompt(human_data, model):
    """For a single prompt + model, compute the 5-dim pass for human eval
       via majority vote across the 3 evaluators."""
    if not human_data:
        return None
    # Collect, per statement key, verdicts across evaluators.
    per_stmt = defaultdict(list)
    for ev in human_data.get("evaluations", []):
        for k, v in ev.get("scores", {}).items():
            if not k.startswith(model + "."):
                continue
            stmt_key = k[len(model) + 1:]  # "video_sa.objects"
            per_stmt[stmt_key].append(v)

    # Majority-vote each statement.
    voted = {k: majority_vote(vs) for k, vs in per_stmt.items()}
    # Group by dimension prefix.
    by_dim = defaultdict(list)
    for k, val in voted.items():
        dim = k.split(".", 1)[0]   # "video_sa"
        by_dim[dim].append(val)

    dim_pass = {d: dimension_pass(by_dim.get(d, [])) for d in DIMS}
    return derive_aggregates(dim_pass)


def derive_aggregates(dim_pass):
    """Given video_sa/audio_sa/video_pc/audio_pc/av_pc -> 0/1/None,
       compute SA / PC / Both with conjunction; treat None as 'not applicable' (=1)."""
    def to_bool(x):
        return 1 if (x is None or x == 1) else 0

    sa = to_bool(dim_pass.get("video_sa")) & to_bool(dim_pass.get("audio_sa"))
    pc = to_bool(dim_pass.get("video_pc")) & to_bool(dim_pass.get("audio_pc")) & to_bool(dim_pass.get("av_pc"))
    both = sa & pc

    out = dict(dim_pass)
    out["SA"] = sa
    out["PC"] = pc
    out["Both"] = both
    return out


def aggregate_judge_prompt(eval_data):
    """For mllm / agent eval files: their `aggregated` block already has
       0/1 per dimension and SA/PC/Both. Just normalize."""
    if not eval_data:
        return None
    agg = eval_data.get("aggregated", {})
    return {k: agg.get(k) for k in DIMS + ["SA", "PC", "Both"]}


def partition_filter(prompt, partition):
    cid = prompt["category_id"]
    if partition == "overall":
        return True
    if partition == "physics":
        return not prompt["is_anti"]
    if partition == "anti":
        return prompt["is_anti"]
    if partition in ("C1", "C2", "C3"):
        return cid == partition and not prompt["is_anti"]
    return True


def compute_leaderboard(prompts, release):
    """Returns dict[eval_type][model][partition][metric] -> float in [0, 1]."""
    eval_types = ["human", "agent_av", "mllm"]
    partitions = ["physics", "overall", "C1", "C2", "C3", "anti"]
    metrics = DIMS + ["SA", "PC", "Both"]

    # Pre-load all per-prompt aggregates.
    # Shape: per_prompt_pass[eval_type][model][index] = {metric: 0/1/None}
    per_prompt = {et: defaultdict(dict) for et in eval_types}

    for prompt in prompts:
        idx = prompt["index"]
        human = load_human(release, idx)
        for model in MODELS:
            # human
            per_prompt["human"][model][idx] = aggregate_human_prompt(human, model)
            # mllm
            per_prompt["mllm"][model][idx] = aggregate_judge_prompt(
                load_eval(release, "mllm_eval", model, idx)
            )
            # agent (av)
            per_prompt["agent_av"][model][idx] = aggregate_judge_prompt(
                load_eval(release, "agent_av_eval", model, idx)
            )

    # Aggregate to leaderboard.
    lb = {et: {m: {p: {} for p in partitions} for m in MODELS} for et in eval_types}
    for et in eval_types:
        for model in MODELS:
            for partition in partitions:
                tally = {met: [0, 0] for met in metrics}  # [pass_count, total_count]
                for prompt in prompts:
                    if not partition_filter(prompt, partition):
                        continue
                    rec = per_prompt[et][model].get(prompt["index"])
                    if rec is None:
                        continue
                    for met in metrics:
                        val = rec.get(met)
                        if val is None:
                            continue
                        tally[met][1] += 1
                        if val:
                            tally[met][0] += 1
                for met in metrics:
                    passed, total = tally[met]
                    lb[et][model][partition][met] = (passed / total) if total > 0 else None
    return lb, per_prompt


# ---------- HTML generation --------------------------------------------------
NAV_SPRITE = """<svg width="0" height="0" style="position:absolute" aria-hidden="true">
<defs>
<symbol id="i-angle-right" viewBox="0 0 256 512"><path fill="currentColor" d="M246.6 278.6c12.5-12.5 12.5-32.8 0-45.3l-160-160c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3L178.7 256 41.4 393.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0l160-160z"/></symbol>
<symbol id="i-github" viewBox="0 0 24 24"><path fill="currentColor" d="M16.24 22a1 1 0 0 1-1-1v-2.6a2.15 2.15 0 0 0-.54-1.66a1 1 0 0 1 .61-1.67C17.75 14.78 20 14 20 9.77a4 4 0 0 0-.67-2.22a2.75 2.75 0 0 1-.41-2.06a3.7 3.7 0 0 0 0-1.41a7.7 7.7 0 0 0-2.09 1.09a1 1 0 0 1-.84.15a10.15 10.15 0 0 0-5.52 0a1 1 0 0 1-.84-.15a7.4 7.4 0 0 0-2.11-1.09a3.5 3.5 0 0 0 0 1.41a2.84 2.84 0 0 1-.43 2.08a4.07 4.07 0 0 0-.67 2.23c0 3.89 1.88 4.93 4.7 5.29a1 1 0 0 1 .82.66a1 1 0 0 1-.21 1a2.06 2.06 0 0 0-.55 1.56V21a1 1 0 0 1-2 0v-.57a6 6 0 0 1-5.27-2.09a3.9 3.9 0 0 0-1.16-.88a1 1 0 1 1 .5-1.94a4.9 4.9 0 0 1 2 1.36c1 1 2 1.88 3.9 1.52a3.9 3.9 0 0 1 .23-1.58c-2.06-.52-5-2-5-7a6 6 0 0 1 1-3.33a.85.85 0 0 0 .13-.62a5.7 5.7 0 0 1 .33-3.21a1 1 0 0 1 .63-.57c.34-.1 1.56-.3 3.87 1.2a12.16 12.16 0 0 1 5.69 0c2.31-1.5 3.53-1.31 3.86-1.2a1 1 0 0 1 .63.57a5.7 5.7 0 0 1 .33 3.22a.75.75 0 0 0 .11.57a6 6 0 0 1 1 3.34c0 5.07-2.92 6.54-5 7a4.3 4.3 0 0 1 .22 1.67V21a1 1 0 0 1-.94 1"/></symbol>
<symbol id="i-hf-outline" viewBox="0 0 24 24">
<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.6"/>
<circle cx="9" cy="10.6" r="1" fill="currentColor"/>
<circle cx="15" cy="10.6" r="1" fill="currentColor"/>
<path fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" d="M8.4 14.2c1 1.5 2.4 2.3 3.6 2.3s2.6-.8 3.6-2.3"/>
</symbol>
<symbol id="i-hf" viewBox="0 0 24 24"><path fill="currentColor" d="M12.025 1.13c-5.77 0-10.449 4.647-10.449 10.378c0 1.112.178 2.181.503 3.185c.064-.222.203-.444.416-.577a.96.96 0 0 1 .524-.15c.293 0 .584.124.84.284c.278.173.48.408.71.694c.226.282.458.611.684.951v-.014c.017-.324.106-.622.264-.874s.403-.487.762-.543c.3-.047.596.06.787.203s.31.313.4.467c.15.257.212.468.233.542c.01.026.653 1.552 1.657 2.54c.616.605 1.01 1.223 1.082 1.912c.055.537-.096 1.059-.38 1.572c.637.121 1.294.187 1.967.187c.657 0 1.298-.063 1.921-.178c-.287-.517-.44-1.041-.384-1.581c.07-.69.465-1.307 1.081-1.913c1.004-.987 1.647-2.513 1.657-2.539c.021-.074.083-.285.233-.542c.09-.154.208-.323.4-.467a1.08 1.08 0 0 1 .787-.203c.359.056.604.29.762.543s.247.55.265.874v.015c.225-.34.457-.67.683-.952c.23-.286.432-.52.71-.694c.257-.16.547-.284.84-.285a.97.97 0 0 1 .524.151c.228.143.373.388.43.625l.006.04a10.3 10.3 0 0 0 .534-3.273c0-5.731-4.678-10.378-10.449-10.378M8.327 6.583a1.5 1.5 0 0 1 .713.174a1.487 1.487 0 0 1 .617 2.013c-.183.343-.762-.214-1.102-.094c-.38.134-.532.914-.917.71a1.487 1.487 0 0 1 .69-2.803m7.486 0a1.487 1.487 0 0 1 .689 2.803c-.385.204-.536-.576-.916-.71c-.34-.12-.92.437-1.103.094a1.487 1.487 0 0 1 .617-2.013a1.5 1.5 0 0 1 .713-.174m-10.68 1.55a.96.96 0 1 1 0 1.921a.96.96 0 0 1 0-1.92m13.838 0a.96.96 0 1 1 0 1.92a.96.96 0 0 1 0-1.92M8.489 11.458c.588.01 1.965 1.157 3.572 1.164c1.607-.007 2.984-1.155 3.572-1.164c.196-.003.305.12.305.454c0 .886-.424 2.328-1.563 3.202c-.22-.756-1.396-1.366-1.63-1.32q-.011.001-.02.006l-.044.026l-.01.008l-.03.024q-.018.017-.035.036l-.032.04a1 1 0 0 0-.058.09l-.014.025q-.049.088-.11.19a1 1 0 0 1-.083.116a1.2 1.2 0 0 1-.173.18q-.035.029-.075.058a1.3 1.3 0 0 1-.251-.243a1 1 0 0 1-.076-.107c-.124-.193-.177-.363-.337-.444c-.034-.016-.104-.008-.2.022q-.094.03-.216.087q-.06.028-.125.063l-.13.074q-.067.04-.136.086a3 3 0 0 0-.135.096a3 3 0 0 0-.26.219a2 2 0 0 0-.12.121a2 2 0 0 0-.106.128l-.002.002a2 2 0 0 0-.09.132l-.001.001a1.2 1.2 0 0 0-.105.212q-.013.036-.024.073c-1.139-.875-1.563-2.317-1.563-3.203c0-.334.109-.457.305-.454m.836 10.354c.824-1.19.766-2.082-.365-3.194c-1.13-1.112-1.789-2.738-1.789-2.738s-.246-.945-.806-.858s-.97 1.499.202 2.362c1.173.864-.233 1.45-.685.64c-.45-.812-1.683-2.896-2.322-3.295s-1.089-.175-.938.647s2.822 2.813 2.562 3.244s-1.176-.506-1.176-.506s-2.866-2.567-3.49-1.898s.473 1.23 2.037 2.16c1.564.932 1.686 1.178 1.464 1.53s-3.675-2.511-4-1.297c-.323 1.214 3.524 1.567 3.287 2.405c-.238.839-2.71-1.587-3.216-.642c-.506.946 3.49 2.056 3.522 2.064c1.29.33 4.568 1.028 5.713-.624m5.349 0c-.824-1.19-.766-2.082.365-3.194c1.13-1.112 1.789-2.738 1.789-2.738s.246-.945.806-.858s.97 1.499-.202 2.362c-1.173.864.233 1.45.685.64c.451-.812 1.683-2.896 2.322-3.295s1.089-.175.938.647s-2.822 2.813-2.562 3.244s1.176-.506 1.176-.506s2.866-2.567 3.49-1.898s-.473 1.23-2.037 2.16c-1.564.932-1.686 1.178-1.464 1.53s3.675-2.511 4-1.297c.323 1.214-3.524 1.567-3.287 2.405c.238.839 2.71-1.587 3.216-.642c.506.946-3.49 2.056-3.522 2.064c-1.29.33-4.568 1.028-5.713-.624"/></symbol>
<symbol id="i-arxiv" viewBox="0 0 448 512"><path fill="currentColor" d="M62.258 8.006a22.22 22.22 0 0 0-20.929 13.448c-3.404 8.169-.96 13.898 6.506 24.59c10.935 16.09 122.178 149.673 122.178 149.673l-24.619 23.038c-20.74 20.735-21.632 48.566-2.34 67.852l28.663 27.3l-79.976 98.235c-6.21 6.614-10.053 18.221-6.585 26.552a22.7 22.7 0 0 0 21.21 14.06a20.23 20.23 0 0 0 15.249-7.536l95.122-88.437L363.33 496.39a27.14 27.14 0 0 0 18.418 7.61a25.3 25.3 0 0 0 7.335-1.108a27.66 27.66 0 0 0 18.4-18.99a25.6 25.6 0 0 0-6.481-23.69L272.219 305.195l23.062-21.443c17.198-15.504 17.29-42.455.197-58.076l-25.257-24.228L357.417 98.46l.115-.133l.103-.14c7.793-10.123 11.52-17.92 7.502-27.806a36.17 36.17 0 0 0-23.647-18.37a24 24 0 0 0-3.166-.212l-.006.018a28.52 28.52 0 0 0-18.252 8.123l-.203.166l-.19.173L218.6 151.925L79.261 18.253S70.995 8.213 62.258 8.006m276.06 51.214q1.115.004 2.22.148a29.3 29.3 0 0 1 17.719 13.81c2.246 5.523 1.554 10.01-6.506 20.484L264.861 196.3l-40.882-39.22l100.68-91.304a21.77 21.77 0 0 1 13.66-6.536zM175.077 201.127L395.19 464.872c4.32 5.408 7.02 10.818 5.18 16.914a20.25 20.25 0 0 1-13.463 14.037a17.6 17.6 0 0 1-5.17.784a19.8 19.8 0 0 1-13.293-5.56l-220.15-209.694c-17.317-17.316-14.698-40.33 2.158-57.186z"/></symbol>
<symbol id="i-sun" viewBox="0 0 32 32"><path fill="currentColor" d="M16 12.005a4 4 0 1 1-4 4a4.005 4.005 0 0 1 4-4m0-2a6 6 0 1 0 6 6a6 6 0 0 0-6-6M5.394 6.813L6.81 5.399l3.505 3.506L8.9 10.319zM2 15.005h5v2H2zm3.394 10.193L8.9 21.692l1.414 1.414l-3.505 3.506zM15 25.005h2v5h-2zm6.687-1.9l1.414-1.414l3.506 3.506l-1.414 1.414zm3.313-8.1h5v2h-5zm-3.313-6.101l3.506-3.506l1.414 1.414l-3.506 3.506zM15 2.005h2v5h-2z"/></symbol>
<symbol id="i-moon" viewBox="0 0 32 32"><path fill="currentColor" d="M13.503 5.414a15.076 15.076 0 0 0 11.593 18.194a11.1 11.1 0 0 1-7.975 3.39c-.138 0-.278.005-.418 0a11.094 11.094 0 0 1-3.2-21.584M14.98 3a1 1 0 0 0-.175.016a13.096 13.096 0 0 0 1.825 25.981c.164.006.328 0 .49 0a13.07 13.07 0 0 0 10.703-5.555a1.01 1.01 0 0 0-.783-1.565A13.08 13.08 0 0 1 15.89 4.38A1.015 1.015 0 0 0 14.98 3"/></symbol>
<symbol id="i-auto" viewBox="0 0 24 24"><path fill="currentColor" d="M11 21q-3.35 0-5.675-2.325T3 13q0-1.45.475-2.762T4.8 7.888t2.037-1.725T9.426 5.3q.625-.075.975.438t.025 1.062q-.3.5-.363 1.063T10 9q0 2.5 1.75 4.25T16 15q.3 0 .6-.012t.6-.113q.525-.2.938.188t.237.912q-.725 2.35-2.8 3.688T11 21m5.4-12l-.5 1.4q-.1.275-.325.438t-.5.162q-.475 0-.737-.387t-.113-.813l2.55-7.175q.1-.275.35-.45t.55-.175h.65q.3 0 .55.175t.35.45l2.55 7.175q.15.425-.112.813t-.738.387q-.275 0-.5-.162T20.1 10.4L19.6 9zm.45-1.35h2.3L18 4z"/></symbol>
<symbol id="i-code" viewBox="0 0 24 24"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m18 16l4-4l-4-4M6 8l-4 4l4 4m8.5-12l-5 16"/></symbol>
<symbol id="i-film" viewBox="0 0 24 24"><g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18M3 7.5h4M3 12h18M3 16.5h4M17 3v18m0-13.5h4m-4 9h4"/></g></symbol>
</defs>
</svg>
"""

def navbar_html(active=""):
    items = [
        ("leaderboard", "", f"{BASE}/"),
        ("TL;DR", "tldr", f"{BASE}/tl-dr/"),
        ("videos", "videos", f"{BASE}/videos/"),
        ("run AV-Phys", "run", GH_CODE),
    ]
    parts = []
    for label, key, href in items:
        ext = ' target="_blank" rel="noopener noreferrer"' if href.startswith("http") else ""
        cls = "nav-item" + (" nav-active" if key == active else "")
        parts.append(f'<a class="{cls}" href="{href}"{ext}>{label}</a>')
    nav_items = "\n      ".join(parts)
    return f"""<header class="nav" id="navbar">
  <a class="logo" href="{BASE}/" aria-label="Home">
    <span>&gt;_AV-Phys</span><span class="blink">_</span>
  </a>
  <nav class="nav-links">
      {nav_items}
      <a class="nav-item" href="{GH_CODE}" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
        <svg class="icon" aria-hidden="true"><use href="#i-github"/></svg>
      </a>
      <a class="nav-item" href="{HF_DATASET}" target="_blank" rel="noopener noreferrer" aria-label="HuggingFace">
        <svg class="icon" aria-hidden="true"><use href="#i-hf"/></svg>
      </a>
      <button class="theme-toggle" id="theme-toggle" aria-label="Toggle theme">
        <svg class="icon icon-auto" aria-hidden="true"><use href="#i-auto"/></svg>
        <svg class="icon icon-light" aria-hidden="true"><use href="#i-sun"/></svg>
        <svg class="icon icon-dark" aria-hidden="true"><use href="#i-moon"/></svg>
      </button>
  </nav>
</header>
"""


FOOTER_HTML = f"""<footer class="footer">
  <span>© AV-Phys Bench 2026. Theme forked from <a href="https://github.com/Renovamen/renovamen.github.io" target="_blank" rel="noopener noreferrer">renovamen</a>.</span>
  <span class="right">
    <a href="{GH_CODE}" target="_blank" rel="noopener noreferrer"><svg class="icon"><use href="#i-code"/></svg><span>Code</span></a>
    <a href="{HF_DATASET}" target="_blank" rel="noopener noreferrer"><svg class="icon"><use href="#i-hf"/></svg><span>Data</span></a>
    <a href="{ARXIV}" target="_blank" rel="noopener noreferrer"><svg class="icon"><use href="#i-arxiv"/></svg><span>arXiv</span></a>
  </span>
</footer>"""


def _esc(s):
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def page_shell(title, body, active, path="/", desc=DESC_DEFAULT, extra_head=""):
    canon = CANON + path
    t = _esc(title)
    d = _esc(desc)
    return f"""<!DOCTYPE html>
<html lang="en" class="theme-auto">
<head>
<meta charset="UTF-8">
<title>{t}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{d}">
<link rel="canonical" href="{canon}">
<meta property="og:site_name" content="AV-Phys Bench">
<meta property="og:type" content="website">
<meta property="og:title" content="{t}">
<meta property="og:description" content="{d}">
<meta property="og:url" content="{canon}">
<meta property="og:image" content="{OG_IMAGE}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{t}">
<meta name="twitter:description" content="{d}">
<meta name="twitter:image" content="{OG_IMAGE}">
{extra_head}<link rel="stylesheet" href="{BASE}/style.css">
<script>
(function () {{
  var s = localStorage.getItem("color-scheme") || "auto";
  var prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  var useDark = s === "dark" || (s !== "light" && prefersDark);
  var r = document.documentElement;
  r.classList.remove("theme-auto","theme-light","theme-dark");
  r.classList.add("theme-" + s);
  r.classList.toggle("dark", useDark);
}})();
</script>
</head>
<body>
{NAV_SPRITE}
{navbar_html(active)}
<main>
{body}
{FOOTER_HTML}
</main>
<script src="{BASE}/theme.js"></script>
</body>
</html>
"""


# ---------- Page bodies ------------------------------------------------------
AUTHORS_HTML = """<p class="authors">
  <a href="https://zijuncui.com" target="_blank" rel="noopener noreferrer">Zijun&nbsp;Cui</a><sup>1,*</sup>,
  <a href="https://dragonliu1995.github.io/" target="_blank" rel="noopener noreferrer">Xiulong&nbsp;Liu</a><sup>2,*</sup>,
  <a href="https://apexhao.github.io/" target="_blank" rel="noopener noreferrer">Hao&nbsp;Fang</a><sup>2,*</sup>,
  <a href="https://scholar.google.com/citations?user=jOZPNeQAAAAJ&amp;hl=zh-CN" target="_blank" rel="noopener noreferrer">Mingwei&nbsp;Xu</a><sup>2</sup>,
  <a href="https://jiagengliu02.github.io/" target="_blank" rel="noopener noreferrer">Jiageng&nbsp;Liu</a><sup>3</sup>,
  <a href="https://zexinxu.com/" target="_blank" rel="noopener noreferrer">Zexin&nbsp;Xu</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=K-ObTwoAAAAJ&amp;hl=zh-CN" target="_blank" rel="noopener noreferrer">Weiguo&nbsp;Pian</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=7LBj70IAAAAJ&amp;hl=en" target="_blank" rel="noopener noreferrer">Shijian&nbsp;Deng</a><sup>1</sup>,
  Feiyu&nbsp;Du<sup>1</sup>, Chenming&nbsp;Ge<sup>2</sup>,
  <a href="https://www.yapengtian.com/" target="_blank" rel="noopener noreferrer">Yapeng&nbsp;Tian</a><sup>1,&dagger;</sup>
</p>
<p class="affil">
  <sup>1</sup>&nbsp;University of Texas at Dallas &nbsp;&nbsp;
  <sup>2</sup>&nbsp;University of Washington &nbsp;&nbsp;
  <sup>3</sup>&nbsp;University of California, Los Angeles
  &nbsp;&nbsp;&nbsp;&nbsp;
  <sup>*</sup>&nbsp;Equal contribution. &nbsp;&nbsp;
  <sup>&dagger;</sup>&nbsp;Corresponding author.
</p>"""

AUTHOR_LINKS = [
    ("Zijun Cui", "https://zijuncui.com/"),
    ("Xiulong Liu", "https://dragonliu1995.github.io/"),
    ("Hao Fang", "https://apexhao.github.io/"),
    ("Mingwei Xu", "https://scholar.google.com/citations?user=jOZPNeQAAAAJ"),
    ("Jiageng Liu", "https://jiagengliu02.github.io/"),
    ("Zexin Xu", "https://zexinxu.com/"),
    ("Weiguo Pian", "https://scholar.google.com/citations?user=K-ObTwoAAAAJ"),
    ("Shijian Deng", "https://scholar.google.com/citations?user=7LBj70IAAAAJ"),
    ("Feiyu Du", None),
    ("Chenming Ge", None),
    ("Yapeng Tian", "https://www.yapengtian.com/"),
]


def index_jsonld():
    authors = []
    for name, url in AUTHOR_LINKS:
        person = {"@type": "Person", "name": name}
        if url:
            person["url"] = url
        authors.append(person)
    article = {
        "@context": "https://schema.org",
        "@type": "ScholarlyArticle",
        "headline": PAPER_TITLE,
        "alternativeHeadline": "AV-Phys Bench",
        "url": CANON + "/",
        "mainEntityOfPage": CANON + "/",
        "image": OG_IMAGE,
        "author": authors,
        "datePublished": "2026-05",
        "description": DESC_DEFAULT,
        "keywords": (
            "audio-video generation, physical commonsense, benchmark, "
            "video generation, audio generation, world models, evaluation"
        ),
        "sameAs": [ARXIV],
    }
    dataset = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": "AV-Phys-Bench",
        "url": HF_DATASET,
        "sameAs": [CANON + "/"],
        "description": (
            "Prompts, per-prompt physics rubrics, generated videos from seven "
            "joint audio-video generation models, and human ratings."
        ),
        "creator": authors,
        "isAccessibleForFree": True,
        "distribution": [{"@type": "DataDownload", "contentUrl": HF_DATASET}],
    }
    payload = json.dumps([article, dataset], separators=(",", ":"))
    return f'<script type="application/ld+json">{payload}</script>\n'


def leaderboard_body(prompts, leaderboard):
    payload = json.dumps(leaderboard, separators=(",", ":"))
    seg_partitions = [
        ("physics", "Physics-following"),
        ("overall", "Overall"),
        ("C1", "C1 Steady State"),
        ("C2", "C2 Event Transition"),
        ("C3", "C3 Environment Transition"),
        ("anti", "Anti-Physics"),
    ]
    seg_partition_html = "\n        ".join(
        f'<button data-partition="{p}"{(" class=on") if i == 0 else ""}>{label}</button>'
        for i, (p, label) in enumerate(seg_partitions)
    )
    return f"""<article class="content wide">

<section class="hero">
  <h1 class="paper-title">Do Joint Audio-Video Generation Models Understand Physics?</h1>
  {AUTHORS_HTML}
  <div class="badges">
    <a class="btn" href="{ARXIV}" target="_blank" rel="noopener noreferrer">
      <svg class="icon"><use href="#i-arxiv"/></svg><span>arXiv</span>
    </a>
    <a class="btn" href="{HF_DATASET}" target="_blank" rel="noopener noreferrer">
      <svg class="icon"><use href="#i-hf"/></svg><span>HF Dataset</span>
    </a>
    <a class="btn" href="{GH_CODE}" target="_blank" rel="noopener noreferrer">
      <svg class="icon"><use href="#i-code"/></svg><span>Code</span>
    </a>
    <a class="btn" href="{BASE}/videos/">
      <svg class="icon"><use href="#i-film"/></svg><span>Videos</span>
    </a>
  </div>
  <p class="lede">AV-Phys Bench is first comprehensive benchmark for evaluating <em>physical commonsense</em> in joint audio-video generation. AV-Phys Bench systematically tests joint audio-video generation models across three scene categories that probe how physical commonsense holds as the scene evolves: <strong>(a) Steady State</strong>, <strong>(b) Event Transition</strong>, and <strong>(c) Environment Transition</strong>. 7 models are evaluated by humans, an MLLM-as-judge baseline, and the AV-Phys Agent.</p>
</section>

<section class="teaser">
  <div class="teaser-grid">
    <figure class="teaser-card">
      <div class="teaser-media">
        <video controls preload="metadata" playsinline poster="{BASE}/assets/thumbs/Seedance-2.0/C2-2-20.webp">
          <source src="{HF_BASE}/videos/Seedance-2.0/C2-2-20.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="teaser-label">
        <span class="teaser-model">Seedance 2.0</span>
        <span class="teaser-mark pass" aria-label="passes physics test">&check;</span>
      </figcaption>
    </figure>
    <figure class="teaser-card">
      <div class="teaser-media">
        <video controls preload="metadata" playsinline poster="{BASE}/assets/thumbs/Kling-3.0-Omni/C2-2-20.webp">
          <source src="{HF_BASE}/videos/Kling-3.0-Omni/C2-2-20.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="teaser-label">
        <span class="teaser-model">Kling 3.0 Omni</span>
        <span class="teaser-mark fail" aria-label="fails physics test">&times;</span>
      </figcaption>
    </figure>
    <figure class="teaser-card">
      <div class="teaser-media">
        <video controls preload="metadata" playsinline poster="{BASE}/assets/thumbs/Veo-3.1/C2-2-20.webp">
          <source src="{HF_BASE}/videos/Veo-3.1/C2-2-20.mp4" type="video/mp4">
        </video>
      </div>
      <figcaption class="teaser-label">
        <span class="teaser-model">Veo 3.1</span>
        <span class="teaser-mark fail" aria-label="fails physics test">&times;</span>
      </figcaption>
    </figure>
  </div>
  <p class="teaser-prompt">
    <span class="quot">&ldquo;</span>A speaker plays <strong>music</strong> at <strong>low volume</strong>, sounding quiet and thin. Then the volume knob is <strong>turned up</strong> gradually until the <strong>music</strong> fills the room.<span class="quot">&rdquo;</span>
    <a class="teaser-link" href="{BASE}/videos/C2-2-20/">C2-2-20 ↗</a>
  </p>
</section>

<h2 id="lb">Leaderboard</h2>

<div class="lb-controls">
  <div class="group">
    <span class="group-label">Evaluator</span>
    <div class="seg" id="eval-toggle">
      <button data-et="human" class="on">Human</button>
      <button data-et="agent_av">AV-Phys Agent</button>
      <button data-et="mllm">MLLM judge</button>
    </div>
  </div>
  <div class="group">
    <span class="group-label">Subset</span>
    <div class="seg" id="partition-toggle">
        {seg_partition_html}
    </div>
  </div>
</div>

<div class="lb-table-wrap">
  <table class="leaderboard" id="lb-table">
    <thead>
      <tr>
        <th class="num">#</th>
        <th>Model</th>
        <th class="num">SA</th>
        <th class="num">PC</th>
        <th class="num">Both</th>
        <th class="num"><span class="dim">V-SA</span></th>
        <th class="num"><span class="dim">A-SA</span></th>
        <th class="num"><span class="dim">V-PC</span></th>
        <th class="num"><span class="dim">A-PC</span></th>
        <th class="num"><span class="dim">AV-PC</span></th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
</div>
<p class="lb-note">
  All numbers are pass-rates in [0, 1]. <strong>SA</strong> = semantic adherence (V-SA ∧ A-SA);
  <strong>PC</strong> = physical commonsense (V-PC ∧ A-PC ∧ AV-PC); <strong>Both</strong> = SA ∧ PC.
  Headline values match Table 3 of the paper on the <em>Physics-following</em> subset.
</p>

<h2 id="taxonomy">Taxonomy</h2>
<p class="muted">
  Prompts are organized on a <strong>scene-evolution axis</strong>: what changes in the scene from start to end.
  Each top category has a 4th <em>Anti-Physics</em> subcategory that deliberately violates a physical principle —
  a stress test for whether the model executes the instruction or defaults to plausible priors.
</p>

<div class="tax-grid">
  <a class="tax-card" href="{BASE}/videos/?cat=C1">
    <div class="tax-head"><span class="tax-code">C1</span><span class="tax-name">Steady State</span></div>
    <p class="tax-count">159 prompts · source, action, environment all fixed</p>
    <p class="tax-subs"><code>C1-1</code> source_material · <code>C1-2</code> source_anchoring · <code>C1-3</code> sound_persistence · <code>C1-4</code> <span class="anti-tag">anti</span></p>
  </a>
  <a class="tax-card" href="{BASE}/videos/?cat=C2">
    <div class="tax-head"><span class="tax-code">C2</span><span class="tax-name">Event Transition</span></div>
    <p class="tax-count">119 prompts · a discrete action changes the source state</p>
    <p class="tax-subs"><code>C2-1</code> source_body · <code>C2-2</code> source_excitation · <code>C2-3</code> source_radiation · <code>C2-4</code> <span class="anti-tag">anti</span></p>
  </a>
  <a class="tax-card" href="{BASE}/videos/?cat=C3">
    <div class="tax-head"><span class="tax-code">C3</span><span class="tax-name">Environment Transition</span></div>
    <p class="tax-count">43 prompts · source fixed, propagation path changes</p>
    <p class="tax-subs"><code>C3-1</code> propagation_medium · <code>C3-2</code> enclosure_geometry · <code>C3-3</code> sound_attenuation · <code>C3-4</code> <span class="anti-tag">anti</span></p>
  </a>
</div>

<h2 id="run">Run AV-Phys on your own model</h2>
<p class="muted">
  Each prompt comes with a hand-authored rubric of 8.6 ± Y/N statements on average. Drop your model's
  outputs into <code>videos/&lt;your-model&gt;/&lt;INDEX&gt;.mp4</code> and the AV-Phys Agent will score them
  against the rubric on 5 dimensions.
</p>
<div class="code-block">
  <span class="lang">bash</span>
  <code><span class="cm"># 1. clone the umbrella repo and enter the eval harness</span>
git clone {GH_CODE}
cd AV-Phys/code

<span class="cm"># 2. fetch the dataset (prompts + rubrics + videos) from HuggingFace</span>
huggingface-cli download ZijunCui/AV-Phys-Bench --repo-type=dataset --local-dir data_release

<span class="cm"># 3. generate one .mp4 per prompt into data_release/videos/&lt;your-model&gt;/</span>
<span class="cm"># 4. follow code/README.md to run the AV-Phys Agent evaluator on your model</span></code>
  <button class="copy-btn">Copy</button>
</div>

<script id="lb-data" type="application/json">{payload}</script>

<script>
(function () {{
  const lbData = JSON.parse(document.getElementById("lb-data").textContent);
  const MODELS = {json.dumps(MODELS)};
  const DISPLAY = {json.dumps(MODEL_DISPLAY)};
  const ORG = {json.dumps(MODEL_ORG)};
  let evType = "human", partition = "physics";

  function fmt(v) {{
    if (v === null || v === undefined) return "—";
    return v.toFixed(3);
  }}

  function render() {{
    const tbody = document.querySelector("#lb-table tbody");
    const rows = MODELS.map(m => {{
      const r = lbData[evType]?.[m]?.[partition] || {{}};
      return [m, r];
    }});
    rows.sort((a, b) => (b[1].Both || 0) - (a[1].Both || 0));
    let out = "";
    rows.forEach(([m, r], i) => {{
      const best = i === 0 && (r.Both != null) ? " best" : "";
      const both = (r.Both || 0);
      const barW = Math.round(both * 100);
      out += `<tr class="${{best.trim()}}">
        <td class="rank num">${{i + 1}}</td>
        <td class="model">${{DISPLAY[m]}}<span class="meta">${{ORG[m]}}</span></td>
        <td class="num">${{fmt(r.SA)}}</td>
        <td class="num">${{fmt(r.PC)}}</td>
        <td class="num headline">
          <span class="score-cell">${{fmt(r.Both)}}
            <span class="score-bar"><i style="width:${{barW}}%"></i></span>
          </span>
        </td>
        <td class="num">${{fmt(r.video_sa)}}</td>
        <td class="num">${{fmt(r.audio_sa)}}</td>
        <td class="num">${{fmt(r.video_pc)}}</td>
        <td class="num">${{fmt(r.audio_pc)}}</td>
        <td class="num">${{fmt(r.av_pc)}}</td>
      </tr>`;
    }});
    tbody.innerHTML = out;
  }}

  document.querySelectorAll("#eval-toggle button").forEach(b => {{
    b.addEventListener("click", () => {{
      document.querySelectorAll("#eval-toggle button").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      evType = b.dataset.et;
      render();
    }});
  }});
  document.querySelectorAll("#partition-toggle button").forEach(b => {{
    b.addEventListener("click", () => {{
      document.querySelectorAll("#partition-toggle button").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      partition = b.dataset.partition;
      render();
    }});
  }});
  render();
}})();
</script>
</article>"""


def gallery_body(prompts):
    cat_counts = defaultdict(int)
    for p in prompts:
        cat_counts[p["category_id"]] += 1
    catalog = [
        {
            "i": p["index"],
            "c": p["category_id"],
            "sc": p["subcategory_id"],
            "scn": p["subcategory_name"],
            "anti": p["is_anti"],
            "prompt": p["prompt"],
        }
        for p in prompts
    ]
    return f"""<article class="content wide">

<h1 style="font-size:1.8em;margin-top:0;">Videos</h1>
<p class="muted">Each card shows the <strong>Seedance 2.0</strong> generation (top-ranked model). Click a card to see all 7 models for the same prompt and the per-prompt rubric.</p>

<div class="gallery">
  <aside class="category-picker" id="cat-picker">
    <button class="cat-btn" data-cat="C1">Steady State</button>
    <button class="cat-btn on" data-cat="C2">Event Transition</button>
    <button class="cat-btn" data-cat="C3">Environment Transition</button>
  </aside>

  <section>
    <div class="sub-chip-row" id="sub-chip-row"></div>
    <div class="video-grid" id="video-grid"></div>
  </section>
</div>

<script id="catalog" type="application/json">{json.dumps(catalog, separators=(',', ':'))}</script>

<script>
(function () {{
  const catalog = JSON.parse(document.getElementById("catalog").textContent);
  const HF = "{HF_BASE}";
  const BASE = "{BASE}";
  const params = new URLSearchParams(window.location.search);
  let activeCat = params.get("cat") || "C2";
  let activeSub = params.get("sub") || null; // null = all in this category
  // Defensive: if sub doesn't belong to active cat (e.g. hand-edited URL), drop it.
  if (activeSub && !activeSub.startsWith(activeCat + "-")) activeSub = null;

  function subcatsOf(cat) {{
    const seen = new Map();
    catalog.filter(p => p.c === cat).forEach(p => {{
      if (!seen.has(p.sc)) seen.set(p.sc, p.scn);
    }});
    return Array.from(seen.entries());
  }}

  function renderSubChips() {{
    const row = document.getElementById("sub-chip-row");
    const subs = subcatsOf(activeCat);
    let html = `<button class="sub-chip ${{activeSub === null ? 'on' : ''}}" data-sub="">all</button>`;
    subs.forEach(([sc, scn]) => {{
      const isAnti = sc.endsWith("-4");
      const cls = "sub-chip" + (isAnti ? " anti" : "") + (activeSub === sc ? " on" : "");
      const label = isAnti ? "anti-physics" : scn;
      html += `<button class="${{cls}}" data-sub="${{sc}}"><code>${{sc}}</code> ${{label}}</button>`;
    }});
    row.innerHTML = html;
    row.querySelectorAll(".sub-chip").forEach(btn => {{
      btn.addEventListener("click", () => {{
        activeSub = btn.dataset.sub || null;
        const url = new URL(window.location);
        if (activeSub) url.searchParams.set("sub", activeSub);
        else url.searchParams.delete("sub");
        history.replaceState(null, "", url.toString());
        render();
      }});
    }});
  }}

  function render() {{
    renderSubChips();
    const grid = document.getElementById("video-grid");
    const items = catalog.filter(p => p.c === activeCat && (activeSub === null || p.sc === activeSub));
    grid.innerHTML = items.map(p => {{
      const antiCls = p.anti ? " anti" : "";
      const videoUrl = `${{HF}}/videos/Seedance-2.0/${{p.i}}.mp4`;
      const thumbUrl = `${{BASE}}/assets/thumbs/Seedance-2.0/${{p.i}}.webp`;
      return `<a class="video-card${{antiCls}}" href="${{BASE}}/videos/${{p.i}}/">
        <div class="media">
          <video poster="${{thumbUrl}}" preload="none" muted playsinline
                 onmouseenter="this.src='${{videoUrl}}';this.play().catch(()=>{{}});"
                 onmouseleave="this.pause();this.removeAttribute('src');this.load();"></video>
        </div>
        <div class="meta">
          <span class="idx">${{p.i}}</span>
          <span class="sub">${{p.scn}}</span>
        </div>
      </a>`;
    }}).join("");
  }}

  document.querySelectorAll("#cat-picker .cat-btn").forEach(b => {{
    b.addEventListener("click", () => {{
      document.querySelectorAll("#cat-picker .cat-btn").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      activeCat = b.dataset.cat;
      activeSub = null;
      const url = new URL(window.location);
      url.searchParams.set("cat", activeCat);
      url.searchParams.delete("sub");
      history.replaceState(null, "", url.toString());
      render();
    }});
  }});

  // Set initial 'on' based on URL.
  document.querySelectorAll("#cat-picker .cat-btn").forEach(b => {{
    b.classList.toggle("on", b.dataset.cat === activeCat);
  }});

  render();
}})();
</script>

</article>"""


def prompt_page_body(prompt, rubric):
    idx = prompt["index"]
    cat_label = CATEGORY_LABELS.get(prompt["category_id"], prompt["category_name"])
    sub_label = prompt["subcategory_name"]
    is_anti = prompt["is_anti"]
    principle = prompt.get("av_phys_principle_name", "")
    discipline = prompt.get("av_phys_principle_discipline", "")

    # Render the 7-model grid.
    cells = []
    for model in MODELS:
        video_url = f"{HF_BASE}/videos/{model}/{idx}.mp4"
        thumb_url = f"{BASE}/assets/thumbs/{model}/{idx}.webp"
        cells.append(f"""<div class="model-cell">
  <div class="media">
    <video controls preload="none" poster="{thumb_url}">
      <source src="{video_url}" type="video/mp4">
    </video>
  </div>
  <div class="label"><span>{MODEL_DISPLAY[model]}</span><span class="org">{MODEL_ORG[model]}</span></div>
</div>""")
    model_grid = "\n".join(cells)

    # Build the rubric body.
    basic = rubric.get("basic_standards", {})
    key = rubric.get("key_standards", {})
    def render_list(items):
        if not items:
            return "<li class='muted'>(none)</li>"
        if isinstance(items, str):
            return f"<li>{items}</li>"
        return "".join(f"<li>{it}</li>" for it in items)

    basic_html = f"""<div class="basic-grid">
  <div>
    <h4>Visual presence</h4>
    <ul>{render_list(basic.get('video', {}).get('objects', []))}</ul>
    <p class="muted" style="margin:.25rem 0;">Event: {basic.get('video', {}).get('event', '—')}</p>
  </div>
  <div>
    <h4>Audio presence</h4>
    <ul>{render_list(basic.get('audio', {}).get('objects', []))}</ul>
    <p class="muted" style="margin:.25rem 0;">Sound: {basic.get('audio', {}).get('sound', '—')}</p>
  </div>
</div>"""

    key_html = ""
    for k, label in [("video_pc", "Visual physical commonsense (V-PC)"),
                     ("audio_pc", "Audio physical commonsense (A-PC)"),
                     ("av_pc", "Cross-modal physical commonsense (AV-PC)")]:
        items = key.get(k, [])
        key_html += f"<h4>{label}</h4><ul>{render_list(items)}</ul>"

    anti_chip = '<span class="chip anti">anti-physics</span>' if is_anti else ""

    return f"""<article class="content wide prompt-detail">

<a class="back-link" href="{BASE}/videos/?cat={prompt['category_id']}&amp;sub={prompt['subcategory_id']}">← all videos</a>

<p class="prompt-text">{prompt['prompt']}</p>
<div class="prompt-meta">
  <span class="chip"><code>{idx}</code></span>
  <span class="chip">{cat_label} · {sub_label}</span>
  {anti_chip}
  <span class="muted">principle: {principle} <em>({discipline})</em></span>
</div>

<div class="model-grid">
{model_grid}
</div>

<details class="rubric">
  <summary>Per-prompt rubric</summary>
  <div class="rubric-body">
    <h4>Basic standards</h4>
    {basic_html}
    <h4>Key standards</h4>
    {key_html}
  </div>
</details>

</article>"""


def tldr_body():
    return f"""<article class="content wide tldr-page">

<h1 style="font-size:1.8em;margin-top:0;">TL;DR</h1>

<p class="muted">A four-minute walkthrough of AV-Phys Bench. The video covers what physical commonsense in joint audio-video generation means, the three scene categories the benchmark tests, and a handful of model outputs so you can see and hear where today's generators succeed and where they break.</p>

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;margin:1.5rem 0;border-radius:6px;">
  <iframe src="https://www.youtube-nocookie.com/embed/J7Nj-sWVmLs?rel=0"
          title="AV-Phys Bench walkthrough"
          frameborder="0"
          loading="lazy"
          allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowfullscreen
          style="position:absolute;top:0;left:0;width:100%;height:100%;"></iframe>
</div>

<p class="muted" style="margin-top:1.5rem;">
  More? See the <a href="{BASE}/">leaderboard</a>, browse the <a href="{BASE}/videos/">per-prompt video gallery</a>, or read the <a href="{ARXIV}" target="_blank" rel="noopener noreferrer">paper</a>.
</p>

</article>"""


# ---------- Build entry point -----------------------------------------------
def build(release: Path, src: Path, out: Path, base_url: str = ""):
    global BASE
    BASE = base_url.rstrip("/")
    out.mkdir(parents=True, exist_ok=True)

    # 1. Copy static assets.
    print(f"[build] copying static assets from {src} to {out}")
    for fname in ("style.css", "theme.js"):
        shutil.copy(src / fname, out / fname)
    if (src / "fonts").exists():
        if (out / "fonts").exists():
            shutil.rmtree(out / "fonts")
        shutil.copytree(src / "fonts", out / "fonts")
    if (src / "assets").exists():
        if (out / "assets").exists():
            shutil.rmtree(out / "assets")
        shutil.copytree(src / "assets", out / "assets")
    (out / "data").mkdir(exist_ok=True)

    # 2. Load prompts.
    print(f"[build] reading prompts.csv ...")
    prompts = load_prompts(release)
    print(f"[build]   {len(prompts)} prompts loaded")

    # 3. Compute leaderboard.
    print(f"[build] aggregating leaderboard scores ...")
    leaderboard, per_prompt = compute_leaderboard(prompts, release)

    with (out / "data" / "leaderboard.json").open("w", encoding="utf-8") as f:
        json.dump(leaderboard, f, separators=(",", ":"))

    # 4. Generate homepage.
    print(f"[build] writing index.html ...")
    body = leaderboard_body(prompts, leaderboard)
    (out / "index.html").write_text(
        page_shell(
            f"AV-Phys Bench: {PAPER_TITLE}",
            body,
            active="",
            path="/",
            desc=(
                "The first comprehensive benchmark for physical commonsense in "
                "joint audio-video generation: a leaderboard of seven models "
                "(Veo 3.1, Seedance 2.0, Kling 3.0 Omni, ...), 321 prompts with "
                "rubrics, and human ratings."
            ),
            extra_head=index_jsonld(),
        ),
        encoding="utf-8",
    )

    # Inject the .tax-grid CSS at the bottom of index page (defined in style.css below).
    # We'll add it to style.css instead in a follow-up.

    # 5. Generate videos/index.html.
    print(f"[build] writing videos/index.html ...")
    (out / "videos").mkdir(exist_ok=True)
    (out / "videos" / "index.html").write_text(
        page_shell(
            "AV-Phys Bench — Per-Prompt Video Gallery",
            gallery_body(prompts),
            active="videos",
            path="/videos/",
            desc=(
                "Browse all 321 AV-Phys Bench prompts with generations from "
                "seven joint audio-video models and per-prompt physics rubric "
                "verdicts."
            ),
        ),
        encoding="utf-8",
    )

    # 5b. Generate tl-dr/index.html (video reads av_phys_demo.mp4 from the same dir).
    print(f"[build] writing tl-dr/index.html ...")
    (out / "tl-dr").mkdir(exist_ok=True)
    (out / "tl-dr" / "index.html").write_text(
        page_shell(
            "AV-Phys Bench — TL;DR",
            tldr_body(),
            active="tldr",
            path="/tl-dr/",
            desc=(
                "A one-minute tour of AV-Phys Bench: how we probe physical "
                "commonsense in joint audio-video generation and where seven "
                "state-of-the-art models fail."
            ),
        ),
        encoding="utf-8",
    )

    # 6. Generate 321 per-prompt pages.
    print(f"[build] writing 321 per-prompt pages ...")
    for i, prompt in enumerate(prompts):
        idx = prompt["index"]
        rubric = load_rubric(release, idx)
        out_dir = out / "videos" / idx
        out_dir.mkdir(exist_ok=True)
        body = prompt_page_body(prompt, rubric)
        ptext = prompt["prompt"].strip()
        if len(ptext) > 200:
            ptext = ptext[:197] + "..."
        (out_dir / "index.html").write_text(
            page_shell(
                f"{idx} · AV-Phys Bench",
                body,
                active="videos",
                path=f"/videos/{idx}/",
                desc=(
                    f'AV-Phys Bench prompt {idx}: "{ptext}" Generations from '
                    "seven audio-video models with physics rubric verdicts."
                ),
            ),
            encoding="utf-8",
        )
        if (i + 1) % 50 == 0:
            print(f"[build]   {i+1}/{len(prompts)} pages")

    # 7. Sitemap for the deployed site.
    urls = [CANON + "/", CANON + "/tl-dr/", CANON + "/videos/"]
    urls += [f"{CANON}/videos/{p['index']}/" for p in prompts]
    today = date.today().isoformat()
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for u in urls:
        lines.append(f"  <url><loc>{u}</loc><lastmod>{today}</lastmod></url>")
    lines.append("</urlset>")
    (out / "sitemap.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[build] sitemap.xml with {len(urls)} URLs")

    print(f"[build] done.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=Path("/home/harry/projects/AV-Phys-Bench/release/data_release"))
    parser.add_argument("--src", type=Path, default=Path("/home/harry/projects/AV-Phys/src"))
    parser.add_argument("--out", type=Path, default=Path("/home/harry/projects/AV-Phys/docs"))
    parser.add_argument("--base-url", default="", help='URL prefix for internal links, e.g. "/AV-Phys"')
    args = parser.parse_args(argv)
    build(args.release.resolve(), args.src.resolve(), args.out.resolve(), args.base_url)


if __name__ == "__main__":
    main()
