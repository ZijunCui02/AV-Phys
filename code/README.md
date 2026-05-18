# AV-Phys Bench Code

Reference code for the four automatic evaluators reported alongside human ratings on AV-Phys Bench.

## Contents

```
code/
├── README.md
├── requirements.txt
├── mllm/            MLLM-as-judge baseline
├── agent_audio/     ReAct agent with audio tools (AV-Phys Agent)
├── agent_visual/    ReAct agent with frame inspection tools
└── agent_av/        ReAct agent with both audio DSP and visual tools
```

Every evaluator reads the dataset's `prompts.csv`, the per-prompt rubrics under `rubrics/{INDEX}.json`, and the generated videos under `videos/{model}/{INDEX}.mp4`. Each writes a per-prompt JSON of the form `{judge_model}/{generator_model}/{INDEX}.json` under its output root.

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=<your AI Studio key>     # or GEMINI_API_KEY
```

Each evaluator uses the [google-genai](https://pypi.org/project/google-genai/) SDK against the AI Studio API. The audio agents additionally use `librosa`, `soundfile`, and `ffmpeg` for DSP measurements; the visual agents use `Pillow` and `ffmpeg` for frame extraction.

`ffmpeg` must be on `PATH`. On Ubuntu: `sudo apt install ffmpeg`.

## Running

From this directory:

```bash
# MLLM-as-judge baseline
python -m mllm.evaluate_videos \
    --video-dir videos \
    --rubric-dir rubrics \
    --output-root results/mllm \
    --run-in-parallel --max-workers 8

# Audio-tools agent (AV-Phys Agent)
python -m agent_audio.evaluate_videos \
    --video-dir videos --rubric-dir rubrics \
    --output-root results/agent_audio \
    --run-in-parallel --max-workers 8

# Visual-tools agent
python -m agent_visual.evaluate_videos \
    --video-dir videos --rubric-dir rubrics \
    --output-root results/agent_visual \
    --run-in-parallel --max-workers 8

# Audio + visual tools agent
python -m agent_av.evaluate_videos \
    --video-dir videos --rubric-dir rubrics \
    --output-root results/agent_av \
    --run-in-parallel --max-workers 8
```

By default each driver iterates the seven generative models reported in the paper (`Seedance-2.0`, `Kling-3.0-Omni`, `Veo-3.1`, `LTX-2.3`, `Ovi`, `JavisDiT++`, `MagiHuman`) and the full 321-prompt set. Restrict with `--generator-models` and `--prompt-ids` if needed.

## Output schema

Each per-prompt JSON contains:

- `prompt_id`, `judge_model`, `generator_model`
- `rubric` — the input rubric (verbatim copy of `rubrics/{INDEX}.json`)
- `per_statement` — list of `{statement_id, verdict, ...}` entries, one per `key_standards` statement plus the four `*_sa` checks
- `aggregated` — per-aspect AND aggregation: `video_sa`, `audio_sa`, `video_pc`, `audio_pc`, `av_pc`, `SA`, `PC`, `Both`
- For the agent variants: `description`, `tool_trace`, `agent_turns`, `include_tool_guide`, `elapsed_s`, `timestamp`

The MLLM baseline omits `description` / `tool_trace` (single call, no scaffolding).

## License

Released under MIT. See `../data_release/LICENSE` for the dataset license.
