# AV-Phys Bench Code: Qwen Backbone

The same four evaluators as in [`code/`](../README.md), with Qwen3-Omni-30B-A3B-Instruct as the judge. Prompts, tools, and aggregation are unchanged, and the audio is passed as a separate 16 kHz mono track.

## Contents

```
qwen/
├── README.md
├── requirements.txt
├── mllm/            MLLM-as-judge baseline
├── agent_audio/     ReAct agent with audio tools (AV-Phys Agent)
├── agent_visual/    ReAct agent with frame inspection tools
└── agent_av/        ReAct agent with both audio DSP and visual tools
```

## Setup

```bash
pip install -r requirements.txt
```

`ffmpeg` must be on `PATH`.

Serve the judge with vLLM on a GPU machine:

```bash
pip install vllm==0.19.1
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir Qwen3-Omni-30B-A3B-Instruct
python -c "import json; open('chat_template.jinja', 'w').write(json.load(open('Qwen3-Omni-30B-A3B-Instruct/chat_template.json'))['chat_template'])"

vllm serve Qwen3-Omni-30B-A3B-Instruct \
    --served-model-name Qwen3-Omni-30B-A3B-Instruct \
    --max-model-len 65536 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --chat-template chat_template.jinja \
    --limit-mm-per-prompt '{"image": 10, "video": 1, "audio": 1}' \
    --seed 42
```

## Running

From this directory, with the server at `http://localhost:8000/v1` (change it with `--base-url`):

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

The other options and the output schema are the same as in [`code/README.md`](../README.md).
