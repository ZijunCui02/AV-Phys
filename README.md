# Do Joint Audio-Video Generation Models Understand Physics?

[![arXiv](https://img.shields.io/badge/arXiv-2605.07061-b31b1b.svg)](https://arxiv.org/abs/2605.07061)
[![Project Page](https://img.shields.io/badge/Project%20Page-zijuncui.com%2FAV--Phys-blue)](https://zijuncui.com/AV-Phys/)
[![HF Dataset](https://img.shields.io/badge/Dataset-AV--Phys--Bench-yellow)](https://huggingface.co/datasets/ZijunCui/AV-Phys-Bench)

#### [Zijun Cui](https://zijuncui.com)<sup>1,*</sup>, [Xiulong Liu](https://dragonliu1995.github.io/)<sup>2,*</sup>, [Hao Fang](https://apexhao.github.io/)<sup>2,*</sup>, Mingwei Xu<sup>2</sup>, [Jiageng Liu](https://jiagengliu02.github.io/)<sup>3</sup>, [Zexin Xu](https://zexinxu.com/)<sup>1</sup>, Weiguo Pian<sup>1</sup>, Shijian Deng<sup>1</sup>, Feiyu Du<sup>1</sup>, Chenming Ge<sup>2</sup>, [Yapeng Tian](https://www.yapengtian.com/)<sup>1,†</sup>

<sup>1</sup> University of Texas at Dallas &nbsp;&nbsp; <sup>2</sup> University of Washington &nbsp;&nbsp; <sup>3</sup> University of California, Los Angeles

<sup>*</sup> Equal contribution. <sup>†</sup> Corresponding author.

***

🎧 **Please put on headphones.** AV-Phys Bench is about audio as much as it is about video, and many of the failures shown here are easier to hear than they are to see.

https://github.com/user-attachments/assets/7fe6d024-34ad-4246-b8c1-ba03c0a4a3be

***

> "A speaker plays **music** at **low volume**, sounding quiet and thin. Then the volume knob is **turned up** gradually until the **music** fills the room."
>
> Prompt [C2-2-20](https://zijuncui.com/AV-Phys/videos/C2-2-20/)

<table>
<tr>
<th width="33%" align="center">Seedance 2.0 ✓</th>
<th width="33%" align="center">Kling 3.0 Omni ✗</th>
<th width="33%" align="center">Veo 3.1 ✗</th>
</tr>
<tr valign="top">
<td><video src="https://github.com/user-attachments/assets/c1521113-736c-4495-bf65-3c2b6e756c21" controls width="100%"></video></td>
<td><video src="https://github.com/user-attachments/assets/0205b6ea-2fec-4e9f-b7a1-148bf22c438c" controls width="100%"></video></td>
<td><video src="https://github.com/user-attachments/assets/7261ecce-d264-46f9-802a-a15e69c26576" controls width="100%"></video></td>
</tr>
</table>

## About

AV-Phys Bench is the first comprehensive benchmark for evaluating physical commonsense in joint audio-video generation. It tests how well a model preserves physical commonsense as a scene evolves over time, across three scene categories:

* **C1 Steady State**: source, action, and environment all stay fixed.
* **C2 Event Transition**: a discrete action changes the physical state of the source.
* **C3 Environment Transition**: the source is held fixed and the propagation path between source and listener changes.

Each scene category also includes an Anti-AV-Physics subcategory that deliberately violates a physical principle, probing whether models possess generative physics knowledge or merely encode physically consistent priors. Seven generators are evaluated by human raters, an MLLM-as-judge baseline, and the AV-Phys Agent. The full leaderboard, the per-prompt video gallery, and the rubrics are on the [project page](https://zijuncui.com/AV-Phys/).

## Project Components

| Component | Location | Description |
|:---|:---|:---|
| **AV-Phys Bench Dataset** | [HuggingFace](https://huggingface.co/datasets/ZijunCui/AV-Phys-Bench) | Prompts, per-prompt rubrics, generated videos from seven models, and human ratings |
| **Project Page** | [zijuncui.com/AV-Phys](https://zijuncui.com/AV-Phys/) | Live leaderboard, video gallery, per-prompt rubric details |

## Getting Started

Install the evaluator dependencies and provide a Google AI Studio key.

```bash
cd code
pip install -r requirements.txt
export GOOGLE_API_KEY=<your AI Studio key>
```

Download the dataset from HuggingFace. This brings the prompts, the rubrics, and the seven sets of generated videos in one step.

```bash
huggingface-cli download ZijunCui/AV-Phys-Bench --repo-type=dataset --local-dir data_release
```

Run any of the four evaluators against the released generations.

```bash
cd code

python -m mllm.evaluate_videos \
    --video-dir ../data_release/videos \
    --rubric-dir ../data_release/rubrics \
    --output-root results/mllm \
    --run-in-parallel --max-workers 8

python -m agent_av.evaluate_videos \
    --video-dir ../data_release/videos \
    --rubric-dir ../data_release/rubrics \
    --output-root results/agent_av \
    --run-in-parallel --max-workers 8
```

Each per-prompt output JSON includes the verdict on every rubric statement, the aggregated pass per aspect (`video_sa`, `audio_sa`, `video_pc`, `audio_pc`, `av_pc`), and the combined `SA`, `PC`, and `Both` scores used in the paper. See [code/README.md](./code/README.md) for the full output schema and the audio-only and visual-only agent variants.

## Score Your Own Model

Generate one MP4 per prompt and place the videos under your own model directory.

```
data_release/videos/<your-model>/<INDEX>.mp4
```

Then point any evaluator at it.

```bash
python -m agent_av.evaluate_videos \
    --video-dir ../data_release/videos \
    --rubric-dir ../data_release/rubrics \
    --generator-models <your-model> \
    --output-root results/agent_av
```

## Repository Layout

```
AV-Phys/
├── code/      AV-Phys Agent and baseline evaluators
├── docs/      Built static site served by GitHub Pages
├── src/       Project page sources (CSS, JS, assets)
└── scripts/   Site builder
```

## Limitations

We strive to maintain the highest quality in our benchmark, but some imperfections may persist. If you notice any, we encourage you to reach out and share your valuable feedback!

## Citation

```bibtex
@article{cui2026joint,
  title={Do Joint Audio-Video Generation Models Understand Physics?},
  author={Cui, Zijun and Liu, Xiulong and Fang, Hao and Xu, Mingwei and Liu, Jiageng and Xu, Zexin and Pian, Weiguo and Deng, Shijian and Du, Feiyu and Ge, Chenming and others},
  journal={arXiv preprint arXiv:2605.07061},
  year={2026}
}
```
