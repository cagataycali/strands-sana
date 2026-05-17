# strands-sana

NVIDIA Sana **inference + training** as [Strands Agents](https://github.com/strands-agents/sdk-python) tools.

[![PyPI](https://img.shields.io/pypi/v/strands-sana.svg)](https://pypi.org/project/strands-sana/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-72%20passing-brightgreen.svg)](tests/)
[![Tools](https://img.shields.io/badge/tools-33-blue.svg)](docs/api.md)

## What is Sana?

[Sana](https://github.com/NVlabs/Sana) is NVIDIA's efficient diffusion
family for high-resolution **image and video** generation:

- **SANA** — text-to-image, 4K in 20s, 20× smaller / 100× faster than FLUX-12B
- **SANA-1.5** — improved GenEval/CLIP scores
- **SANA-Sprint** — 1-2 step distilled (~0.1s/img on H100)
- **SANA-Video** — text-to-video & image-to-video (5s clips at 480/720p)
- **LongSANA** — real-time minute-long video at 27 FPS
- **Sol-RL** — NVFP4 rollout RL post-training
- **DC-AE** — 32× compression vs SDXL's 8×

`strands-sana` exposes the **entire upstream Sana family** — including
training scripts — as 33 clean Strands `@tool` functions.

## Quickstart

```bash
pip install strands-sana
```

### Image generation
```python
from strands import Agent
from strands_sana import sana_generate, sana_video_generate

agent = Agent(tools=[sana_generate, sana_video_generate])
agent("Generate a cyberpunk cityscape, then turn it into a 5-second video")
```

### Training (LoRA on your own data)
```python
from strands_sana import sana_train_lora

# Auto-detects ./Sana/ from `git clone https://github.com/NVlabs/Sana.git`
result = sana_train_lora(
    instance_data_dir="./my-photos/",
    instance_prompt="a photo of sks dog",
    max_train_steps=500,
    num_processes=4,
    dry_run=False,  # actually launch
)
print(result["pid"], result["command"])
```

## Coverage

✅ **33 tools, 16 models, 7 pipelines, 10 schedulers, 6 training jobs**

| Area | Status | Notes |
|---|:---:|---|
| Sana 1.0 / 1.5 (0.6B / 1.6B / 4.8B) | ✅ | 6 checkpoints |
| Sana-Sprint (T2I + Img2Img)         | ✅ | 1-2 step distilled |
| 2K / 4K with auto VAE tiling        | ✅ | |
| **SANA-Video (T2V + I2V)**          | ✅ | 4 video models incl. LongSANA |
| **LongSANA (real-time, 1-min)**     | ✅ | 27 FPS × 720 frames |
| PAG, ControlNet, LoRA, Inpaint      | ✅ / 🟡 inpaint via diffusers next minor |
| Quantization (int4 / int8)          | ✅ | Quanto / bnb / Nunchaku auto-probe |
| **Training: LoRA / DreamBooth**     | ✅ | `sana_train_lora` |
| **Training: full pretrain/finetune** | ✅ | `sana_train` (FSDP/DDP) |
| **Training: sCM-LADD distillation** | ✅ | `sana_train_scm_ladd` |
| **Training: Sol-RL (NVFP4)**        | ✅ | `sana_train_solrl` |
| **Training: SANA-Video FSDP**       | ✅ | `sana_train_video` (+ chunked) |
| **Training: LongSANA**              | ✅ | `sana_train_longsana` |
| Schedulers (10 aliases)             | ✅ | |
| Streaming preview, ComfyUI export   | ✅ | |
| Metrics (CLIP / ImageReward)        | ✅ | |
| HF upload, SGLang serve             | ✅ | |

Full gap analysis: see [DIFF.md](DIFF.md).

## Tools (33)

**Inference — images** (7): `sana_generate` · `sana_batch` · `sana_load_model` · `sana_sprint_generate` · `sana_inpaint` · `sana_controlnet_generate` · `sana_img2img`
**Inference — videos** (2): `sana_video_generate` · `sana_image_to_video`
**Adapters** (2): `sana_load_lora` · `sana_unload_loras`
**Memory & cache** (2): `sana_set_memory_mode` · `sana_clear_cache`
**Schedulers** (2): `sana_set_scheduler` · `sana_list_schedulers`
**Optimization** (2): `sana_quantize` · `sana_swap_vae`
**Quality** (3): `sana_inference_scale` · `sana_metric_clip` · `sana_metric_imagereward`
**DX** (3): `sana_enhance_prompt` · `sana_export_comfyui_workflow` · `sana_safety_check`
**Distribution** (3): `sana_upload_to_hf` · `sana_serve` · `sana_prefetch_model`
**Training** (7): `sana_train_lora` · `sana_train` · `sana_train_scm_ladd` · `sana_train_solrl` · `sana_train_video` · `sana_train_longsana` · `sana_list_training_configs`

See [docs/api.md](docs/api.md) for full signatures.

## Architecture

```
strands_sana/
├── tools/
│   ├── generate.py    # 13 core inference tools
│   ├── extras.py      # 10 schedulers, quant, metrics, …
│   ├── video.py       # T2V + I2V
│   ├── img2img.py     # Sprint Img2Img
│   └── training.py    # 7 training tools (wrap upstream Sana scripts)
├── pipeline/          # SanaPipelineWrapper (7 pipeline kinds)
├── models/            # 16-model registry, list_models, tag/kind filters
└── utils/             # io, prompt enhancement, COMPLEX_HUMAN_INSTRUCTION
```

## Training: how it works

Training tools wrap upstream `Sana/train_scripts/*.py` via subprocess.

1. `git clone https://github.com/NVlabs/Sana.git` next to your project
   (or set `SANA_ROOT=/path/to/Sana`)
2. Call any `sana_train_*` tool with `dry_run=True` to preview the command
3. Pass `dry_run=False` to launch (returns PID for monitoring)

## License

Apache-2.0. Sana model weights + training assets governed by NVIDIA's license.

## Acknowledgments

- [NVIDIA Sana](https://github.com/NVlabs/Sana) — model & training code
- [HuggingFace `diffusers`](https://github.com/huggingface/diffusers) — pipeline implementations
- [Strands Agents](https://github.com/strands-agents/sdk-python) — agent framework
