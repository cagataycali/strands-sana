# strands-sana

NVIDIA Sana text-to-image diffusion exposed as Strands Agent tools.

## Install

```bash
pip install strands-sana
```

## Quickstart

```python
from strands import Agent
from strands_sana import sana_generate

agent = Agent(tools=[sana_generate])
agent("Generate a cyberpunk cityscape at night, photorealistic")
```

## What's covered

- ✅ **Sana 1.0 / 1.5** (10 checkpoints registered)
- ✅ **Sana-Sprint** — 1-2 step ultra-fast inference
- ✅ **PAG** — perturbed-attention guidance
- ✅ **ControlNet** (HED / Canny / depth)
- ✅ **2K / 4K** with auto VAE tiling
- ✅ **LoRA** load/unload at runtime
- ✅ **Quantization** scaffolding (Quanto / bitsandbytes / Nunchaku)
- ✅ **VAE swap** to DC-AE-Lite / DCAE-1.1
- ✅ **Inference scaling** (generate-K-and-pick-best, CLIP scoring)
- ✅ **Schedulers** (10 aliases: flow-match, DPM-Solver, Euler, …)
- ✅ **Metrics** (CLIPScore, ImageReward)
- ✅ **HuggingFace upload**
- ✅ **ComfyUI workflow export**
- ✅ **Streaming preview callback**

See [API reference](api.md) for the full tool list.

## Coverage vs upstream

We track gaps against [NVlabs/Sana](https://github.com/NVlabs/Sana) in
[DIFF.md](https://github.com/cagataycali/strands-sana/blob/main/DIFF.md).
