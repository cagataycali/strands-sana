# strands-sana

NVIDIA Sana text-to-image diffusion as [Strands Agents](https://github.com/strands-agents/sdk-python) tools.

[![PyPI version](https://img.shields.io/pypi/v/strands-sana.svg)](https://pypi.org/project/strands-sana/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen.svg)](tests/)

## What is Sana?

[Sana](https://github.com/NVlabs/Sana) is NVIDIA's efficient high-resolution text-to-image diffusion model: 1024×1024 in <1s, 2K in 4s, 4K in 20s, **20× smaller and 100× faster than FLUX-12B**.

Key tech:
- **DC-AE** — 32× compression vs SDXL's 8×
- **Linear DiT** — linear attention scales to 4K
- **Decoder-only LLM text encoder** — Gemma-2 instead of T5
- **Flow matching** — faster, more stable than DDPM

`strands-sana` wraps the entire upstream Sana family as a clean set of `@tool` functions, ready to drop into any Strands Agent.

## Quickstart

```bash
pip install strands-sana
```

```python
from strands import Agent
from strands_sana import sana_generate

agent = Agent(tools=[sana_generate])
agent("Generate a cyberpunk duck riding a neon motorcycle, 1024x1024")
```

## Coverage

✅ **23 tools, 10 model checkpoints, 5 pipeline variants, 10 schedulers**

| Capability | Status |
|---|:---:|
| Sana 1.0 / 1.5 (0.6B / 1.6B / 4.8B) | ✅ |
| Sana-Sprint (1-2 step distilled) | ✅ |
| 2K / 4K with auto VAE tiling | ✅ |
| PAG (Perturbed Attention Guidance) | ✅ |
| ControlNet | ✅ |
| Inpainting | 🟡 (auto-detects diffusers support) |
| LoRA load/unload | ✅ |
| Quantization (int4 / int8) | ✅ |
| VAE swap (DC-AE-Lite / DCAE-1.1) | ✅ |
| Inference scaling (CLIP-pick-best) | ✅ |
| Schedulers (flow-match, DPM-Solver, Euler, …) | ✅ |
| Streaming preview callback | ✅ |
| ComfyUI workflow export | ✅ |
| Metrics (CLIPScore, ImageReward) | ✅ |
| HuggingFace upload | ✅ |
| Memory modes (fast / balanced / low / ultra-low) | ✅ |
| Safety check (keyword + ShieldGemma path) | 🟡 |

Full gap analysis vs upstream: see [DIFF.md](DIFF.md).

## Models

| Alias | Pipeline | Resolution | Params | HF Repo |
|---|---|:---:|:---:|---|
| `sana-0.6b-512`        | t2i    | 512  | 590M | Efficient-Large-Model/Sana_600M_512px_diffusers |
| `sana-0.6b-1024`       | t2i    | 1024 | 590M | Efficient-Large-Model/Sana_600M_1024px_diffusers |
| `sana-1.6b-1024`       | t2i    | 1024 | 1.6B | Efficient-Large-Model/Sana_1600M_1024px_diffusers |
| `sana-1.6b-multiling`  | t2i    | 1024 | 1.6B | Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers |
| `sana-1.5-1.6b-1024`   | t2i    | 1024 | 1.6B | Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers |
| `sana-1.5-4.8b-1024`   | t2i    | 1024 | 4.8B | Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers |
| `sana-1.6b-2k`         | t2i    | 2048 | 1.6B | Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers |
| `sana-1.6b-4k`         | t2i    | 4096 | 1.6B | Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers |
| `sana-sprint-0.6b-1024`| sprint | 1024 | 590M | Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers |
| `sana-sprint-1.6b-1024`| sprint | 1024 | 1.6B | Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers |

## Tools (21)

**Core**: `sana_generate` · `sana_batch` · `sana_load_model`
**Variants**: `sana_sprint_generate` · `sana_inpaint` · `sana_controlnet_generate`
**Adapters**: `sana_load_lora` · `sana_unload_loras`
**Memory**: `sana_set_memory_mode` · `sana_clear_cache`
**Schedulers**: `sana_set_scheduler` · `sana_list_schedulers`
**Optimization**: `sana_quantize` · `sana_swap_vae`
**Quality**: `sana_inference_scale` · `sana_metric_clip` · `sana_metric_imagereward`
**DX**: `sana_enhance_prompt` · `sana_export_comfyui_workflow` · `sana_safety_check`
**Distribution**: `sana_upload_to_hf`

See [docs/api.md](docs/api.md) for full signatures.

## Architecture

```
strands_sana/
├── tools/
│   ├── generate.py    # 14 core tools
│   └── extras.py      # 9 extras (schedulers, quantize, metrics, …)
├── pipeline/          # SanaPipelineWrapper, callbacks
├── models/            # registry, list_models, tag/kind filters
└── utils/             # io, prompt enhancement, COMPLEX_HUMAN_INSTRUCTION
```

## License

Apache-2.0. Sana model weights are governed by NVIDIA's license.

## Acknowledgments

- [NVIDIA Sana](https://github.com/NVlabs/Sana) — original model & paper
- [HuggingFace `diffusers`](https://github.com/huggingface/diffusers) — pipeline implementations
- [Strands Agents](https://github.com/strands-agents/sdk-python) — agent framework
