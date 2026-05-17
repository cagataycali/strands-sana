# strands-sana

NVIDIA Sana text-to-image diffusion for [Strands Agents](https://github.com/strands-agents/sdk-python).

[![PyPI version](https://img.shields.io/pypi/v/strands-sana.svg)](https://pypi.org/project/strands-sana/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## What is Sana?

[Sana](https://github.com/NVlabs/Sana) is NVIDIA's efficient high-resolution text-to-image diffusion model. It generates 1024×1024 (and beyond) images at remarkable speed using:

- **DC-AE (Deep Compression AutoEncoder)** — 32× compression vs SDXL's 8×
- **Linear DiT (Diffusion Transformer)** — linear attention for high resolution
- **Decoder-only LLM as text encoder** — Gemma-2 instead of T5
- **Flow matching** — faster, more stable than DDPM

`strands-sana` wraps Sana as a Strands Agent tool, so any agent can generate images via natural language.

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

## Models

| Model | Resolution | Params | HF Hub |
|-------|-----------|--------|--------|
| Sana-0.6B | 512/1024 | 590M | `Efficient-Large-Model/Sana_600M_512px` |
| Sana-1.6B | 1024/2048 | 1.6B  | `Efficient-Large-Model/Sana_1600M_1024px` |
| Sana-1.6B-MultiLing | 1024 | 1.6B | `Efficient-Large-Model/Sana_1600M_1024px_MultiLing` |

## Tools

- `sana_generate` — text → image
- `sana_batch` — batch text-to-image
- `sana_load_model` — switch between checkpoints

## Architecture

```
strands_sana/
├── tools/          # @tool-decorated functions for agents
├── pipeline/       # Sana inference wrapper (diffusers)
├── models/         # Model registry & lazy loading
└── utils/          # Image I/O, prompt processing
```

## License

Apache-2.0. Sana model weights are governed by NVIDIA's license.

## Acknowledgments

- [NVIDIA Sana](https://github.com/NVlabs/Sana) — original model & paper
- [Strands Agents](https://github.com/strands-agents/sdk-python) — agent framework
