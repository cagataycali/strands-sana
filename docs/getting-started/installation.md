# Installation

## Basic install

```bash
pip install strands-sana
```

This pulls in `diffusers`, `torch`, `transformers`, `accelerate`, `pyyaml`.

## With GPU acceleration

You'll want a CUDA-compatible torch wheel for non-trivial inference:

```bash
# CUDA 12.1
pip install torch>=2.4 --extra-index-url https://download.pytorch.org/whl/cu121

pip install strands-sana
```

On Apple Silicon, MPS works out of the box — no extra wheel needed.

## Optional extras

```bash
pip install 'strands-sana[hf]'           # HF Hub + safetensors
pip install 'strands-sana[video]'        # imageio + imageio-ffmpeg (MP4 export)
pip install 'strands-sana[lora]'         # peft for LoRA support
pip install 'strands-sana[quantization]' # bitsandbytes + optimum-quanto
pip install 'strands-sana[dev]'          # pytest, ruff, black, mypy
pip install 'strands-sana[docs]'         # mkdocs-material
```

## With training support

Training tools shell out to upstream NVlabs/Sana scripts. Either:

```bash
git clone https://github.com/NVlabs/Sana.git
# Tools auto-detect ./Sana
```

…or set `SANA_ROOT`:

```bash
export SANA_ROOT=/path/to/Sana
```

## Verify install

```python
import strands_sana
print(strands_sana.__version__)

from strands_sana import sana_load_model
print(sana_load_model(model="list"))
```

You should see 15 models registered.

## Hardware requirements

| Model | VRAM (default) | VRAM (`mode='low'`) |
|-------|---:|---:|
| Sana 0.6B | 4 GB | 2 GB |
| Sana 1.6B / 1.5 1.6B | 8 GB | 4 GB |
| Sana 1.5 4.8B | 16 GB | 8 GB |
| Sana 1.6B @ 4K | 22 GB | 8 GB |
| SANA-Video 2B | 12 GB | 6 GB |
| LongSANA | 20 GB | 10 GB |

CPU-only works for tiny experiments but is slow (~5 min/image).
