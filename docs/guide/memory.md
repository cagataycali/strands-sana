# Memory & Quantization

Run Sana on smaller GPUs. The biggest model (Sana-1.5 4.8B) normally needs 16 GB; with `mode='low'` it fits in 8 GB.

## Memory modes

```python
from strands_sana import sana_set_memory_mode

sana_set_memory_mode(mode="low", model="sana-1.6b-1024")
```

| mode | What it enables | When to use |
|---|---|---|
| `"fast"` | Nothing extra — full GPU residence | A100/H100, plenty of VRAM |
| `"balanced"` | `model_cpu_offload` | **Default for CUDA** |
| `"low"` | `sequential_cpu_offload` + `vae_tiling` + `attention_slicing` | 8-12 GB VRAM |
| `"ultra-low"` | low + future int4 quant | 4-8 GB VRAM (with `[quantization]`) |

## Quantization

```python
from strands_sana import sana_quantize

# Auto-probe: tries quanto → bitsandbytes → nunchaku
sana_quantize(bits=4, backend="auto", model="sana-1.6b-1024")
```

| Backend | Install | Status |
|---|---|---|
| `quanto` | `pip install optimum-quanto` | runtime quant |
| `bitsandbytes` | `pip install bitsandbytes` | requires reload-with-config |
| `nunchaku` | `pip install nunchaku` (SVDQuant 4-bit) | pre-quantized weights |

```bash
# Bundle install:
pip install 'strands-sana[quantization]'
```

## VAE swap (DC-AE-Lite)

Switch to a more efficient VAE without retraining the diffusion backbone:

```python
from strands_sana import sana_swap_vae

sana_swap_vae(
    vae_repo="mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers",
    model="sana-1.6b-1024",
)
```

DC-AE-Lite is ~30% faster decode with similar reconstruction quality.

## Cache management

Free GPU memory between runs:

```python
from strands_sana import sana_clear_cache

sana_clear_cache()
# {"freed_pipelines": 3}
```

## Hardware suggestions

| GPU | Recommended config |
|---|---|
| RTX 4090 (24 GB) | `mode="fast"` + Sana-1.5 4.8B |
| RTX 4070 Ti (12 GB) | `mode="balanced"` + Sana-1.6B |
| RTX 3060 (12 GB) | `mode="low"` + Sana-0.6B or Sprint |
| RTX 2060 (8 GB) | `mode="low"` + 4-bit quant + Sprint |
| Apple M-series | `mode="balanced"`, MPS auto-detected |
| CPU only | `mode="balanced"`, expect ~5min/image |
