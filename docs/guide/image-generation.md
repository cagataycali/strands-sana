# Image Generation

The core tool is `sana_generate`. It auto-dispatches to PAG when `pag_scale > 0`, handles aspect-ratio binning, and supports up to 4K with VAE tiling.

```python
sana_generate(prompt: str, ...) -> dict
```

## Basic call

```python
from strands_sana import sana_generate

result = sana_generate(
    prompt="a serene koi pond, ink wash painting",
    model="sana-1.6b-1024",
    steps=20,
    seed=42,
)
```

## All parameters

| arg | type | default | notes |
|---|---|---|---|
| `prompt` | `str` | required | text description |
| `negative_prompt` | `str` | `""` | what to avoid |
| `model` | `str?` | `None` | alias from registry |
| `width` / `height` | `int?` | model resolution | snapped to bucket |
| `steps` | `int?` | per-model (20 default) | denoising steps |
| `guidance_scale` | `float?` | per-model (4.5) | CFG strength |
| `seed` | `int?` | `None` | reproducibility |
| `pag_scale` | `float` | `0.0` | >0 → SanaPAGPipeline |
| `use_complex_instruction` | `bool` | `False` | prepend Gemma-2 prompt expander |
| `use_resolution_binning` | `bool` | `True` | snap to trained bucket |
| `num_images` | `int` | `1` | per prompt |
| `output_dir` | `str?` | `./generated` | save directory |

## Multi-image batch

```python
sana_generate(
    prompt="a clay figurine of a frog",
    model="sana-0.6b-512",
    num_images=4,
    seed=20,
)
# Returns: { "paths": [path1, path2, path3, path4], "count": 4, ... }
```

## Multiple prompts

For different prompts, use `sana_batch`:

```python
from strands_sana import sana_batch

sana_batch(
    prompts=["a duck", "a frog", "a cat"],
    model="sana-0.6b-512",
    seed=42,
)
```

## Different aspect ratios

```python
sana_generate(prompt="a cinematic landscape",
              width=1280, height=768, model="sana-1.6b-1024")
```

`use_resolution_binning=True` (default) snaps to the closest trained bucket — for Sana-1024 buckets are e.g. `(1024, 1024)`, `(1280, 768)`, `(768, 1280)`, etc.

## With Gemma-2 prompt expansion

Sana uses a Gemma-2 LLM as text encoder. Pass `use_complex_instruction=True` to prepend the official prompt-expansion template — short prompts get fleshed out automatically.

```python
sana_generate(
    prompt="a duck",   # short
    model="sana-1.6b-1024",
    use_complex_instruction=True,  # expanded server-side via Gemma-2
)
```
