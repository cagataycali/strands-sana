# Multi-Style Batch

Generate the same scene in different aesthetic styles using `sana_img2img`.

```python
from strands_sana import sana_generate, sana_img2img, sana_clear_cache

# 1. Establish a base image
base = sana_generate(
    prompt="a tranquil cabin in a misty forest at dawn",
    model="sana-1.6b-1024", steps=20, seed=42,
)
src = base["path"]

# 2. Restyle 4 different ways with Sprint Img2Img (much cheaper)
sana_clear_cache()
styles = [
    ("oil-painting",   "oil painting, thick brushstrokes, masterpiece"),
    ("anime",          "anime cel-shaded style, vibrant colors"),
    ("watercolor",     "watercolor painting, soft pastel washes"),
    ("cyberpunk",      "cyberpunk neon, holographic, rain"),
]
for style_name, style_prompt in styles:
    sana_img2img(
        prompt=style_prompt,
        image_path=src,
        model="sana-sprint-i2i-1.6b-1024",
        strength=0.55, steps=2,
        seed=42, output_dir=f"./out/{style_name}",
    )
```

## Why this works

- Base image: full Sana-1.6B at 20 steps for compositional anchor
- Restyles: Sana-Sprint Img2Img at 2 steps — ~10× cheaper
- `strength=0.55` keeps layout but allows aesthetic shift
- Same `seed` ensures all variants share latent structure

## Result on Thor

We did this in the [Gallery](../gallery.md#img2img-restyle-anything) — 9.3s per restyle after warmup.
