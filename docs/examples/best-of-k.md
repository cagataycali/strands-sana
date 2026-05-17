# Best of K

Generate K candidates, pick the best by CLIP, optionally animate it.

```python
from strands_sana import sana_inference_scale, sana_image_to_video

# 1. Generate 8 candidates with same prompt, 8 different seeds
result = sana_inference_scale(
    prompt="a futuristic robot in a Japanese garden",
    model="sana-1.6b-1024",
    n_samples=8,
    seed_start=1000,
    score_fn="clip",
    steps=15,
    output_dir="./out",
)

print(f"Best: seed={result['best']['seed']} score={result['scores'][result['best']['index']]['score']:.3f}")

# 2. Animate the winner
sana_image_to_video(
    image_path=result["best"]["path"],
    prompt="slow zoom out, leaves rustle in the breeze",
    model="sana-video-i2v-480",
    frames=49, steps=10,
    output_dir="./out",
)
```

## Why best-of-K works

CLIP scoring is cheap (~0.5s per image), generation is the expensive part. Burning 8× compute on candidates and picking the winner often beats one careful inference.

## Cost vs benefit

| Approach | Compute | Quality |
|---|---|---|
| Single Sana-1.5 4.8B (20 steps) | 1× | High |
| Best-of-4 with Sana-1.6B | 4× | Often higher |
| Best-of-16 with Sana-Sprint | ~1.6× | Variable |

## Variations

- `score_fn="first"` — skip scoring, useful for batch generation
- Custom scorer: monkeypatch `_clip_score` in `extras.py` for your own ranker
- Combine with PAG: `sana_inference_scale(..., pag_scale=2.0)` (after wiring)
