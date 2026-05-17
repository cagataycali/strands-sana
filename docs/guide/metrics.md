# Metrics

Score generated images programmatically.

```python
from strands_sana import sana_metric_clip, sana_metric_imagereward

sana_metric_clip(image_path="duck.png",
                 prompt="a happy rubber duck on a motorcycle")
# {"score": 0.46, "metric": "clip-score"}

sana_metric_imagereward(image_path="duck.png", prompt="...")
# {"score": 1.34, "metric": "image-reward"}  # higher = better
```

## CLIPScore

OpenAI CLIP-vit-base-patch32 cosine similarity. Range 0..1, higher = better text-image alignment.

- ✅ Fast (~0.5s)
- ✅ Built-in to `[hf]` extra
- ❌ Not perfect — known biases (favors detail-rich images)

## ImageReward

```bash
pip install image-reward
```

Trained on human preferences. More expensive (~2s) but better correlated with quality.

## Use in inference scaling

```python
from strands_sana import sana_inference_scale

# Built-in CLIP scoring:
sana_inference_scale(prompt="...", n_samples=8, score_fn="clip")
```

For custom scorers, see [Inference Scaling](inference-scaling.md).

## Future: full upstream metrics

Upstream NVlabs/Sana ships:

- **GenEval** — compositional generation benchmark
- **DPG-Bench** — Dense Prompt Graph benchmark
- **FID** — Frechet Inception Distance
- **CLIP-T** — text alignment

Wiring these in as `sana_metric_*` is on the v0.5 roadmap. For now, use upstream's evaluation scripts directly.
