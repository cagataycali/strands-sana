# PAG — Perturbed Attention Guidance

PAG ([Perturbed Attention Guidance](https://arxiv.org/abs/2403.17377)) is a CFG-style guidance technique that improves quality without text conditioning. It costs ~2× compute.

```python
from strands_sana import sana_generate

sana_generate(
    prompt="a stained glass dragon",
    model="sana-1.5-1.6b-1024",
    pag_scale=2.0,        # > 0 → switches to SanaPAGPipeline
    steps=15,
    seed=42,
)
```

## How `pag_scale` interacts with `guidance_scale`

| `pag_scale` | `guidance_scale` | Pipeline |
|:---:|:---:|---|
| 0 | any | `SanaPipeline` (regular CFG) |
| > 0 | any | `SanaPAGPipeline` (PAG + CFG) |

Recommended: `pag_scale=2.0`, keep `guidance_scale=4.5` (default).

## Demo

<p align="center">
  <img src="../assets/tool_pag.png" width="60%" alt="PAG output" />
</p>
<p align="center"><sub>Sana-1.5 1.6B + PAG (`pag_scale=2.0`) · 15 steps · 62s on Thor</sub></p>

## When to use PAG

- ✅ Subjects with intricate detail (jewelry, glass, fur)
- ✅ Off-distribution prompts where CFG alone overshoots
- ❌ Sprint models (already distilled)
- ❌ Speed-critical pipelines (~2× compute)
