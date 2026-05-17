# Inference Scaling

Generate `K` candidates, pick the best by CLIP score (or NVILA when ready).

```python
from strands_sana import sana_inference_scale

result = sana_inference_scale(
    prompt="a cherry blossom branch in soft moonlight",
    model="sana-1.6b-1024",
    n_samples=4,
    score_fn="clip",    # or "first" to skip scoring
    seed_start=200,
)

print(result["best"])   # best candidate path + score
print(result["scores"]) # all 4 scores
```

## How it works

```mermaid
graph LR
    P[Prompt] --> G[Generate K with seeds]
    G --> S[Score each with CLIP]
    S --> B[Return best + all]

    style B fill:#5e35b1,stroke:#333,color:#fff
```

## Demo — 4 candidates

<div class="compare-grid" markdown>
<figure markdown>
  ![Cand 0](../assets/tool_scale_candidate_0.png)
  <figcaption>seed 200<br/>CLIP 0.431</figcaption>
</figure>
<figure markdown>
  ![Cand 1](../assets/tool_scale_candidate_1.png)
  <figcaption>seed 201<br/>CLIP 0.453</figcaption>
</figure>
<figure markdown>
  ![Winner](../assets/tool_scale_candidate_2.png)
  <figcaption><strong>seed 202<br/>CLIP 0.462 ← winner</strong></figcaption>
</figure>
<figure markdown>
  ![Cand 3](../assets/tool_scale_candidate_3.png)
  <figcaption>seed 203<br/>CLIP 0.459</figcaption>
</figure>
</div>

## When to use

- ✅ Production where quality matters more than latency
- ✅ Difficult prompts (compositional, off-distribution)
- ✅ Building a "best-of-N" agent loop
- ❌ Real-time UX (latency is K× single inference)

## Score functions

| `score_fn` | Method | Speed |
|---|---|---|
| `"clip"` | OpenAI CLIP-vit-base-patch32 | ~0.5s/image |
| `"first"` | No scoring, return seed_start | 0s |

For more sophisticated scoring (NVILA-2B, ImageReward, MPS), add a custom `score_fn` — see API reference.
