# Video Pipeline

Full pipeline: generate hero image → animate → score → output.

```python
from strands_sana import (
    sana_generate, sana_image_to_video, sana_metric_clip, sana_clear_cache,
)

PROMPT = "a serene koi pond at sunrise, cherry blossoms drifting"

# 1. Generate base image with best quality
sana_clear_cache()
img = sana_generate(prompt=PROMPT, model="sana-1.5-1.6b-1024",
                    steps=20, seed=42, output_dir="./out")

# 2. Score it for sanity
score = sana_metric_clip(image_path=img["path"], prompt=PROMPT)
print(f"CLIP alignment: {score['score']:.3f}")

if score["score"] < 0.30:
    print("Re-rolling...")
    # could fall through to inference_scale here
else:
    # 3. Animate
    sana_clear_cache()
    video = sana_image_to_video(
        image_path=img["path"],
        prompt=PROMPT + ", soft camera drift",
        model="sana-video-i2v-480",
        frames=121, steps=15, seed=42,
        output_dir="./out",
    )
    print("Final video:", video["path"])
```

## Strands agent variant

```python
from strands import Agent
from strands_sana import (
    sana_generate, sana_image_to_video, sana_metric_clip, sana_clear_cache,
)

agent = Agent(tools=[
    sana_generate, sana_image_to_video, sana_metric_clip, sana_clear_cache,
])

agent("""
Generate an image of a serene koi pond at sunrise. Score it with CLIP.
If alignment is above 0.3, animate it as a 5-second clip. Otherwise re-roll.
""")
```

The agent figures out the right tool sequence on its own.
