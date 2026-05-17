# Your first video

This produces the cyberpunk teaser at the top of the homepage.

## Code

```python
from strands_sana import sana_video_generate

result = sana_video_generate(
    prompt=("neon rain falling on the streets of a cyberpunk city, "
            "reflections in puddles, holographic billboards"),
    model="sana-video-2b-480",
    steps=10,        # fast
    frames=49,       # ~2s @ 24fps
    seed=99,
    output_dir="./out",
)
print(result)
```

## What you get

```json
{
  "status": "success",
  "path": "./out/sana_video_<ts>.mp4",
  "model": "sana-video-2b-480",
  "frames": 49,
  "fps": 24,
  "duration_s": 2.04
}
```

<p align="center">
  <img src="../assets/hero_video_cyberpunk.gif" width="60%" alt="Cyberpunk video" />
</p>

## MP4 encoding

For real MP4 output (instead of WebP fallback), install the `[video]` extra:

```bash
pip install 'strands-sana[video]'   # imageio + imageio-ffmpeg
```

Without it, `strands-sana` writes animated WebP — works everywhere but harder to share.

→ Next: **[Image-to-video](../guide/video-generation.md#image-to-video)** to animate a still.
