# Video Generation

Two tools cover all video use cases: **text-to-video** (`sana_video_generate`) and **image-to-video** (`sana_image_to_video`).

## Text-to-Video

```python
from strands_sana import sana_video_generate

sana_video_generate(
    prompt="a slow cinematic pan over snowy mountains, golden hour",
    model="sana-video-2b-480",   # 480p · 5s clips
    steps=15,
    frames=121,                   # ~5s @ 24fps
    seed=42,
)
```

### Models for video

| Model | Resolution | Default frames | Notes |
|---|:---:|:---:|---|
| `sana-video-2b-480` | 480 | 121 (5s) | **Default** — best speed/quality |
| `sana-video-2b-720` | 720 | 121 (5s) | 720p with LTX-VAE refiner |
| `longsana-2b-480` | 480 | 720 (~27s @ 27fps) | Real-time minute-long |

### Output formats

`sana_video_generate` writes:

1. **MP4** if you have `imageio-ffmpeg` (`pip install 'strands-sana[video]'`)
2. Animated **WebP** as fallback (no extra deps)

```python
result = sana_video_generate(prompt="...", frames=49)
# {
#   "path": "./generated/sana_video_<ts>.mp4",
#   "frames": 49, "fps": 24, "duration_s": 2.04,
# }
```

## Image-to-Video

Animate a still:

```python
from strands_sana import sana_image_to_video

sana_image_to_video(
    image_path="hero.png",
    prompt="zoom out slowly, leaves swirling in the wind",
    model="sana-video-i2v-480",
    frames=49,
    seed=42,
)
```

The source image is encoded into the first frame's latent, then the diffusion model unrolls subsequent frames.

## Live demo

<p align="center">
  <img src="../assets/tool_image_to_video.gif" width="60%" alt="I2V demo" />
</p>

## Resolution constraints

SANA-Video requires `(height, width)` divisible by 32. The wrapper snaps automatically and disables binning to ensure your sizes are honored.

```python
sana_video_generate(prompt="...", height=480, width=480)  # OK
sana_video_generate(prompt="...", height=300)             # → snaps to 288
```

## Long videos

For minute-long real-time clips:

```python
sana_video_generate(
    prompt="a calm zen garden, soft camera drift",
    model="longsana-2b-480",
    frames=720,    # 27s @ 27fps
    steps=4,       # LongSANA is distilled — few steps
)
```
