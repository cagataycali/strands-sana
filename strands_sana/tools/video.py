"""SANA-Video tools (T2V + I2V) — v0.4.0."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from strands import tool

from ..pipeline.sana_pipeline import get_pipeline
from ..utils.io import load_image, ensure_output_dir
from ..utils.prompts import COMPLEX_HUMAN_INSTRUCTION

logger = logging.getLogger(__name__)


def _save_video_frames(frames, output_dir: Optional[str], fps: int,
                       prefix: str = "sana_video") -> str:
    """Save a list of PIL.Image frames as an MP4 (preferred) or WebP fallback."""
    out = ensure_output_dir(output_dir)
    import time
    base = out / f"{prefix}_{int(time.time() * 1000)}"

    # Try MP4 via diffusers / imageio first
    try:
        from diffusers.utils import export_to_video
        path = str(base.with_suffix(".mp4"))
        export_to_video(frames, path, fps=fps)
        return path
    except Exception as e:
        logger.warning(f"export_to_video failed, falling back to WebP: {e}")

    # Fallback: animated WebP
    path = str(base.with_suffix(".webp"))
    if frames:
        frames[0].save(
            path, save_all=True, append_images=frames[1:],
            duration=int(1000 / fps), loop=0,
        )
    return path


# ────────────────────────────────────────────────────────────────────
# Text-to-Video
# ────────────────────────────────────────────────────────────────────
@tool
def sana_video_generate(
    prompt: str,
    negative_prompt: str = "",
    model: str = "sana-video-2b-480",
    width: Optional[int] = None,
    height: Optional[int] = None,
    frames: Optional[int] = None,
    fps: int = 24,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    use_complex_instruction: bool = False,
) -> dict:
    """Generate a video from a text prompt using SANA-Video.

    Models:
      - sana-video-2b-256 — fastest (49 frames @ 24fps ≈ 2s clip)
      - sana-video-2b-480 — default 480p, 121 frames ≈ 5s clip
      - sana-video-2b-720 — 720p with LTX-VAE refiner
      - longsana-2b-480  — minute-long, 27fps real-time (720 frames)

    Args:
        prompt: Text description.
        frames: Number of frames (model default).
        fps: Output framerate.

    Returns:
        Dict with `path` (mp4 or webp), `frames`, `model`, `prompt`.
    """
    pipe = get_pipeline(model_name=model)
    if pipe.kind != "video":
        return {"status": "error",
                "error": f"Model '{model}' is kind='{pipe.kind}', expected 'video'"}

    out_frames = pipe.generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height, width=width,
        frames=frames,
        seed=seed,
        num_videos=1,
        complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION if use_complex_instruction else None,
    )
    # out_frames is list of list (one per video) — first batch
    video = out_frames[0] if out_frames and isinstance(out_frames[0], list) else out_frames
    actual_fps = fps or pipe.info.fps
    path = _save_video_frames(video, output_dir, fps=actual_fps)

    return {
        "status": "success",
        "path": path,
        "model": pipe.model_name,
        "prompt": prompt,
        "frames": len(video) if hasattr(video, "__len__") else 0,
        "fps": actual_fps,
        "duration_s": (len(video) / actual_fps) if hasattr(video, "__len__") else 0,
    }


# ────────────────────────────────────────────────────────────────────
# Image-to-Video
# ────────────────────────────────────────────────────────────────────
@tool
def sana_image_to_video(
    image_path: str,
    prompt: str,
    negative_prompt: str = "",
    model: str = "sana-video-i2v-480",
    width: Optional[int] = None,
    height: Optional[int] = None,
    frames: Optional[int] = None,
    fps: int = 24,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Animate a still image into a video clip.

    Args:
        image_path: Path/URL of the source image.
        prompt: Description of the motion.
    """
    pipe = get_pipeline(model_name=model)
    if pipe.kind != "image-to-video":
        return {"status": "error",
                "error": f"Model '{model}' is kind='{pipe.kind}', expected 'image-to-video'"}

    img = load_image(image_path)
    out_frames = pipe.generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height, width=width,
        frames=frames,
        seed=seed,
        num_videos=1,
        image=img,
    )
    video = out_frames[0] if out_frames and isinstance(out_frames[0], list) else out_frames
    actual_fps = fps or pipe.info.fps
    path = _save_video_frames(video, output_dir, fps=actual_fps, prefix="sana_i2v")

    return {
        "status": "success",
        "path": path,
        "model": pipe.model_name,
        "prompt": prompt,
        "source_image": image_path,
        "frames": len(video) if hasattr(video, "__len__") else 0,
        "fps": actual_fps,
    }
