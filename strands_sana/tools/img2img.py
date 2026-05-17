"""Image-to-Image generation via Sana-Sprint Img2Img (v0.4.0)."""
from __future__ import annotations

from typing import Optional

from strands import tool

from ..pipeline.sana_pipeline import get_pipeline
from ..utils.io import load_image, save_image


@tool
def sana_img2img(
    prompt: str,
    image_path: str,
    model: str = "sana-sprint-i2i-1.6b-1024",
    strength: float = 0.6,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Sana-Sprint image-to-image: re-render a source image with new prompt.

    Args:
        prompt: New description.
        image_path: Path/URL of source image.
        strength: 0..1 — 0 = no change, 1 = full regenerate.
        steps: 4 is the Sprint sweet spot for img2img.
    """
    pipe = get_pipeline(model_name=model)
    if pipe.kind != "sprint-i2i":
        return {"status": "error",
                "error": f"Model '{model}' is kind='{pipe.kind}', expected 'sprint-i2i'"}

    src = load_image(image_path)
    images = pipe.generate(
        prompt=prompt,
        image=src,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width, height=height,
        seed=seed,
        num_images=1,
    )
    path = save_image(images[0], output_dir=output_dir, prefix="i2i")
    return {
        "status": "success",
        "path": path,
        "model": pipe.model_name,
        "prompt": prompt,
        "source_image": image_path,
        "strength": strength,
    }
