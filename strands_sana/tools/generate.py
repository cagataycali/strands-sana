"""Strands @tool-decorated functions for Sana text-to-image."""
from __future__ import annotations

from typing import List, Optional

from strands import tool

from ..pipeline.sana_pipeline import get_pipeline
from ..models.registry import SANA_MODELS, default_model
from ..utils.io import save_image


@tool
def sana_generate(
    prompt: str,
    negative_prompt: str = "",
    model: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: int = 20,
    guidance_scale: float = 4.5,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Generate an image from a text prompt using NVIDIA Sana.

    Args:
        prompt: Text description of the image to generate.
        negative_prompt: What to avoid in the image.
        model: Sana model alias (e.g. 'sana-1.6b-1024'). Defaults to sana-1.6b-1024.
        width: Image width in pixels (default = model resolution).
        height: Image height in pixels (default = model resolution).
        steps: Diffusion steps (more = higher quality, slower).
        guidance_scale: Classifier-free guidance strength.
        seed: Random seed for reproducibility.
        output_dir: Where to save the image (default: ./generated/).

    Returns:
        Dict with 'path', 'model', 'prompt', 'width', 'height'.
    """
    pipe = get_pipeline(model_name=model)
    images = pipe.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        seed=seed,
        num_images=1,
    )
    path = save_image(images[0], output_dir=output_dir)
    return {
        "status": "success",
        "path": path,
        "model": pipe.model_name,
        "prompt": prompt,
        "width": images[0].width,
        "height": images[0].height,
    }


@tool
def sana_batch(
    prompts: List[str],
    model: Optional[str] = None,
    steps: int = 20,
    guidance_scale: float = 4.5,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Generate multiple images in a batch.

    Args:
        prompts: List of text prompts.
        model: Sana model alias.
        steps: Diffusion steps.
        guidance_scale: CFG strength.
        seed: Optional seed (incremented per prompt).
        output_dir: Where to save images.

    Returns:
        Dict with list of generated image paths.
    """
    pipe = get_pipeline(model_name=model)
    paths = []
    for i, p in enumerate(prompts):
        s = (seed + i) if seed is not None else None
        imgs = pipe.generate(
            prompt=p,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=s,
            num_images=1,
        )
        paths.append(save_image(imgs[0], output_dir=output_dir))
    return {
        "status": "success",
        "count": len(paths),
        "paths": paths,
        "model": pipe.model_name,
    }


@tool
def sana_load_model(model: Optional[str] = None) -> dict:
    """Pre-load a Sana model into memory (warm-up).

    Args:
        model: Sana model alias. Use 'list' to see options.

    Returns:
        Dict with model info.
    """
    if model == "list":
        return {
            "status": "success",
            "available": [
                {"name": k, "hf_repo": v.hf_repo, "resolution": v.resolution,
                 "params": v.params, "description": v.description}
                for k, v in SANA_MODELS.items()
            ],
            "default": default_model(),
        }
    pipe = get_pipeline(model_name=model)
    pipe.load()  # warm up
    return {
        "status": "success",
        "model": pipe.model_name,
        "device": pipe.device,
        "resolution": pipe.info.resolution,
        "params": pipe.info.params,
    }
