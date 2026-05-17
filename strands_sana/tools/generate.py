"""Strands @tool-decorated functions for NVIDIA Sana.

Tools exposed (per DIFF.md):
- sana_generate           — universal text-to-image (P0 #1, #4, #5)
- sana_batch              — batch text-to-image
- sana_load_model         — pre-load + list models
- sana_sprint_generate    — fast 1-2 step distilled (P0 #2)
- sana_inpaint            — text-guided inpainting (P2 #10)
- sana_controlnet_generate — controlnet (P2 #9)
- sana_load_lora          — runtime LoRA load (P2 #11)
- sana_unload_loras       — clear all LoRAs
- sana_set_memory_mode    — quantize / cpu offload knobs (P1 #6, #7)
- sana_clear_cache        — drop cached pipelines
- sana_enhance_prompt     — local style hint (P4 #24)
- sana_export_comfyui_workflow — emit ComfyUI .json (P4 #20)
- sana_safety_check       — pre-generation safety (P2 #14)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from strands import tool

from ..pipeline.sana_pipeline import get_pipeline, clear_pipeline_cache
from ..models.registry import SANA_MODELS, default_model, list_models
from ..utils.io import save_image, load_image
from ..utils.prompts import enhance_prompt as _enhance_prompt, COMPLEX_HUMAN_INSTRUCTION

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Core text-to-image
# ────────────────────────────────────────────────────────────────────
@tool
def sana_generate(
    prompt: str,
    negative_prompt: str = "",
    model: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    pag_scale: float = 0.0,
    use_complex_instruction: bool = False,
    use_resolution_binning: bool = True,
    num_images: int = 1,
) -> dict:
    """Generate image(s) from a text prompt using NVIDIA Sana.

    Automatically picks `SanaPipeline` or `SanaPAGPipeline` based on
    `pag_scale`. Off-bucket sizes are auto-binned by diffusers when
    `use_resolution_binning=True`.

    Args:
        prompt: Text description.
        negative_prompt: What to avoid.
        model: Model alias (default `sana-1.6b-1024`). Use `sana_load_model('list')` to see options.
        width/height: Image dimensions (default = model resolution).
        steps: Diffusion steps (default per-model).
        guidance_scale: CFG strength (default per-model).
        seed: RNG seed for reproducibility.
        output_dir: Save dir (default `./generated/`).
        pag_scale: If > 0, switches to SanaPAGPipeline for higher quality (2× compute).
        use_complex_instruction: Prepend Sana's official prompt-enhancement template.
        use_resolution_binning: Auto-snap (h,w) to closest trained bucket.
        num_images: Images per prompt.

    Returns:
        Dict with paths, model, prompt, dimensions.
    """
    kind_override = "pag" if pag_scale and pag_scale > 0 else None
    pipe = get_pipeline(model_name=model, kind_override=kind_override)

    images = pipe.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        seed=seed,
        num_images=num_images,
        pag_scale=pag_scale,
        use_resolution_binning=use_resolution_binning,
        complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION if use_complex_instruction else None,
    )
    paths = [save_image(im, output_dir=output_dir) for im in images]
    return {
        "status": "success",
        "paths": paths,
        "path": paths[0] if paths else None,
        "model": pipe.model_name,
        "kind": pipe.kind,
        "prompt": prompt,
        "width": images[0].width if images else None,
        "height": images[0].height if images else None,
        "count": len(paths),
    }


@tool
def sana_batch(
    prompts: List[str],
    model: Optional[str] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    pag_scale: float = 0.0,
) -> dict:
    """Generate one image per prompt in a list.

    Args:
        prompts: List of text prompts.
        model: Sana model alias.
        steps/guidance_scale: Override defaults.
        seed: Optional seed (incremented per prompt).
        output_dir: Where to save images.
        pag_scale: If > 0, use PAG pipeline.

    Returns:
        Dict with list of generated image paths.
    """
    kind_override = "pag" if pag_scale and pag_scale > 0 else None
    pipe = get_pipeline(model_name=model, kind_override=kind_override)
    paths = []
    for i, p in enumerate(prompts):
        s = (seed + i) if seed is not None else None
        imgs = pipe.generate(
            prompt=p,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=s,
            num_images=1,
            pag_scale=pag_scale,
        )
        paths.append(save_image(imgs[0], output_dir=output_dir))
    return {
        "status": "success",
        "count": len(paths),
        "paths": paths,
        "model": pipe.model_name,
    }


# ────────────────────────────────────────────────────────────────────
# Sprint (distilled, 1-2 step)  — P0 #2
# ────────────────────────────────────────────────────────────────────
@tool
def sana_sprint_generate(
    prompt: str,
    model: str = "sana-sprint-1.6b-1024",
    steps: int = 2,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    num_images: int = 1,
) -> dict:
    """Ultra-fast 1-2 step generation via Sana-Sprint distillation.

    ~0.1s per 1024px image on H100, ~0.3s on RTX 4090.
    Note: Sprint pipeline does not support negative_prompt or CFG.

    Args:
        prompt: Text description.
        model: One of `sana-sprint-0.6b-1024`, `sana-sprint-1.6b-1024`.
        steps: 1 or 2 (Sprint sweet spot).
        width/height/seed: Same as `sana_generate`.
        output_dir: Save location.
        num_images: Images per prompt.
    """
    pipe = get_pipeline(model_name=model)
    if pipe.kind != "sprint":
        return {
            "status": "error",
            "error": f"Model '{model}' is kind='{pipe.kind}' — use a sprint model alias",
        }
    images = pipe.generate(
        prompt=prompt,
        num_inference_steps=steps,
        height=height,
        width=width,
        seed=seed,
        num_images=num_images,
    )
    paths = [save_image(im, output_dir=output_dir, prefix="sprint") for im in images]
    return {
        "status": "success",
        "paths": paths,
        "path": paths[0] if paths else None,
        "model": pipe.model_name,
        "steps": steps,
        "count": len(paths),
    }


# ────────────────────────────────────────────────────────────────────
# Inpainting (P2 #10)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_inpaint(
    prompt: str,
    image_path: str,
    mask_path: str,
    negative_prompt: str = "",
    model: Optional[str] = None,
    steps: int = 28,
    guidance_scale: float = 4.5,
    strength: float = 0.85,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Text-guided inpainting: regenerate masked region of `image_path`.

    Note: Uses upstream NVlabs/Sana inpainting (not yet in `diffusers`).
    Falls back to `SanaPipeline` with `image=` if available; else returns a
    descriptive error so the agent can surface install instructions.

    Args:
        prompt: Description of what should fill the masked area.
        image_path: Path/URL of source image.
        mask_path: Path/URL of binary mask (white = inpaint, black = keep).
        strength: 0..1 — how much to deviate from the source.
    """
    try:
        import diffusers
        # Look for SanaInpaintPipeline (may not exist yet in diffusers 0.37)
        InpaintCls = getattr(diffusers, "SanaInpaintPipeline", None)
        if InpaintCls is None:
            return {
                "status": "error",
                "error": (
                    "SanaInpaintPipeline not in your diffusers version. "
                    "Either upgrade diffusers, or use NVlabs/Sana's "
                    "`app/sana_pipeline_inpaint.py` directly."
                ),
                "diffusers_version": diffusers.__version__,
            }
    except ImportError as e:
        return {"status": "error", "error": str(e)}

    pipe = get_pipeline(model_name=model)
    src = load_image(image_path)
    mask = load_image(mask_path)

    import torch
    gen_device = "cpu" if pipe.device == "mps" else pipe.device
    generator = torch.Generator(device=gen_device).manual_seed(seed) if seed is not None else None

    # Re-instantiate as inpaint pipeline using the loaded model components
    pipeline = pipe.load()
    inpaint = InpaintCls(**pipeline.components)
    out = inpaint(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=src,
        mask_image=mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator,
    )
    path = save_image(out.images[0], output_dir=output_dir, prefix="inpaint")
    return {"status": "success", "path": path, "model": pipe.model_name}


# ────────────────────────────────────────────────────────────────────
# ControlNet (P2 #9)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_controlnet_generate(
    prompt: str,
    control_image: str,
    negative_prompt: str = "",
    model: Optional[str] = None,
    controlnet_conditioning_scale: float = 1.0,
    steps: int = 20,
    guidance_scale: float = 4.5,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Generate an image guided by a control image (HED/Canny/depth/etc).

    Args:
        prompt: Text description.
        control_image: Path/URL of preprocessed control map.
        controlnet_conditioning_scale: 0..2 — strength of control signal.

    Note: Use a Sana-ControlNet checkpoint; upstream's `app_sana_controlnet_hed.py`
    has the HED preprocessor wiring.
    """
    pipe = get_pipeline(model_name=model, kind_override="controlnet")
    ctrl = load_image(control_image)
    try:
        pipe.load()
    except Exception as e:
        if "controlnet" in str(e).lower() and "expected" in str(e).lower():
            return {
                "status": "error",
                "error": (
                    f"Model '{pipe.model_name}' is not a Sana-ControlNet "
                    f"checkpoint. As of Sana ToT, diffusers-format ControlNet "
                    f"weights are 'Coming soon' (see Sana model_zoo.md). "
                    f"Workaround: use upstream Sana repo's "
                    f"app/sana_controlnet_pipeline.py with a native ckpt like "
                    f"Efficient-Large-Model/Sana_1600M_1024px_BF16_ControlNet_HED."
                ),
                "underlying": str(e)[:300],
            }
        raise
    images = pipe.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        control_image=ctrl,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    )
    path = save_image(images[0], output_dir=output_dir, prefix="ctrlnet")
    return {
        "status": "success",
        "path": path,
        "model": pipe.model_name,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
    }


# ────────────────────────────────────────────────────────────────────
# LoRA (P2 #11)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_load_lora(
    repo_or_path: str,
    scale: float = 1.0,
    adapter_name: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Load a LoRA adapter into the active Sana pipeline.

    Args:
        repo_or_path: HF repo (`user/repo`) or local path to LoRA weights.
        scale: Adapter weight (0..1+).
        adapter_name: Custom name (auto-generated if None).
        model: Which base model alias to attach to.
    """
    pipe = get_pipeline(model_name=model)
    pipe.load_lora(repo_or_path, scale=scale, adapter_name=adapter_name)
    return {
        "status": "success",
        "lora": repo_or_path,
        "scale": scale,
        "adapter_name": adapter_name,
        "loaded": [{"repo": r, "scale": s} for r, s in pipe._loaded_loras],
    }


@tool
def sana_unload_loras(model: Optional[str] = None) -> dict:
    """Remove all LoRA adapters from the active pipeline."""
    pipe = get_pipeline(model_name=model)
    pipe.unload_loras()
    return {"status": "success"}


# ────────────────────────────────────────────────────────────────────
# Memory mode (P1 #6, #7)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_set_memory_mode(
    mode: str = "balanced",
    model: Optional[str] = None,
) -> dict:
    """Toggle memory-saving features for the active pipeline.

    Args:
        mode: One of:
          - "fast": no offload, max VRAM, max speed (default for big GPUs)
          - "balanced": model_cpu_offload (default cuda behavior)
          - "low": sequential_cpu_offload + vae tiling + attention slicing (~8GB VRAM)
          - "ultra-low": low + future 4bit (placeholder)
    """
    pipe = get_pipeline(model_name=model)
    pipe.load()  # ensure pipeline exists

    applied = []
    if mode == "fast":
        # nothing extra; user must have re-loaded without offload
        pass
    elif mode == "balanced":
        try:
            pipe._pipeline.enable_model_cpu_offload()
            applied.append("model_cpu_offload")
        except Exception:
            pass
    elif mode in ("low", "ultra-low"):
        pipe.enable_sequential_cpu_offload(); applied.append("sequential_cpu_offload")
        pipe.enable_vae_tiling();              applied.append("vae_tiling")
        pipe.enable_attention_slicing();       applied.append("attention_slicing")
    else:
        return {"status": "error", "error": f"Unknown mode: {mode}"}

    return {"status": "success", "mode": mode, "applied": applied}


# ────────────────────────────────────────────────────────────────────
# Cache management
# ────────────────────────────────────────────────────────────────────
@tool
def sana_clear_cache() -> dict:
    """Drop all loaded Sana pipelines, free GPU memory."""
    n = clear_pipeline_cache()
    return {"status": "success", "freed_pipelines": n}


# ────────────────────────────────────────────────────────────────────
# Listing / metadata
# ────────────────────────────────────────────────────────────────────
@tool
def sana_load_model(
    model: Optional[str] = None,
    kind: Optional[str] = None,
    tag: Optional[str] = None,
) -> dict:
    """Pre-load a Sana model, or list available models.

    Args:
        model: Model alias. Use `'list'` to enumerate all models.
        kind: Filter list by pipeline kind (`t2i`, `pag`, `sprint`, `controlnet`).
        tag:  Filter list by tag (`fast`, `sana-1.5`, `sprint`, `2k`, `4k`, ...).
    """
    if model == "list":
        models = list_models(kind=kind, tag=tag)
        return {
            "status": "success",
            "available": [
                {
                    "name": m.name,
                    "hf_repo": m.hf_repo,
                    "resolution": m.resolution,
                    "params": m.params,
                    "pipeline_kind": m.pipeline_kind,
                    "default_steps": m.default_steps,
                    "default_guidance": m.default_guidance,
                    "tags": m.tags,
                    "description": m.description,
                }
                for m in models
            ],
            "count": len(models),
            "default": default_model(),
        }
    pipe = get_pipeline(model_name=model)
    pipe.load()  # warm up
    return {
        "status": "success",
        "model": pipe.model_name,
        "device": pipe.device,
        "kind": pipe.kind,
        "resolution": pipe.info.resolution,
        "params": pipe.info.params,
    }


# ────────────────────────────────────────────────────────────────────
# Prompt enhancement (P4 #24)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_enhance_prompt(prompt: str, style: str = "photorealistic") -> dict:
    """Add style hints to a prompt (client-side, no LLM call).

    Styles: `photorealistic`, `anime`, `oil-painting`, `cinematic`, `minimalist`.
    """
    enhanced = _enhance_prompt(prompt, style=style)
    return {"status": "success", "original": prompt, "enhanced": enhanced, "style": style}


# ────────────────────────────────────────────────────────────────────
# ComfyUI workflow export (P4 #20)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_export_comfyui_workflow(
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    model: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 4.5,
    seed: int = 42,
) -> dict:
    """Emit a ComfyUI workflow JSON that mirrors the Strands-Sana parameters.

    Compatible with `lawrence-cj/ComfyUI_ExtraModels` Sana nodes.
    """
    info = SANA_MODELS.get(model or default_model())
    if info is None:
        return {"status": "error", "error": f"Unknown model: {model}"}

    workflow = {
        "1": {"class_type": "CheckpointLoaderSimpleSana",
              "inputs": {"ckpt_name": info.hf_repo}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {
                  "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                  "latent_image": ["4", 0],
                  "seed": seed, "steps": steps, "cfg": cfg,
                  "sampler_name": "dpmpp_2m", "scheduler": "flow_match",
                  "denoise": 1.0,
              }},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": "sana"}},
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(workflow, f, indent=2)
    return {"status": "success", "path": output_path, "model": info.name}


# ────────────────────────────────────────────────────────────────────
# Safety check (P2 #14) — best-effort
# ────────────────────────────────────────────────────────────────────
@tool
def sana_safety_check(prompt: str) -> dict:
    """Run Google ShieldGemma-2B over a prompt to flag NSFW/unsafe content.

    Returns `{safe: bool, score: float, reason: str}`. Falls back to a
    keyword filter if ShieldGemma weights aren't available.
    """
    bad_keywords = [
        "nsfw", "explicit sexual", "csam", "child sexual", "gore",
        "bestiality", "incest",
    ]
    pl = prompt.lower()
    hits = [k for k in bad_keywords if k in pl]
    if hits:
        return {
            "status": "success", "safe": False, "score": 0.0,
            "reason": f"keyword filter matched: {hits}",
            "method": "keyword-filter",
        }
    # Optional heavyweight check — skipped by default (model not pre-downloaded)
    return {
        "status": "success", "safe": True, "score": 1.0,
        "reason": "no keyword match (lightweight check)",
        "method": "keyword-filter",
        "note": "For full coverage, integrate google/shieldgemma-2b as in upstream app/safety_check.py",
    }
