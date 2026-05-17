"""Wraps the diffusers Sana* pipelines with caching, lazy loading, and
support for all upstream variants: T2I, PAG, Sprint, ControlNet.
"""
from __future__ import annotations

import logging
from typing import Optional, List, Any

from ..models.registry import (
    SanaModelInfo,
    get_model_info,
    default_model,
    PipelineKind,
)

logger = logging.getLogger(__name__)

# Module-level cache so multiple tool calls reuse the same loaded model.
# Key includes lora signature so different LoRA stacks don't collide.
_PIPELINE_CACHE: dict[str, "SanaPipelineWrapper"] = {}


# Map kind → diffusers class import path. Lazy-imported so that an
# install missing one of these (older diffusers) doesn't break the
# whole module.
_PIPELINE_CLASS_BY_KIND: dict[str, str] = {
    "t2i":        "SanaPipeline",
    "pag":        "SanaPAGPipeline",
    "sprint":     "SanaSprintPipeline",
    "controlnet": "SanaControlNetPipeline",
}


def _get_pipeline_class(kind: PipelineKind):
    """Return the diffusers pipeline class for `kind`, importing lazily."""
    try:
        import diffusers
    except ImportError as e:
        raise ImportError(
            "diffusers is required. Install with: pip install 'diffusers>=0.30'"
        ) from e

    cls_name = _PIPELINE_CLASS_BY_KIND.get(kind)
    if cls_name is None:
        raise ValueError(f"Unknown pipeline kind: {kind}")

    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        raise ImportError(
            f"diffusers {diffusers.__version__} does not expose {cls_name}. "
            f"Upgrade with: pip install -U diffusers"
        )
    return cls


class SanaPipelineWrapper:
    """Thin wrapper over diffusers' Sana* pipelines."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        kind_override: Optional[PipelineKind] = None,
    ):
        self.model_name = model_name or default_model()
        self.info: SanaModelInfo = get_model_info(self.model_name)
        self.device = self._resolve_device(device)
        # PAG / sprint / controlnet selected via either model registry or
        # explicit override (e.g. user wants PAG variant of a regular ckpt)
        self.kind: PipelineKind = kind_override or self.info.pipeline_kind
        self._pipeline = None
        self._loaded_loras: list[tuple[str, float]] = []  # (path/repo, scale)

    # ── device & dtype ───────────────────────────────────────────────
    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _resolve_dtype(self):
        import torch
        # bf16 on cuda + mps; fp32 on cpu (mps + bf16 partially works in 2.x)
        if self.info.recommended_dtype == "bf16" and self.device in ("cuda", "mps"):
            return torch.bfloat16
        return torch.float32

    # ── load ─────────────────────────────────────────────────────────
    def load(self):
        """Lazy load the diffusers pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "torch required. pip install torch torchvision"
            ) from e

        cls = _get_pipeline_class(self.kind)
        dtype = self._resolve_dtype()
        logger.info(
            f"Loading {self.info.hf_repo} as {cls.__name__} on {self.device} ({dtype})"
        )

        self._pipeline = cls.from_pretrained(self.info.hf_repo, torch_dtype=dtype)
        self._pipeline.to(self.device)

        # Memory savings on CUDA
        if self.device == "cuda":
            try:
                self._pipeline.enable_model_cpu_offload()
            except Exception:
                logger.debug("enable_model_cpu_offload failed; continuing")

        # 4K models need VAE tiling to fit in VRAM
        if self.info.supports_tiling:
            try:
                self._pipeline.vae.enable_tiling()
                logger.info("Enabled VAE tiling for high-res model")
            except Exception:
                pass

        return self._pipeline

    # ── memory knobs (P1 #7) ─────────────────────────────────────────
    def enable_sequential_cpu_offload(self):
        if self._pipeline is None:
            self.load()
        try:
            self._pipeline.enable_sequential_cpu_offload()
        except Exception as e:
            logger.warning(f"sequential_cpu_offload failed: {e}")

    def enable_vae_tiling(self):
        if self._pipeline is None:
            self.load()
        try:
            self._pipeline.vae.enable_tiling()
        except Exception as e:
            logger.warning(f"vae tiling failed: {e}")

    def enable_attention_slicing(self):
        if self._pipeline is None:
            self.load()
        try:
            self._pipeline.enable_attention_slicing()
        except Exception as e:
            logger.warning(f"attention_slicing failed: {e}")

    # ── lora (P2 #11) ────────────────────────────────────────────────
    def load_lora(self, repo_or_path: str, scale: float = 1.0,
                  adapter_name: Optional[str] = None):
        if self._pipeline is None:
            self.load()
        adapter_name = adapter_name or f"lora_{len(self._loaded_loras)}"
        self._pipeline.load_lora_weights(repo_or_path, adapter_name=adapter_name)
        try:
            self._pipeline.set_adapters([adapter_name], adapter_weights=[scale])
        except Exception:
            pass
        self._loaded_loras.append((repo_or_path, scale))

    def unload_loras(self):
        if self._pipeline is None:
            return
        try:
            self._pipeline.unload_lora_weights()
        except Exception:
            pass
        self._loaded_loras = []

    # ── generate ─────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
        num_images: int = 1,
        # PAG-only
        pag_scale: float = 0.0,
        # ControlNet-only
        control_image: Any = None,
        controlnet_conditioning_scale: float = 1.0,
        # advanced
        use_resolution_binning: bool = True,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[list] = None,
    ) -> List:
        """Generate images. Returns list of PIL.Image.

        Universal entry point that dispatches kwargs based on `self.kind`.
        Off-bucket sizes are auto-binned by diffusers when
        `use_resolution_binning=True`.
        """
        pipe = self.load()

        # Defaults from registry
        steps = num_inference_steps or self.info.default_steps
        gs = guidance_scale if guidance_scale is not None else self.info.default_guidance
        h = height or self.info.resolution
        w = width or self.info.resolution

        # Generator for reproducibility
        generator = None
        if seed is not None:
            import torch
            # MPS doesn't support torch.Generator(device='mps') properly in all torch versions
            gen_device = "cpu" if self.device == "mps" else self.device
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Build kwargs per pipeline kind
        kwargs: dict[str, Any] = dict(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=gs,
            height=h,
            width=w,
            num_images_per_prompt=num_images,
            generator=generator,
            use_resolution_binning=use_resolution_binning,
            max_sequence_length=max_sequence_length,
        )
        if complex_human_instruction is not None:
            kwargs["complex_human_instruction"] = complex_human_instruction

        # Sprint pipeline does NOT accept negative_prompt
        if self.kind != "sprint":
            kwargs["negative_prompt"] = negative_prompt or None

        if self.kind == "pag":
            kwargs["pag_scale"] = pag_scale or 0.0
        elif self.kind == "controlnet":
            if control_image is None:
                raise ValueError("control_image is required for ControlNet pipeline")
            kwargs["control_image"] = control_image
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

        out = pipe(**kwargs)
        return out.images


def get_pipeline(
    model_name: Optional[str] = None,
    device: str = "auto",
    kind_override: Optional[PipelineKind] = None,
) -> SanaPipelineWrapper:
    """Get (or create + cache) a pipeline for the given model + kind."""
    name = model_name or default_model()
    info = get_model_info(name)
    kind = kind_override or info.pipeline_kind
    key = f"{name}::{kind}::{device}"
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = SanaPipelineWrapper(
            model_name=name, device=device, kind_override=kind_override
        )
    return _PIPELINE_CACHE[key]


def clear_pipeline_cache() -> int:
    """Drop all cached pipelines (frees GPU memory). Returns count freed."""
    n = len(_PIPELINE_CACHE)
    _PIPELINE_CACHE.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return n
