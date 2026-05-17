"""Wraps the diffusers SanaPipeline with caching and lazy loading."""
from __future__ import annotations

import logging
from typing import Optional, List

from ..models.registry import get_model_info, default_model

logger = logging.getLogger(__name__)

# Module-level cache so multiple tool calls reuse the same loaded model
_PIPELINE_CACHE: dict = {}


class SanaPipelineWrapper:
    """Thin wrapper over diffusers.SanaPipeline."""

    def __init__(self, model_name: Optional[str] = None, device: str = "auto"):
        self.model_name = model_name or default_model()
        self.info = get_model_info(self.model_name)
        self.device = self._resolve_device(device)
        self._pipeline = None

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

    def load(self):
        """Lazy load the diffusers pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from diffusers import SanaPipeline
        except ImportError as e:
            raise ImportError(
                "diffusers and torch required. Install with: "
                "pip install 'strands-sana[hf]' or pip install diffusers torch"
            ) from e

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        logger.info(f"Loading {self.info.hf_repo} on {self.device} ({dtype})")

        self._pipeline = SanaPipeline.from_pretrained(
            self.info.hf_repo,
            torch_dtype=dtype,
        )
        self._pipeline.to(self.device)

        # Memory savings on CUDA
        if self.device == "cuda":
            try:
                self._pipeline.enable_model_cpu_offload()
            except Exception:
                pass

        return self._pipeline

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
        num_images: int = 1,
    ) -> List:
        """Generate images. Returns list of PIL.Image."""
        pipe = self.load()
        h = height or self.info.resolution
        w = width or self.info.resolution

        generator = None
        if seed is not None:
            import torch
            generator = torch.Generator(device=self.device).manual_seed(seed)

        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=h,
            width=w,
            num_images_per_prompt=num_images,
            generator=generator,
        )
        return out.images


def get_pipeline(model_name: Optional[str] = None, device: str = "auto") -> SanaPipelineWrapper:
    """Get (or create + cache) a pipeline for the given model."""
    key = f"{model_name or default_model()}::{device}"
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = SanaPipelineWrapper(model_name=model_name, device=device)
    return _PIPELINE_CACHE[key]
