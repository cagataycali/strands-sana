"""Wraps the diffusers Sana* pipelines with caching, lazy loading, and
support for all upstream variants: T2I, PAG, Sprint, Sprint-I2I,
ControlNet, Video, Image-to-Video.
"""
from __future__ import annotations

import logging
from typing import Optional, List, Any

from ..models.registry import (
    SanaModelInfo, get_model_info, default_model, PipelineKind,
)

logger = logging.getLogger(__name__)

_PIPELINE_CACHE: dict[str, "SanaPipelineWrapper"] = {}

_PIPELINE_CLASS_BY_KIND: dict[str, str] = {
    "t2i":            "SanaPipeline",
    "pag":            "SanaPAGPipeline",
    "sprint":         "SanaSprintPipeline",
    "sprint-i2i":     "SanaSprintImg2ImgPipeline",
    "controlnet":     "SanaControlNetPipeline",
    "video":          "SanaVideoPipeline",
    "image-to-video": "SanaImageToVideoPipeline",
}


def _get_pipeline_class(kind: PipelineKind):
    try:
        import diffusers
    except ImportError as e:
        raise ImportError("diffusers required: pip install 'diffusers>=0.32'") from e
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
        self.kind: PipelineKind = kind_override or self.info.pipeline_kind
        self._pipeline = None
        self._loaded_loras: list[tuple[str, float]] = []

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
        if self.info.recommended_dtype == "bf16" and self.device in ("cuda", "mps"):
            return torch.bfloat16
        return torch.float32

    def load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("torch required: pip install torch torchvision") from e

        cls = _get_pipeline_class(self.kind)
        dtype = self._resolve_dtype()
        logger.info(f"Loading {self.info.hf_repo} as {cls.__name__} on {self.device} ({dtype})")

        self._pipeline = cls.from_pretrained(self.info.hf_repo, torch_dtype=dtype)
        self._pipeline.to(self.device)

        if self.device == "cuda":
            try:
                self._pipeline.enable_model_cpu_offload()
            except Exception:
                pass

        if self.info.supports_tiling:
            try:
                self._pipeline.vae.enable_tiling()
                logger.info("VAE tiling enabled (high-res / video)")
            except Exception:
                pass

        return self._pipeline

    # ── memory knobs ─────────────────────────────────────────────────
    def enable_sequential_cpu_offload(self):
        if self._pipeline is None: self.load()
        try: self._pipeline.enable_sequential_cpu_offload()
        except Exception as e: logger.warning(f"sequential offload: {e}")

    def enable_vae_tiling(self):
        if self._pipeline is None: self.load()
        try: self._pipeline.vae.enable_tiling()
        except Exception as e: logger.warning(f"vae tiling: {e}")

    def enable_attention_slicing(self):
        if self._pipeline is None: self.load()
        try: self._pipeline.enable_attention_slicing()
        except Exception as e: logger.warning(f"attention_slicing: {e}")

    # ── lora ─────────────────────────────────────────────────────────
    def load_lora(self, repo_or_path: str, scale: float = 1.0,
                  adapter_name: Optional[str] = None):
        if self._pipeline is None: self.load()
        adapter_name = adapter_name or f"lora_{len(self._loaded_loras)}"
        self._pipeline.load_lora_weights(repo_or_path, adapter_name=adapter_name)
        try:
            self._pipeline.set_adapters([adapter_name], adapter_weights=[scale])
        except Exception:
            pass
        self._loaded_loras.append((repo_or_path, scale))

    def unload_loras(self):
        if self._pipeline is None: return
        try: self._pipeline.unload_lora_weights()
        except Exception: pass
        self._loaded_loras = []

    # ── generate (image) ─────────────────────────────────────────────
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
        pag_scale: float = 0.0,
        control_image: Any = None,
        controlnet_conditioning_scale: float = 1.0,
        # img2img
        image: Any = None,
        strength: float = 0.6,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[list] = None,
    ) -> List:
        """Generate images. Returns list of PIL.Image."""
        pipe = self.load()
        steps = num_inference_steps or self.info.default_steps
        gs = guidance_scale if guidance_scale is not None else self.info.default_guidance
        h = height or self.info.resolution
        w = width or self.info.resolution

        generator = None
        if seed is not None:
            import torch
            gen_device = "cpu" if self.device == "mps" else self.device
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        kwargs: dict[str, Any] = dict(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=gs,
            height=h, width=w,
            num_images_per_prompt=num_images,
            generator=generator,
            use_resolution_binning=use_resolution_binning,
            max_sequence_length=max_sequence_length,
        )
        if complex_human_instruction is not None:
            kwargs["complex_human_instruction"] = complex_human_instruction

        # Sprint pipelines don't accept negative_prompt
        if self.kind not in ("sprint", "sprint-i2i"):
            kwargs["negative_prompt"] = negative_prompt or ""

        if self.kind == "pag":
            kwargs["pag_scale"] = pag_scale or 0.0
        elif self.kind == "controlnet":
            if control_image is None:
                raise ValueError("control_image required for ControlNet")
            kwargs["control_image"] = control_image
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
        elif self.kind == "sprint-i2i":
            if image is None:
                raise ValueError("image required for Sprint Img2Img")
            kwargs["image"] = image
            kwargs["strength"] = strength

        out = pipe(**kwargs)
        return out.images

    # ── generate_video ────────────────────────────────────────────────
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        frames: Optional[int] = None,
        seed: Optional[int] = None,
        num_videos: int = 1,
        # I2V
        image: Any = None,
        max_sequence_length: int = 300,
        use_resolution_binning: bool = True,
        complex_human_instruction: Optional[list] = None,
    ) -> List:
        """Generate video frames. Returns list of [list of PIL.Image] per video.

        For T2V (`kind='video'`): use prompt only.
        For I2V (`kind='image-to-video'`): also pass `image=`.
        """
        if self.kind not in ("video", "image-to-video"):
            raise ValueError(
                f"generate_video requires kind=video or image-to-video, got {self.kind}"
            )
        pipe = self.load()
        steps = num_inference_steps or self.info.default_steps
        gs = guidance_scale if guidance_scale is not None else self.info.default_guidance
        h = height or self.info.resolution
        w = width or self.info.resolution
        f = frames or self.info.default_frames

        generator = None
        if seed is not None:
            import torch
            gen_device = "cpu" if self.device == "mps" else self.device
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        kwargs: dict[str, Any] = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            num_inference_steps=steps,
            guidance_scale=gs,
            height=h, width=w,
            frames=f,
            num_videos_per_prompt=num_videos,
            generator=generator,
            use_resolution_binning=use_resolution_binning,
            max_sequence_length=max_sequence_length,
        )
        if complex_human_instruction is not None:
            kwargs["complex_human_instruction"] = complex_human_instruction
        if self.kind == "image-to-video":
            if image is None:
                raise ValueError("image required for Image-to-Video")
            kwargs["image"] = image

        out = pipe(**kwargs)
        # SanaVideoPipeline returns .frames (list of list of PIL.Image)
        return out.frames if hasattr(out, "frames") else out.images


def get_pipeline(
    model_name: Optional[str] = None,
    device: str = "auto",
    kind_override: Optional[PipelineKind] = None,
) -> SanaPipelineWrapper:
    name = model_name or default_model()
    info = get_model_info(name)
    kind = kind_override or info.pipeline_kind
    key = f"{name}::{kind}::{device}"
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = SanaPipelineWrapper(
            model_name=name, device=device, kind_override=kind_override,
        )
    return _PIPELINE_CACHE[key]


def clear_pipeline_cache() -> int:
    n = len(_PIPELINE_CACHE)
    _PIPELINE_CACHE.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return n


def make_step_callback(every_n: int = 5, on_step=None):
    """Build a `callback_on_step_end` that fires every N steps."""
    def _cb(pipe, step, timestep, callback_kwargs):
        if step % every_n == 0:
            latents = callback_kwargs.get("latents")
            if on_step is not None:
                try: on_step(step, latents)
                except Exception as e: logger.debug(f"on_step: {e}")
            else:
                logger.info(f"step {step}/{int(timestep)}")
        return callback_kwargs
    return _cb
