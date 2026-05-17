"""Sana model registry — checkpoints available on HuggingFace.

Covers all upstream variants: Sana 1.0/1.5, Sprint, ControlNet, plus
SANA-Video & LongSANA. Resolution-binning aspect ratios are sourced
from upstream `diffusion/data/datasets/utils.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal


PipelineKind = Literal[
    "t2i", "pag", "sprint", "sprint-i2i", "controlnet",
    "video", "image-to-video",
]


@dataclass
class SanaModelInfo:
    name: str
    hf_repo: str
    resolution: int
    params: str
    description: str
    pipeline_kind: PipelineKind = "t2i"
    recommended_dtype: str = "bf16"
    default_steps: int = 20
    default_guidance: float = 4.5
    supports_tiling: bool = False
    # Video-specific
    default_frames: int = 0  # 0 = not a video model
    fps: int = 24
    tags: list[str] = field(default_factory=list)


SANA_MODELS: dict[str, SanaModelInfo] = {
    # ── Sana 1.0 baseline ────────────────────────────────────────────
    "sana-0.6b-512": SanaModelInfo(
        name="sana-0.6b-512",
        hf_repo="Efficient-Large-Model/Sana_600M_512px_diffusers",
        resolution=512, params="590M",
        description="Sana 0.6B @ 512px — fast",
        tags=["sana-1.0", "fast"],
    ),
    "sana-0.6b-1024": SanaModelInfo(
        name="sana-0.6b-1024",
        hf_repo="Efficient-Large-Model/Sana_600M_1024px_diffusers",
        resolution=1024, params="590M",
        description="Sana 0.6B @ 1024px",
        tags=["sana-1.0"],
    ),
    "sana-1.6b-1024": SanaModelInfo(
        name="sana-1.6b-1024",
        hf_repo="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        resolution=1024, params="1.6B",
        description="Sana 1.6B @ 1024px — best 1.0 quality",
        tags=["sana-1.0", "default"],
    ),
    "sana-1.6b-multiling": SanaModelInfo(
        name="sana-1.6b-multiling",
        hf_repo="Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers",
        resolution=1024, params="1.6B",
        description="Sana 1.6B multilingual",
        tags=["sana-1.0", "multiling"],
    ),

    # ── Sana 1.5 ─────────────────────────────────────────────────────
    "sana-1.5-1.6b-1024": SanaModelInfo(
        name="sana-1.5-1.6b-1024",
        hf_repo="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        resolution=1024, params="1.6B",
        description="Sana-1.5 1.6B",
        tags=["sana-1.5"],
    ),
    "sana-1.5-4.8b-1024": SanaModelInfo(
        name="sana-1.5-4.8b-1024",
        hf_repo="Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        resolution=1024, params="4.8B",
        description="Sana-1.5 4.8B — top quality",
        tags=["sana-1.5", "best-quality"],
    ),

    # ── 2K / 4K ──────────────────────────────────────────────────────
    "sana-1.6b-2k": SanaModelInfo(
        name="sana-1.6b-2k",
        hf_repo="Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers",
        resolution=2048, params="1.6B",
        description="Sana 1.6B @ 2048px",
        supports_tiling=True,
        tags=["sana-1.0", "2k"],
    ),
    "sana-1.6b-4k": SanaModelInfo(
        name="sana-1.6b-4k",
        hf_repo="Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers",
        resolution=4096, params="1.6B",
        description="Sana 1.6B @ 4096px",
        supports_tiling=True,
        tags=["sana-1.0", "4k"],
    ),

    # ── Sana-Sprint (T2I) ────────────────────────────────────────────
    "sana-sprint-0.6b-1024": SanaModelInfo(
        name="sana-sprint-0.6b-1024",
        hf_repo="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        resolution=1024, params="590M",
        description="Sana-Sprint 0.6B (1-2 step)",
        pipeline_kind="sprint",
        default_steps=2, default_guidance=0.0,
        tags=["sprint", "fast"],
    ),
    "sana-sprint-1.6b-1024": SanaModelInfo(
        name="sana-sprint-1.6b-1024",
        hf_repo="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        resolution=1024, params="1.6B",
        description="Sana-Sprint 1.6B (~0.1s on H100)",
        pipeline_kind="sprint",
        default_steps=2, default_guidance=0.0,
        tags=["sprint", "fast", "best-sprint"],
    ),

    # ── Sana-Sprint Img2Img (NEW v0.4.0) ─────────────────────────────
    "sana-sprint-i2i-1.6b-1024": SanaModelInfo(
        name="sana-sprint-i2i-1.6b-1024",
        hf_repo="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        resolution=1024, params="1.6B",
        description="Sana-Sprint 1.6B for image-to-image",
        pipeline_kind="sprint-i2i",
        default_steps=4, default_guidance=0.0,
        tags=["sprint", "i2i"],
    ),

    # ── SANA-Video (NEW v0.4.0) ──────────────────────────────────────
    "sana-video-2b-256": SanaModelInfo(
        name="sana-video-2b-256",
        hf_repo="Efficient-Large-Model/SANA_Video_2B_256px",
        resolution=256, params="2B",
        description="SANA-Video 2B @ 256px — fastest",
        pipeline_kind="video",
        default_steps=50, default_guidance=6.0,
        default_frames=49, fps=24,
        tags=["video", "fast"],
    ),
    "sana-video-2b-480": SanaModelInfo(
        name="sana-video-2b-480",
        hf_repo="Efficient-Large-Model/SANA_Video_2B_480px",
        resolution=480, params="2B",
        description="SANA-Video 2B @ 480px (5s clips)",
        pipeline_kind="video",
        default_steps=50, default_guidance=6.0,
        default_frames=121, fps=24,
        tags=["video", "default-video"],
    ),
    "sana-video-2b-720": SanaModelInfo(
        name="sana-video-2b-720",
        hf_repo="Efficient-Large-Model/SANA_Video_2B_720px",
        resolution=720, params="2B",
        description="SANA-Video 2B @ 720px (LTX-VAE refiner)",
        pipeline_kind="video",
        default_steps=50, default_guidance=6.0,
        default_frames=121, fps=24,
        supports_tiling=True,
        tags=["video", "hi-res-video"],
    ),
    "sana-video-i2v-480": SanaModelInfo(
        name="sana-video-i2v-480",
        hf_repo="Efficient-Large-Model/SANA_Video_I2V_2B_480px",
        resolution=480, params="2B",
        description="SANA-Video Image-to-Video @ 480px",
        pipeline_kind="image-to-video",
        default_steps=50, default_guidance=6.0,
        default_frames=121, fps=24,
        tags=["video", "i2v"],
    ),

    # ── LongSANA (real-time minute-long video) ───────────────────────
    "longsana-2b-480": SanaModelInfo(
        name="longsana-2b-480",
        hf_repo="Efficient-Large-Model/LongSANA_2B_480px",
        resolution=480, params="2B",
        description="LongSANA 2B @ 480px — minute-long, 27 FPS real-time",
        pipeline_kind="video",
        default_steps=4, default_guidance=1.0,
        default_frames=720, fps=27,  # 27fps × ~27s
        tags=["video", "long-video", "real-time"],
    ),
}


def default_model() -> str:
    return "sana-1.6b-1024"


def get_model_info(name: Optional[str] = None) -> SanaModelInfo:
    name = name or default_model()
    if name not in SANA_MODELS:
        raise ValueError(
            f"Unknown Sana model '{name}'. "
            f"Available: {sorted(SANA_MODELS.keys())}"
        )
    return SANA_MODELS[name]


def list_models(
    kind: Optional[PipelineKind] = None,
    tag: Optional[str] = None,
) -> list[SanaModelInfo]:
    out = list(SANA_MODELS.values())
    if kind:
        out = [m for m in out if m.pipeline_kind == kind]
    if tag:
        out = [m for m in out if tag in m.tags]
    return out
