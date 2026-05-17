"""Sana model registry — checkpoints available on HuggingFace.

Covers all upstream variants: Sana-1.0, Sana-1.5, Sana-Sprint, plus 2K/4K.
Resolution-binning aspect ratios are sourced from upstream
`diffusion/data/datasets/utils.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal


PipelineKind = Literal["t2i", "pag", "sprint", "controlnet", "video"]


@dataclass
class SanaModelInfo:
    name: str
    hf_repo: str
    resolution: int
    params: str
    description: str
    # Which diffusers pipeline class to instantiate
    pipeline_kind: PipelineKind = "t2i"
    # Recommended dtype: "bf16" for most, "fp32" for CPU fallback
    recommended_dtype: str = "bf16"
    # Recommended steps (Sprint = 2, others = 20)
    default_steps: int = 20
    # Default guidance_scale
    default_guidance: float = 4.5
    # Whether the model can support 2k/4k via vae tiling
    supports_tiling: bool = False
    # Tags for filtering
    tags: list[str] = field(default_factory=list)


SANA_MODELS: dict[str, SanaModelInfo] = {
    # ── Sana 1.0 baseline (already shipped) ────────────────────────────
    "sana-0.6b-512": SanaModelInfo(
        name="sana-0.6b-512",
        hf_repo="Efficient-Large-Model/Sana_600M_512px_diffusers",
        resolution=512,
        params="590M",
        description="Sana 0.6B @ 512px — fast",
        default_steps=20,
        tags=["sana-1.0", "fast"],
    ),
    "sana-0.6b-1024": SanaModelInfo(
        name="sana-0.6b-1024",
        hf_repo="Efficient-Large-Model/Sana_600M_1024px_diffusers",
        resolution=1024,
        params="590M",
        description="Sana 0.6B @ 1024px — lightweight",
        tags=["sana-1.0"],
    ),
    "sana-1.6b-1024": SanaModelInfo(
        name="sana-1.6b-1024",
        hf_repo="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        resolution=1024,
        params="1.6B",
        description="Sana 1.6B @ 1024px — best 1.0 quality",
        tags=["sana-1.0", "default"],
    ),
    "sana-1.6b-multiling": SanaModelInfo(
        name="sana-1.6b-multiling",
        hf_repo="Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers",
        resolution=1024,
        params="1.6B",
        description="Sana 1.6B multilingual (Chinese/English/Emoji)",
        tags=["sana-1.0", "multiling"],
    ),

    # ── Sana 1.5 (P0 #1) ───────────────────────────────────────────────
    "sana-1.5-1.6b-1024": SanaModelInfo(
        name="sana-1.5-1.6b-1024",
        hf_repo="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        resolution=1024,
        params="1.6B",
        description="Sana-1.5 1.6B — improved GenEval/CLIP",
        tags=["sana-1.5"],
    ),
    "sana-1.5-4.8b-1024": SanaModelInfo(
        name="sana-1.5-4.8b-1024",
        hf_repo="Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        resolution=1024,
        params="4.8B",
        description="Sana-1.5 4.8B — top-tier quality",
        tags=["sana-1.5", "best-quality"],
    ),

    # ── 2K / 4K (P0 #3) ────────────────────────────────────────────────
    "sana-1.6b-2k": SanaModelInfo(
        name="sana-1.6b-2k",
        hf_repo="Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers",
        resolution=2048,
        params="1.6B",
        description="Sana 1.6B @ 2048px (BF16, ~10GB VRAM)",
        supports_tiling=True,
        tags=["sana-1.0", "2k"],
    ),
    "sana-1.6b-4k": SanaModelInfo(
        name="sana-1.6b-4k",
        hf_repo="Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers",
        resolution=4096,
        params="1.6B",
        description="Sana 1.6B @ 4096px (BF16, ~22GB VRAM, needs vae tiling)",
        supports_tiling=True,
        tags=["sana-1.0", "4k"],
    ),

    # ── Sana-Sprint (P0 #2) — one/few-step distilled ───────────────────
    "sana-sprint-0.6b-1024": SanaModelInfo(
        name="sana-sprint-0.6b-1024",
        hf_repo="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        resolution=1024,
        params="590M",
        description="Sana-Sprint 0.6B — 1-2 step ultra-fast (~0.3s on RTX 4090)",
        pipeline_kind="sprint",
        default_steps=2,
        default_guidance=0.0,  # Sprint is distilled — no CFG
        tags=["sprint", "fast"],
    ),
    "sana-sprint-1.6b-1024": SanaModelInfo(
        name="sana-sprint-1.6b-1024",
        hf_repo="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        resolution=1024,
        params="1.6B",
        description="Sana-Sprint 1.6B — 2 step inference (~0.1s on H100)",
        pipeline_kind="sprint",
        default_steps=2,
        default_guidance=0.0,
        tags=["sprint", "fast", "best-sprint"],
    ),
}


def default_model() -> str:
    """Default model alias."""
    return "sana-1.6b-1024"


def get_model_info(name: Optional[str] = None) -> SanaModelInfo:
    """Look up model info by alias."""
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
    """Filter registry by pipeline kind and/or tag."""
    out = list(SANA_MODELS.values())
    if kind:
        out = [m for m in out if m.pipeline_kind == kind]
    if tag:
        out = [m for m in out if tag in m.tags]
    return out
