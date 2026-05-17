"""Sana model registry — checkpoints available on HuggingFace."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SanaModelInfo:
    name: str
    hf_repo: str
    resolution: int
    params: str
    description: str


SANA_MODELS = {
    "sana-0.6b-512": SanaModelInfo(
        name="sana-0.6b-512",
        hf_repo="Efficient-Large-Model/Sana_600M_512px_diffusers",
        resolution=512,
        params="590M",
        description="Fast 512x512 generation",
    ),
    "sana-0.6b-1024": SanaModelInfo(
        name="sana-0.6b-1024",
        hf_repo="Efficient-Large-Model/Sana_600M_1024px_diffusers",
        resolution=1024,
        params="590M",
        description="Lightweight 1024x1024",
    ),
    "sana-1.6b-1024": SanaModelInfo(
        name="sana-1.6b-1024",
        hf_repo="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        resolution=1024,
        params="1.6B",
        description="Best quality 1024x1024",
    ),
    "sana-1.6b-multiling": SanaModelInfo(
        name="sana-1.6b-multiling",
        hf_repo="Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers",
        resolution=1024,
        params="1.6B",
        description="Multilingual prompts (incl. Chinese)",
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
            f"Unknown Sana model '{name}'. Available: {list(SANA_MODELS.keys())}"
        )
    return SANA_MODELS[name]
