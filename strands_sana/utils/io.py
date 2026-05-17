"""Image I/O helpers."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional


def ensure_output_dir(path: Optional[str] = None) -> Path:
    """Make sure the output directory exists; return Path."""
    out = Path(path or os.getenv("STRANDS_SANA_OUTPUT_DIR", "./generated"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_image(image, output_dir: Optional[str] = None, prefix: str = "sana") -> str:
    """Save a PIL.Image to disk with a timestamped filename. Returns path."""
    out = ensure_output_dir(output_dir)
    fname = f"{prefix}_{int(time.time() * 1000)}.png"
    fpath = out / fname
    image.save(fpath)
    return str(fpath)
