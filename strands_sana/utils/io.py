"""Image I/O helpers."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Optional, Union


def ensure_output_dir(path: Optional[str] = None) -> Path:
    """Make sure the output directory exists; return Path."""
    out = Path(path or os.getenv("STRANDS_SANA_OUTPUT_DIR", "./generated"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_image(image, output_dir: Optional[str] = None, prefix: str = "sana",
               format: str = "PNG") -> str:
    """Save a PIL.Image to disk with a timestamped filename. Returns path."""
    out = ensure_output_dir(output_dir)
    ext = format.lower()
    fname = f"{prefix}_{int(time.time() * 1000)}.{ext}"
    fpath = out / fname
    image.save(fpath, format=format)
    return str(fpath)


def load_image(source: Union[str, Path, bytes]) -> "Image.Image":  # noqa: F821
    """Load an image from a file path, URL, or raw bytes."""
    from PIL import Image
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    src = str(source)
    if src.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(src) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(src).convert("RGB")
