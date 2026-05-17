"""Test utility functions."""
import tempfile
from pathlib import Path

from strands_sana.utils.io import ensure_output_dir
from strands_sana.utils.prompts import enhance_prompt, COMPLEX_HUMAN_INSTRUCTION


def test_ensure_output_dir_creates():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "nested" / "dir"
        result = ensure_output_dir(str(target))
        assert result.exists()
        assert result.is_dir()


def test_ensure_output_dir_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        ensure_output_dir(tmp)
        result = ensure_output_dir(tmp)
        assert result.exists()


def test_enhance_prompt_styles():
    for style in ("photorealistic", "anime", "oil-painting", "cinematic", "minimalist"):
        out = enhance_prompt("a duck", style=style)
        assert "duck" in out
        assert len(out) > len("a duck")


def test_complex_human_instruction_format():
    assert isinstance(COMPLEX_HUMAN_INSTRUCTION, list)
    assert all(isinstance(s, str) for s in COMPLEX_HUMAN_INSTRUCTION)
    assert "User Prompt:" in COMPLEX_HUMAN_INSTRUCTION[-1]
