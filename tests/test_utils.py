"""Test utility functions."""
import tempfile
from pathlib import Path

from strands_sana.utils.io import ensure_output_dir


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
