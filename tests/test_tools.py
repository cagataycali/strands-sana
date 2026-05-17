"""Tests for new strands-sana tools (no model download required)."""
import json
import tempfile
from pathlib import Path

from strands_sana.tools.generate import (
    sana_enhance_prompt,
    sana_export_comfyui_workflow,
    sana_safety_check,
    sana_clear_cache,
)


def _call(tool, **kw):
    if hasattr(tool, "original_func"):
        return tool.original_func(**kw)
    return tool(**kw)


def test_enhance_prompt_tool():
    r = _call(sana_enhance_prompt, prompt="a cat", style="anime")
    assert r["status"] == "success"
    assert "anime" in r["enhanced"]
    assert r["original"] == "a cat"


def test_safety_check_safe():
    r = _call(sana_safety_check, prompt="a peaceful sunrise")
    assert r["safe"] is True


def test_safety_check_unsafe():
    r = _call(sana_safety_check, prompt="csam content")
    assert r["safe"] is False


def test_comfyui_workflow_export():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "wf.json"
        r = _call(
            sana_export_comfyui_workflow,
            prompt="a duck", output_path=str(out),
            seed=42, steps=20, cfg=4.5,
        )
        assert r["status"] == "success"
        wf = json.loads(out.read_text())
        assert wf["5"]["inputs"]["seed"] == 42
        assert wf["5"]["inputs"]["steps"] == 20
        assert wf["5"]["inputs"]["cfg"] == 4.5


def test_comfyui_workflow_unknown_model():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "wf.json"
        r = _call(
            sana_export_comfyui_workflow,
            prompt="x", output_path=str(out), model="bogus-model",
        )
        assert r["status"] == "error"


def test_clear_cache():
    r = _call(sana_clear_cache)
    assert r["status"] == "success"
    assert "freed_pipelines" in r
