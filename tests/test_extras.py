"""Tests for v0.2.0 extras (no model download)."""

from strands_sana.tools.extras import (
    sana_list_schedulers,
    sana_set_scheduler,
    sana_quantize,
    sana_swap_vae,
    sana_upload_to_hf,
    sana_metric_clip,
    SCHEDULER_MAP,
)


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


def test_list_schedulers():
    r = _call(sana_list_schedulers)
    assert r["status"] == "success"
    assert r["count"] >= 10
    aliases = [s["alias"] for s in r["schedulers"]]
    assert "flow-match-euler" in aliases
    assert "dpm-solver" in aliases


def test_scheduler_unknown_name():
    r = _call(sana_set_scheduler, name="nonexistent-scheduler")
    assert r["status"] == "error"


def test_scheduler_map_classes_exist():
    """Every alias maps to a real diffusers class."""
    import diffusers
    for alias, cls_name in SCHEDULER_MAP.items():
        assert hasattr(diffusers, cls_name), f"diffusers missing {cls_name}"


def test_quantize_invalid_bits():
    r = _call(sana_quantize, bits=2)
    assert r["status"] == "error"


def test_metric_clip_bad_path():
    r = _call(sana_metric_clip, image_path="/nonexistent/foo.png", prompt="test")
    assert r["status"] == "error"


def test_upload_no_hf_lib():
    """If huggingface_hub isn't there, returns clean error."""
    # If installed, this should at least not crash on a bogus path
    r = _call(sana_upload_to_hf, path="/tmp/does/not/exist", repo_id="me/test")
    assert r["status"] == "error"  # error either way (auth or path)
