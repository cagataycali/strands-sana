"""Tests for SANA-Video tools (no model download)."""
import inspect

import pytest

from strands_sana.tools.video import sana_video_generate, sana_image_to_video


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


def test_video_models_registered():
    from strands_sana.models import list_models
    videos = list_models(kind="video")
    assert len(videos) >= 4
    names = {m.name for m in videos}
    assert "sana-video-2b-480" in names
    assert "longsana-2b-480" in names


def test_video_pipeline_classes_resolvable():
    from strands_sana.pipeline.sana_pipeline import _get_pipeline_class
    cls = _get_pipeline_class("video")
    assert cls.__name__ == "SanaVideoPipeline"
    cls = _get_pipeline_class("image-to-video")
    assert cls.__name__ == "SanaImageToVideoPipeline"


def test_video_pipeline_signature_compatibility():
    from diffusers import SanaVideoPipeline, SanaImageToVideoPipeline
    common = {"prompt", "frames", "num_inference_steps", "guidance_scale",
              "height", "width", "num_videos_per_prompt"}
    for cls in (SanaVideoPipeline, SanaImageToVideoPipeline):
        params = set(inspect.signature(cls.__call__).parameters.keys())
        assert common.issubset(params), f"{cls.__name__} missing: {common - params}"
    # I2V also accepts image
    i2v = set(inspect.signature(SanaImageToVideoPipeline.__call__).parameters.keys())
    assert "image" in i2v


def test_video_default_frames_set():
    from strands_sana.models import get_model_info
    for name in ("sana-video-2b-480", "sana-video-2b-720", "longsana-2b-480"):
        m = get_model_info(name)
        assert m.default_frames > 0
        assert m.fps > 0


def test_image_to_video_wrong_model():
    """Passing a non-i2v model should error gracefully."""
    r = _call(sana_image_to_video,
              image_path="/tmp/fake.png",
              prompt="test",
              model="sana-1.6b-1024")
    assert r["status"] == "error"


def test_video_generate_wrong_model():
    r = _call(sana_video_generate, prompt="x", model="sana-1.6b-1024")
    assert r["status"] == "error"
