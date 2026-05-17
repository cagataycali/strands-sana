"""Tests for sana_img2img."""

from strands_sana.tools.img2img import sana_img2img


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


def test_sprint_i2i_model_present():
    from strands_sana.models import list_models
    i2i = list_models(kind="sprint-i2i")
    assert len(i2i) >= 1
    assert i2i[0].default_steps == 4
    assert i2i[0].default_guidance == 0.0


def test_pipeline_class_resolvable():
    from strands_sana.pipeline.sana_pipeline import _get_pipeline_class
    cls = _get_pipeline_class("sprint-i2i")
    assert cls.__name__ == "SanaSprintImg2ImgPipeline"


def test_img2img_wrong_kind():
    r = _call(sana_img2img, prompt="x", image_path="/tmp/fake.png",
              model="sana-1.6b-1024")
    assert r["status"] == "error"


def test_img2img_pipeline_signature():
    """Verify our wrapper kwargs match SanaSprintImg2ImgPipeline signature."""
    import inspect
    from diffusers import SanaSprintImg2ImgPipeline
    params = set(inspect.signature(SanaSprintImg2ImgPipeline.__call__).parameters.keys())
    needed = {"prompt", "image", "strength", "num_inference_steps",
              "guidance_scale", "height", "width"}
    assert needed.issubset(params), f"missing: {needed - params}"
