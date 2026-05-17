"""Tests for SanaPipelineWrapper (no checkpoint download)."""
import inspect
import pytest

from strands_sana.pipeline.sana_pipeline import (
    SanaPipelineWrapper,
    get_pipeline,
    clear_pipeline_cache,
    _get_pipeline_class,
    _PIPELINE_CACHE,
)


def test_resolve_device():
    # auto-resolve should pick something
    d = SanaPipelineWrapper._resolve_device("auto")
    assert d in ("cuda", "mps", "cpu")
    assert SanaPipelineWrapper._resolve_device("cpu") == "cpu"


def test_get_pipeline_class_t2i():
    cls = _get_pipeline_class("t2i")
    assert cls.__name__ == "SanaPipeline"


def test_get_pipeline_class_sprint():
    cls = _get_pipeline_class("sprint")
    assert cls.__name__ == "SanaSprintPipeline"


def test_get_pipeline_class_pag():
    cls = _get_pipeline_class("pag")
    assert cls.__name__ == "SanaPAGPipeline"


def test_get_pipeline_class_invalid():
    with pytest.raises(ValueError):
        _get_pipeline_class("nonexistent")


def test_get_pipeline_caches():
    clear_pipeline_cache()
    p1 = get_pipeline("sana-0.6b-512")
    p2 = get_pipeline("sana-0.6b-512")
    assert p1 is p2


def test_get_pipeline_kind_override():
    clear_pipeline_cache()
    p = get_pipeline("sana-1.6b-1024", kind_override="pag")
    assert p.kind == "pag"


def test_wrapper_generate_kwargs_match_pipeline():
    """Our wrapper kwargs are recognized by all 4 pipeline classes."""
    from diffusers import (
        SanaPipeline, SanaSprintPipeline, SanaPAGPipeline, SanaControlNetPipeline,
    )
    # Walk through our wrapper.generate body manually — we know which kwargs
    # are passed for each kind.
    common_kwargs = {
        "prompt", "num_inference_steps", "guidance_scale", "height", "width",
        "num_images_per_prompt", "generator", "use_resolution_binning",
        "max_sequence_length",
    }
    for cls in [SanaPipeline, SanaSprintPipeline, SanaPAGPipeline, SanaControlNetPipeline]:
        params = set(inspect.signature(cls.__call__).parameters.keys())
        assert common_kwargs.issubset(params), (
            f"{cls.__name__} missing: {common_kwargs - params}"
        )


def test_clear_pipeline_cache():
    p = get_pipeline("sana-0.6b-512")
    n = clear_pipeline_cache()
    assert n >= 1
    assert len(_PIPELINE_CACHE) == 0
