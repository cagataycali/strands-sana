"""Tests for the model registry."""
import pytest
from strands_sana.models.registry import (
    SANA_MODELS, get_model_info, default_model, list_models,
)


def test_default_model_exists():
    assert default_model() in SANA_MODELS


def test_get_model_info_default():
    info = get_model_info()
    assert info.name == default_model()


def test_get_model_info_specific():
    info = get_model_info("sana-0.6b-512")
    assert info.resolution == 512
    assert "590M" in info.params


def test_get_model_info_invalid():
    with pytest.raises(ValueError):
        get_model_info("totally-fake-model")


def test_all_models_have_required_fields():
    for name, info in SANA_MODELS.items():
        assert info.hf_repo
        assert info.resolution > 0
        assert info.params
        assert info.description
        assert info.pipeline_kind in ("t2i", "pag", "sprint", "controlnet", "video")


def test_sana_15_models_present():
    assert "sana-1.5-1.6b-1024" in SANA_MODELS
    assert "sana-1.5-4.8b-1024" in SANA_MODELS


def test_sprint_models_present():
    sprints = list_models(kind="sprint")
    assert len(sprints) >= 2
    for m in sprints:
        assert m.default_steps <= 4
        assert m.default_guidance == 0.0


def test_2k_4k_models_present():
    assert "sana-1.6b-2k" in SANA_MODELS
    assert "sana-1.6b-4k" in SANA_MODELS
    assert SANA_MODELS["sana-1.6b-2k"].resolution == 2048
    assert SANA_MODELS["sana-1.6b-4k"].resolution == 4096
    assert SANA_MODELS["sana-1.6b-4k"].supports_tiling is True


def test_list_models_filters():
    fast = list_models(tag="fast")
    assert len(fast) >= 2
    sana15 = list_models(tag="sana-1.5")
    assert len(sana15) == 2
    t2i = list_models(kind="t2i")
    assert len(t2i) >= 8
