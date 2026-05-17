"""Tests for the model registry."""
import pytest
from strands_sana.models.registry import (
    SANA_MODELS, get_model_info, default_model
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
