"""Test that the public API is importable + version is set."""

def test_top_level_imports():
    from strands_sana import (
        sana_generate, sana_batch, sana_load_model,
        sana_sprint_generate, sana_inpaint, sana_controlnet_generate,
        sana_load_lora, sana_unload_loras,
        sana_set_memory_mode, sana_clear_cache,
        sana_enhance_prompt, sana_export_comfyui_workflow, sana_safety_check,
    )
    for fn in [
        sana_generate, sana_batch, sana_load_model,
        sana_sprint_generate, sana_inpaint, sana_controlnet_generate,
        sana_load_lora, sana_unload_loras,
        sana_set_memory_mode, sana_clear_cache,
        sana_enhance_prompt, sana_export_comfyui_workflow, sana_safety_check,
    ]:
        assert callable(fn) or hasattr(fn, "original_func"), f"{fn} not callable"


def test_version():
    import strands_sana
    assert strands_sana.__version__ == "0.1.0"


def test_pipeline_import():
    from strands_sana.pipeline import (
        SanaPipelineWrapper, get_pipeline, clear_pipeline_cache,
    )
    assert SanaPipelineWrapper
    assert callable(get_pipeline)
    assert callable(clear_pipeline_cache)


def test_utils_import():
    from strands_sana.utils import (
        save_image, ensure_output_dir, load_image,
        enhance_prompt, COMPLEX_HUMAN_INSTRUCTION,
    )
    assert callable(save_image)
    assert callable(ensure_output_dir)
    assert callable(load_image)
    assert callable(enhance_prompt)
    assert isinstance(COMPLEX_HUMAN_INSTRUCTION, list)
    assert len(COMPLEX_HUMAN_INSTRUCTION) > 0


def test_models_import():
    from strands_sana.models import (
        SANA_MODELS, SanaModelInfo, get_model_info, default_model, list_models,
    )
    assert isinstance(SANA_MODELS, dict)
    assert len(SANA_MODELS) >= 10
