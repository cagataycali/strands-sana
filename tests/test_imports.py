"""Test that public API is importable."""
def test_top_level_imports():
    from strands_sana import sana_generate, sana_batch, sana_load_model
    assert callable(sana_generate)
    assert callable(sana_batch)
    assert callable(sana_load_model)


def test_version():
    import strands_sana
    assert strands_sana.__version__


def test_pipeline_import():
    from strands_sana.pipeline import SanaPipelineWrapper, get_pipeline
    assert SanaPipelineWrapper
    assert callable(get_pipeline)


def test_utils_import():
    from strands_sana.utils import save_image, ensure_output_dir
    assert callable(save_image)
    assert callable(ensure_output_dir)
