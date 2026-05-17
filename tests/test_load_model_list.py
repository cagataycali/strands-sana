"""Test the list action of sana_load_model (no model download)."""
from strands_sana.tools.generate import sana_load_model


def test_list_action():
    # Strands @tool-decorated functions are callable for direct use
    result = sana_load_model.original_func("list") if hasattr(sana_load_model, "original_func") else sana_load_model("list")
    assert result["status"] == "success"
    assert "default" in result
    assert isinstance(result["available"], list)
    assert len(result["available"]) >= 1
