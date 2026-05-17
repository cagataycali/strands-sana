"""Test the list action of sana_load_model (no model download)."""
from strands_sana.tools.generate import sana_load_model


def _call(tool, **kw):
    if hasattr(tool, "original_func"):
        return tool.original_func(**kw)
    return tool(**kw)


def test_list_action():
    r = _call(sana_load_model, model="list")
    assert r["status"] == "success"
    assert "default" in r
    assert isinstance(r["available"], list)
    assert r["count"] >= 10


def test_list_filter_by_kind():
    r = _call(sana_load_model, model="list", kind="sprint")
    assert r["count"] >= 2
    assert all(m["pipeline_kind"] == "sprint" for m in r["available"])


def test_list_filter_by_tag():
    r = _call(sana_load_model, model="list", tag="sana-1.5")
    assert r["count"] == 2
