"""List available Sana models, filtered by kind/tag."""
from strands_sana.tools.generate import sana_load_model


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    print("All models:")
    print(_call(sana_load_model, model="list"))
    print("\nSprint only:")
    print(_call(sana_load_model, model="list", kind="sprint"))
    print("\n2K/4K only:")
    print(_call(sana_load_model, model="list", tag="4k"))
