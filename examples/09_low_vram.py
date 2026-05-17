"""Fit Sana into 8 GB VRAM via memory-saver mode."""
from strands_sana import sana_set_memory_mode, sana_generate


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    _call(sana_set_memory_mode, mode="low")
    print(_call(sana_generate, prompt="a peaceful tundra under aurora", steps=14))
