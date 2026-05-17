"""Load a LoRA adapter, generate, then unload."""
from strands_sana import sana_load_lora, sana_generate, sana_unload_loras


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    _call(sana_load_lora, repo_or_path="some-user/some-sana-lora", scale=0.8)
    print(_call(sana_generate, prompt="a duck in the style of the loaded LoRA"))
    _call(sana_unload_loras)
