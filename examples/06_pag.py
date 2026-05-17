"""PAG (Perturbed Attention Guidance) for higher-quality generation."""
from strands_sana import sana_generate


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    # pag_scale > 0 → auto-switches to SanaPAGPipeline
    result = _call(
        sana_generate,
        prompt="A photorealistic portrait of a wolf in moonlight",
        pag_scale=2.0,
        steps=20,
        seed=42,
    )
    print(result)
