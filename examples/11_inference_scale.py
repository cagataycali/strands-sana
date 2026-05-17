"""Inference-time scaling: generate K candidates, return best by CLIPScore."""
from strands_sana import sana_inference_scale


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    print(_call(
        sana_inference_scale,
        prompt="a serene mountain lake reflecting the milky way",
        n_samples=4,
        seed_start=100,
        score_fn="clip",
    ))
