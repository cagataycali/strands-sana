"""ControlNet generation guided by a HED edge map."""
from strands_sana import sana_controlnet_generate


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    # Replace with a real HED-preprocessed image path
    print(_call(
        sana_controlnet_generate,
        prompt="a futuristic robot",
        control_image="./examples/hed_edge.png",
        controlnet_conditioning_scale=0.8,
    ))
