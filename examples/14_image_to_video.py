"""Animate a still image into a video clip."""
from strands_sana import sana_image_to_video


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    print(_call(
        sana_image_to_video,
        image_path="./input.png",
        prompt="zoom out slowly while leaves swirl in the wind",
        seed=42,
    ))
