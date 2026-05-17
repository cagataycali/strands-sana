"""Export a strands-sana run to a ComfyUI workflow JSON."""
from strands_sana import sana_export_comfyui_workflow


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    print(_call(
        sana_export_comfyui_workflow,
        prompt="a serene koi pond, ink wash painting",
        output_path="./generated/comfyui_workflow.json",
        seed=1234,
    ))
