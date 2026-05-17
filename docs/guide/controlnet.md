# ControlNet

```python
from strands_sana import sana_controlnet_generate

sana_controlnet_generate(
    prompt="a futuristic robot",
    control_image="hed_edges.png",
    model="<a Sana-ControlNet checkpoint>",
    controlnet_conditioning_scale=0.8,
    steps=20,
)
```

!!! warning "Diffusers ControlNet weights are 'Coming soon' upstream"

    As of NVlabs/Sana model_zoo.md, the diffusers-format ControlNet
    checkpoints are listed as "Coming soon". Native checkpoints exist
    (e.g. `Efficient-Large-Model/Sana_1600M_1024px_BF16_ControlNet_HED`)
    but require upstream's `app/sana_controlnet_pipeline.py` loader.

    The tool detects this and returns a helpful error pointing to the workaround.

## When upstream ships diffusers ControlNet

Just point at the new repo:

```python
sana_controlnet_generate(
    model="Efficient-Large-Model/Sana_1600M_1024px_BF16_ControlNet_HED_diffusers",
    prompt="...", control_image="...",
)
```

## Control image preprocessing

ControlNet expects pre-processed control maps (HED edges, Canny, depth, normal). Use upstream's HED preprocessor or any ControlNet-compatible preprocessor:

```python
import cv2
img = cv2.Canny(cv2.imread("source.png"), 100, 200)
cv2.imwrite("canny.png", img)
```

## `controlnet_conditioning_scale`

| value | effect |
|:---:|---|
| 0.0 | ControlNet disabled |
| 0.5 | gentle guidance |
| 1.0 | **default** — strong guidance |
| 1.5+ | over-saturated, may break |
