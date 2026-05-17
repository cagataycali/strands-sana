# Examples

Pick a recipe based on what you want to build.

## Single image, default settings
```python
from strands_sana import sana_generate
sana_generate(prompt="a serene koi pond")
```

## Fast inference (Sprint)
```python
from strands_sana import sana_sprint_generate
sana_sprint_generate(prompt="...", model="sana-sprint-1.6b-1024", steps=2)
```

## Video clip (5s)
```python
from strands_sana import sana_video_generate
sana_video_generate(prompt="...", frames=121, steps=15)
```

## Animate a still
```python
from strands_sana import sana_image_to_video
sana_image_to_video(image_path="src.png", prompt="zoom out")
```

## Restyle an existing image
```python
from strands_sana import sana_img2img
sana_img2img(prompt="anime style", image_path="src.png", strength=0.5)
```

## Best of K
→ [Best-of-K example](best-of-k.md)

## Multi-style batch
→ [Multi-style example](multi-style.md)

## Full video pipeline
→ [Video pipeline](video-pipeline.md)

## In-repo examples

The [`examples/`](https://github.com/cagataycali/strands-sana/tree/main/examples) directory has 17 runnable scripts:

```
01_basic_generate.py        # sana_generate
02_list_models.py           # sana_load_model("list")
03_batch.py                 # sana_batch
04_direct_pipeline.py       # SanaPipelineWrapper directly
05_sprint.py                # sana_sprint_generate
06_pag.py                   # PAG variant
07_controlnet.py            # ControlNet
08_lora.py                  # sana_load_lora
09_low_vram.py              # sana_set_memory_mode("low")
10_comfyui_export.py        # sana_export_comfyui_workflow
11_inference_scale.py       # sana_inference_scale
12_scheduler.py             # sana_set_scheduler
13_video.py                 # generate hero assets (used to make this site)
14_image_to_video.py        # sana_image_to_video
15_train_lora.py            # sana_train_lora
16_train_video.py           # sana_train_video / longsana
17_solrl.py                 # sana_train_solrl
```
