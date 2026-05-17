# API Reference

All 33 `@tool` functions exposed by `strands-sana`. Auto-callable, fully typed.

## Inference — Images (8)

### `sana_generate`
Primary text-to-image. Auto-dispatches to PAG when `pag_scale > 0`.
→ See [Image Generation](guide/image-generation.md)

### `sana_batch(prompts, ...)`
One image per prompt; same defaults as `sana_generate`.

### `sana_load_model(model="list", kind=None, tag=None)`
List or pre-load a checkpoint. Supports `kind=` and `tag=` filters.
→ See [Models](guide/models.md)

### `sana_sprint_generate`
1-2 step distilled inference.
→ See [Sana-Sprint](guide/sprint.md)

### `sana_inpaint(prompt, image_path, mask_path, ...)`
Text-guided inpainting (auto-detects `SanaInpaintPipeline` in diffusers).

### `sana_controlnet_generate`
HED/Canny/depth-guided generation.
→ See [ControlNet](guide/controlnet.md)

### `sana_img2img`
Sana-Sprint image-to-image restyle.
→ See [Img2Img](guide/img2img.md)

### `sana_inference_scale(prompt, n_samples, score_fn="clip")`
Generate K candidates, pick best by CLIP.
→ See [Inference Scaling](guide/inference-scaling.md)

---

## Inference — Videos (2)

### `sana_video_generate`
Text-to-video. 4 models, default 480p × 121 frames @ 24fps.
→ See [Video Generation](guide/video-generation.md)

### `sana_image_to_video(image_path, prompt, ...)`
Animate a still image into a video clip.

---

## Adapters (2)

### `sana_load_lora(repo_or_path, scale=1.0, adapter_name=None, model=None)`
Load LoRA weights.
→ See [LoRA](guide/lora.md)

### `sana_unload_loras(model=None)`
Remove all LoRA adapters.

---

## Memory & Cache (2)

### `sana_set_memory_mode(mode="balanced", model=None)`
Modes: `fast` · `balanced` · `low` · `ultra-low`.
→ See [Memory & Quantization](guide/memory.md)

### `sana_clear_cache()`
Drop loaded pipelines, free GPU memory.

---

## Schedulers (2)

### `sana_set_scheduler(name, model=None, use_flow_sigmas=True)`
Swap the active scheduler. 10 aliases.
→ See [Schedulers](guide/schedulers.md)

### `sana_list_schedulers()`
List available aliases.

---

## Optimization (2)

### `sana_quantize(bits=8, backend="auto", model=None)`
Quantize the active pipeline. Tries Quanto / bnb / Nunchaku in order.

### `sana_swap_vae(vae_repo, model=None)`
Replace the VAE (e.g. with DC-AE-Lite for faster decode).

---

## Quality / Metrics (2)

### `sana_metric_clip(image_path, prompt)`
CLIPScore between an image and a prompt.

### `sana_metric_imagereward(image_path, prompt)`
ImageReward score (requires `pip install image-reward`).

---

## DX / Output (3)

### `sana_enhance_prompt(prompt, style="photorealistic")`
Local style hint. Styles: `photorealistic`, `anime`, `oil-painting`, `cinematic`, `minimalist`.

### `sana_export_comfyui_workflow(prompt, output_path, ...)`
Emit ComfyUI graph JSON.
→ See [ComfyUI Export](guide/comfyui.md)

### `sana_safety_check(prompt)`
Keyword-filter NSFW/CSAM gate. ShieldGemma upgrade path documented.

---

## Distribution (3)

### `sana_upload_to_hf(path, repo_id, repo_type="model")`
Push local artifacts (LoRA weights, generated images) to HuggingFace.

### `sana_serve(model="sana-1.6b-1024", port=30000, ...)`
Spawn an SGLang inference server.

### `sana_prefetch_model(model="sana-1.6b-1024", quiet=False)`
Pre-download a checkpoint with progress reporting.

---

## Training (7)

### `sana_train_lora(instance_data_dir, ...)`
DreamBooth + LoRA fine-tuning. Wraps upstream `train_dreambooth_lora_sana.py`.

### `sana_train(config_path, ...)`
Full pretrain / finetune (FSDP / DDP).

### `sana_train_scm_ladd(config_path, ...)`
Sana-Sprint sCM-LADD distillation.

### `sana_train_solrl(config_spec, ...)`
Sol-RL post-training (NVFP4 rollout + BF16 optimization).

### `sana_train_video(config_path, chunk=False, ...)`
SANA-Video FSDP training.

### `sana_train_longsana(config_path, ...)`
LongSANA real-time minute-long video training.

### `sana_list_training_configs(sana_root=None)`
Enumerate all upstream YAML/PY configs (~38 of them).

→ Full training guide: [Training](guide/training.md)

---

## Internal helpers

### `strands_sana.pipeline.SanaPipelineWrapper`
The class behind every inference tool. Supports kind dispatch, LoRA, memory modes, scheduler swap.

### `strands_sana.pipeline.make_step_callback(every_n=5, on_step=None)`
Build a `callback_on_step_end` for streaming preview decodes.

### `strands_sana.models.list_models(kind=None, tag=None)`
Programmatic registry filter.

### `strands_sana.utils.io.save_image / load_image`
PIL helpers; `load_image` accepts file path / URL / bytes.

### `strands_sana.utils.prompts.COMPLEX_HUMAN_INSTRUCTION`
The official Sana Gemma-2 prompt-expansion template.
