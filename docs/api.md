# API Reference

`strands-sana` v0.1.0 wraps the entire NVlabs/Sana family (Sana 1.0/1.5,
Sprint, PAG, ControlNet, 2K/4K) as 13 Strands `@tool` functions.

---

## Core text-to-image

### `sana_generate(...)`
Universal text-to-image. Auto-switches to PAG pipeline when `pag_scale > 0`.

| arg | type | default | notes |
|-----|------|---------|-------|
| `prompt` | str | required | text description |
| `negative_prompt` | str | "" | what to avoid |
| `model` | str? | None | alias from registry |
| `width` / `height` | int? | None | model resolution |
| `steps` | int? | per-model | diffusion steps |
| `guidance_scale` | float? | per-model | CFG strength |
| `seed` | int? | None | RNG |
| `pag_scale` | float | 0.0 | >0 enables PAG |
| `use_complex_instruction` | bool | False | prepend Sana's prompt expander |
| `use_resolution_binning` | bool | True | snap to trained bucket |
| `num_images` | int | 1 | per prompt |
| `output_dir` | str? | None | default: `./generated/` |

### `sana_batch(prompts, ...)`
One image per prompt; same kwargs as `sana_generate`.

### `sana_sprint_generate(...)`
1-2 step distilled inference (~0.1s/image on H100).
Sprint does **not** support `negative_prompt` or CFG.

---

## Variant pipelines

### `sana_inpaint(prompt, image_path, mask_path, ...)`
Text-guided inpainting. Requires `SanaInpaintPipeline` in your diffusers
build (auto-detects, falls back to descriptive error).

### `sana_controlnet_generate(prompt, control_image, ...)`
Conditioned on a preprocessed control map (HED / Canny / depth).

---

## Adapters

### `sana_load_lora(repo_or_path, scale=1.0, adapter_name=None, model=None)`
Attach a LoRA to the active pipeline.

### `sana_unload_loras(model=None)`
Remove all adapters.

---

## Memory & cache

### `sana_set_memory_mode(mode="balanced", model=None)`
Modes: `fast` · `balanced` · `low` · `ultra-low`.

### `sana_clear_cache()`
Drop all loaded pipelines, free GPU memory.

---

## Utilities

### `sana_load_model(model="list", kind=None, tag=None)`
List or pre-load. Filter by `kind` ∈ {t2i, pag, sprint, controlnet} or `tag`.

### `sana_enhance_prompt(prompt, style="photorealistic")`
Add style suffix client-side. Styles: `photorealistic`, `anime`,
`oil-painting`, `cinematic`, `minimalist`.

### `sana_export_comfyui_workflow(prompt, output_path, ...)`
Emit a ComfyUI graph JSON compatible with `lawrence-cj/ComfyUI_ExtraModels`.

### `sana_safety_check(prompt)`
Keyword-filter NSFW/CSAM gate. Documented upgrade path to ShieldGemma-2B.

---

## Models

| Alias | HF Repo | Resolution | Params | Kind |
|-------|---------|-----------|--------|------|
| sana-0.6b-512        | Efficient-Large-Model/Sana_600M_512px_diffusers       | 512  | 590M  | t2i |
| sana-0.6b-1024       | Efficient-Large-Model/Sana_600M_1024px_diffusers      | 1024 | 590M  | t2i |
| sana-1.6b-1024       | Efficient-Large-Model/Sana_1600M_1024px_diffusers     | 1024 | 1.6B  | t2i |
| sana-1.6b-multiling  | Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers | 1024 | 1.6B | t2i |
| sana-1.5-1.6b-1024   | Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers   | 1024 | 1.6B  | t2i |
| sana-1.5-4.8b-1024   | Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers   | 1024 | 4.8B  | t2i |
| sana-1.6b-2k         | Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers  | 2048 | 1.6B  | t2i |
| sana-1.6b-4k         | Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers  | 4096 | 1.6B  | t2i |
| sana-sprint-0.6b-1024 | Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers | 1024 | 590M | sprint |
| sana-sprint-1.6b-1024 | Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers | 1024 | 1.6B | sprint |

`pag` and `controlnet` kinds are activated via `kind_override` or by passing
`pag_scale > 0`. See [DIFF.md](../DIFF.md) for upstream coverage status.
