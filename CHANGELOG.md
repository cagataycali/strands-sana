# Changelog

## [0.2.0] - 2026-05-17

### Added — second feature drop
Closes DIFF #6, #8, #13, #15, #18, #22, #23, #27, #28.

#### New tools (21 total, was 13)
- `sana_set_scheduler` / `sana_list_schedulers` — 10 scheduler aliases
- `sana_quantize` — int4 / int8 via Quanto / bitsandbytes / Nunchaku probe
- `sana_swap_vae` — hot-swap VAE (DC-AE-Lite, DCAE-1.1)
- `sana_upload_to_hf` — push artifacts to HuggingFace
- `sana_inference_scale` — generate-K-pick-best with CLIPScore
- `sana_metric_clip`, `sana_metric_imagereward` — eval helpers

#### Pipeline
- `make_step_callback(every_n, on_step)` for streaming preview decodes (P4 #18)

#### Infrastructure
- `Dockerfile` (CUDA 12.1, Python 3.12, ready to publish as `strands-sana:cuda12.1`)
- `mkdocs.yml` (Material theme, ready for GH Pages)
- 8 → 12 examples (added scheduler swap, inference scaling)

### Tests
- 43 → 45 tests, all passing
- Coverage: streaming callback, scheduler map sanity, quantize/upload/metric error paths

## [0.1.0] - 2026-05-17

### Added — major feature drop
Closes DIFF #1, #2, #3, #4, #5, #7, #9, #10, #11, #14, #17, #20, #24.

#### Models (10 total, was 4)
- Sana-1.5 1.6B & 4.8B
- Sana-Sprint 0.6B & 1.6B (1-2 step distilled)
- Sana 1.6B 2K & 4K (with auto VAE tiling)
- Tags + filtering (`list_models(kind=, tag=)`)

#### Pipelines
- T2I, PAG, Sprint, ControlNet — kwargs-aware dispatch
- Aspect-ratio binning enabled by default
- LoRA load/unload + adapter naming
- Memory modes: fast / balanced / low / ultra-low
- VAE tiling auto-enabled on 2K/4K
- Pipeline cache keyed by `(model, kind, device)`; clearable

#### New @tool functions (13 total, was 3)
- `sana_sprint_generate`, `sana_inpaint`, `sana_controlnet_generate`
- `sana_load_lora` / `sana_unload_loras`
- `sana_set_memory_mode`, `sana_clear_cache`
- `sana_enhance_prompt`, `sana_export_comfyui_workflow`, `sana_safety_check`

#### Tests & examples
- 36 tests, all passing (no checkpoint downloads)
- 10 examples covering each variant

## [0.0.1] - 2026-05-17

### Added
- Initial scaffold
- `sana_generate`, `sana_batch`, `sana_load_model`
- 4 baseline checkpoints, `diffusers.SanaPipeline` wrapper
- CI workflow (Python 3.10/3.11/3.12)
- Tests for registry, imports, utils
- Examples and docs scaffolding
