# Changelog

## [0.1.0] - 2026-05-17

### Added — major feature drop
Implements DIFF.md P0 → P4 backlog (closes #1, #2, #3, #4, #5, #6, #7, #9,
#10, #11, #14, #17, #20, #24).

#### Models (now 10 vs original 4)
- Sana-1.5 1.6B & 4.8B checkpoints
- Sana-Sprint 0.6B & 1.6B (1-2 step distilled)
- Sana 1.6B 2K & 4K (with auto VAE tiling)
- Tags + filtering via `list_models(kind=, tag=)`

#### Pipelines
- `SanaPipelineWrapper` now dispatches T2I / PAG / Sprint / ControlNet
- Auto-binning aspect ratios (`use_resolution_binning=True` by default)
- LoRA support: load/unload at runtime with adapter naming
- Memory-saver modes: `fast`, `balanced`, `low`, `ultra-low`
- VAE tiling auto-enabled for 2K/4K models
- Pipeline cache keyed by `(model, kind, device)`; clearable

#### New @tool functions
- `sana_sprint_generate` — fast 1-2 step inference
- `sana_inpaint` — text-guided inpainting (uses SanaInpaintPipeline if available)
- `sana_controlnet_generate` — HED/Canny/Depth-guided generation
- `sana_load_lora` / `sana_unload_loras`
- `sana_set_memory_mode` — VRAM knobs
- `sana_clear_cache` — release GPU memory
- `sana_enhance_prompt` — client-side style hints (no LLM call)
- `sana_export_comfyui_workflow` — bridge to lawrence-cj/ComfyUI_ExtraModels
- `sana_safety_check` — keyword-filter safety gate (with note for ShieldGemma upgrade)

#### Existing tools enhanced
- `sana_generate`: now supports `pag_scale`, `use_complex_instruction`,
  `use_resolution_binning`, `num_images`
- `sana_load_model`: added `kind=` and `tag=` filters

#### Utilities
- `load_image()` — file/URL/bytes loader
- `enhance_prompt()` — 5 preset styles
- `COMPLEX_HUMAN_INSTRUCTION` — Sana's official prompt expander template

#### Tests
- 36 tests, all passing
- Coverage: registry, imports, utils, list_action, tools, pipeline wrapper
- No checkpoint downloads in test suite (verifies signatures via `inspect`)

#### Examples
- 01_basic_generate (existing)
- 02_list_models (filter by kind/tag)
- 05_sprint
- 06_pag
- 07_controlnet
- 08_lora
- 09_low_vram
- 10_comfyui_export

### Compatibility
- Requires `diffusers >= 0.32` (verified against 0.37.1)
- Tested on macOS arm64 + MPS (CPU fallback works)

## [0.0.1] - 2026-05-17

### Added
- Initial scaffold
- `sana_generate` tool — single-image text-to-image
- `sana_batch` tool — batch generation
- `sana_load_model` tool — model warm-up & listing
- Model registry with 4 Sana checkpoints
- Pipeline wrapper around `diffusers.SanaPipeline`
- CI workflow (Python 3.10/3.11/3.12)
- Tests for registry, imports, utils
- Examples and docs scaffolding
