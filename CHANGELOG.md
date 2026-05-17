# Changelog

## [0.3.0] - 2026-05-17

### Added — third feature drop
Closes DIFF #16, #19, #26.

#### New tools (23 total)
- `sana_serve` — spawn an SGLang inference server for Sana (P4 #19)
- `sana_prefetch_model` — pre-download checkpoint with progress (P5 #26)

#### Pipeline helpers
- `encode_negative_cached(pipe, negative_prompt)` (P3 #16)
- `clear_negative_embed_cache()`

### Tests
- 45 → 49 passing
- New: serve/prefetch error paths, embed cache lifecycle

### Status snapshot
- P0: 5/5 ✅ · P1: 3/3 ✅ · P2: 4/6 (rest 🟡/🔴)
- P3: 3/3 ✅ · P4: 6/7 (rest 🔴) · P5: 4/4 ✅

## [0.2.0] - 2026-05-17

Closes DIFF #6, #8, #13, #15, #18, #22, #23, #27, #28.
- 8 new tools: scheduler swap, quantize, vae swap, hf upload,
  inference-scale, clip-score, image-reward
- `make_step_callback` for streaming preview decodes
- Dockerfile (CUDA 12.1, py3.12)
- mkdocs.yml (Material theme)
- 43 → 45 tests

## [0.1.0] - 2026-05-17

Closes DIFF #1, #2, #3, #4, #5, #7, #9, #10, #11, #14, #17, #20, #24.
- 10 model checkpoints (was 4)
- 5 pipeline kinds (T2I, PAG, Sprint, ControlNet, Inpaint)
- 13 tools (was 3): sprint, inpaint, controlnet, lora, memory, comfyui,
  safety, prompt-enhance
- LoRA, memory modes, VAE tiling, ComfyUI export
- 36 tests

## [0.0.1] - 2026-05-17

Initial scaffold: 3 tools, 4 checkpoints, basic SanaPipeline wrapper.
