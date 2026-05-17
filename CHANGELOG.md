# Changelog

## [0.4.1] - 2026-05-17

### 🐛 Bug fixes — verified on real GPU (NVIDIA Thor)

11 bugs found by stress-testing all 33 tools on Thor dev kit:

#### Critical fixes
- **#3** `negative_prompt or None` → AttributeError when CFG > 1
- **#4** Video model HF repos were 404 (made-up names) — now use canonical
  upstream names from model_zoo.md
- **#5** Sana-Sprint with `steps != 2` failed with intermediate_timesteps error
- **#6** SANA-Video required (h, w) divisible by 32; binning produced 624
- **#9** Scheduler swap to non-flow schedulers (DDIM/Heun) failed with
  prediction_type=flow_prediction error

#### Medium fixes
- **#10** ControlNet with non-ControlNet checkpoint now returns helpful error
- **#11** LoRA load now wraps RepositoryNotFoundError, PEFT-missing into
  error dicts

#### Infra / DX
- **#2** Training tests auto-clone Sana shallow if missing (cross-machine)
- **#7** MP4 export multi-tier fallback + new `[video]` extra
- New `[lora]` extra (peft) for LoRA support

### Tests
- 72/72 passing on Thor (aarch64, sm_110, CUDA 13)
- All 33 tools verified end-to-end against real GPU
- Generated images at 512/1024/1024-PAG via Sana-1.0/Sana-1.5/Sprint
- Generated MP4 videos at 33/65/97 frames via SANA-Video 480p
- Generated MP4 via Image-to-Video pipeline

## [0.4.0] - 2026-05-17

### 🎉 Full upstream parity — video + training in scope

User clarified that **training is in scope** and **SANA-Video lives in
strands-sana directly** (not a sibling pkg). v0.4.0 closes that:

#### New: SANA-Video (P2 #12 → ✅)
- `sana_video_generate` — text-to-video (4 models: 2B-256/480/720, LongSANA)
- `sana_image_to_video` — image-to-video (SANA-Video I2V)
- Auto MP4 export via `diffusers.utils.export_to_video` (WebP fallback)
- Default frames + fps per-model (49→720 frames, 24-27 fps)

#### New: Sana-Sprint Img2Img (bonus)
- `sana_img2img` — re-render an image with a new prompt at Sprint speed

#### New: Training (P4 #21 → ✅, plus 6 brand-new tools)
Wraps upstream `Sana/train_scripts/` & `train_video_scripts/`. All
default to `dry_run=True` (prints command); pass `dry_run=False` to
launch. Auto-resolves `Sana/` repo root via cwd → `SANA_ROOT` env →
package-relative.

- `sana_train_lora` — DreamBooth + LoRA fine-tuning
- `sana_train` — full pretrain / finetune (FSDP/DDP)
- `sana_train_scm_ladd` — Sana-Sprint sCM-LADD distillation
- `sana_train_solrl` — Sol-RL post-training (NVFP4 rollout + BF16)
- `sana_train_video` — SANA-Video FSDP training (with `chunk=` for long-context)
- `sana_train_longsana` — LongSANA real-time minute-long video training
- `sana_list_training_configs` — enumerate all upstream YAML/PY configs

#### Models (10 → 16)
New entries in registry:
- `sana-sprint-i2i-1.6b-1024` (sprint-i2i kind)
- `sana-video-2b-256` / `-480` / `-720` (video kind)
- `sana-video-i2v-480` (image-to-video kind)
- `longsana-2b-480` (real-time minute-long, 27fps × 720 frames)

Each video model has correct `default_frames` + `fps` metadata so callers
get sensible defaults without specifying.

#### Pipeline wrapper
- New kinds: `sprint-i2i`, `video`, `image-to-video`
- New `generate_video()` method on `SanaPipelineWrapper`
- `_PIPELINE_CLASS_BY_KIND` now covers 7 diffusers pipeline classes

### Tools: 23 → 33

### Tests: 49 → 72
- 6 new training tests (all dry-run, no GPU required)
- 6 new video tests (signature/registry/error paths)
- 4 new img2img tests
- All passing.

## [0.3.0] - 2026-05-17

Closes DIFF #16, #19, #26.
- New tools: `sana_serve`, `sana_prefetch_model`
- Pipeline helpers: `encode_negative_cached`
- 49 tests passing.

## [0.2.0] - 2026-05-17

Closes DIFF #6, #8, #13, #15, #18, #22, #23, #27, #28.
- 8 new tools: scheduler swap, quantize, vae swap, hf upload,
  inference-scale, clip-score, image-reward
- `make_step_callback` for streaming preview decodes
- Dockerfile (CUDA 12.1, py3.12)
- mkdocs.yml (Material theme)

## [0.1.0] - 2026-05-17

Closes DIFF #1, #2, #3, #4, #5, #7, #9, #10, #11, #14, #17, #20, #24.
- 10 model checkpoints (was 4)
- 5 pipeline kinds (T2I, PAG, Sprint, ControlNet, Inpaint)
- 13 tools (was 3)
- LoRA, memory modes, VAE tiling, ComfyUI export
- 36 tests

## [0.0.1] - 2026-05-17

Initial scaffold: 3 tools, 4 checkpoints, basic SanaPipeline wrapper.
