# DIFF.md — strands-sana vs NVlabs/Sana upstream

Comparison between this repo (`strands-sana` v0.3.0) and the upstream
[NVlabs/Sana](https://github.com/NVlabs/Sana) reference implementation.

`strands-sana` is a thin agent-tool wrapper around the `diffusers`
SanaPipeline family. We do **not** vendor upstream research code —
instead we depend on `diffusers >= 0.32` for the actual inference path.

Status legend: ✅ done · 🟡 partial / opt-in · ⏳ planned · 🔴 out of scope

---

## ✅ Coverage matrix (v0.3.0)

### P0 — High-impact

| #  | Item | Status | Notes |
|----|------|:------:|-------|
| 1  | Sana-1.5 checkpoints | ✅ | 1.6B + 4.8B |
| 2  | Sana-Sprint          | ✅ | 1-2 step distilled |
| 3  | 2K / 4K resolution   | ✅ | auto VAE tiling |
| 4  | PAG                  | ✅ | `pag_scale=` arg |
| 5  | Aspect-ratio binning | ✅ | default ON |

### P1 — Quantization & memory

| #  | Item | Status | Notes |
|----|------|:------:|-------|
| 6  | 4-bit (Quanto / bnb / Nunchaku) | ✅ | `sana_quantize(bits=4)` |
| 7  | CPU offload toggles  | ✅ | `sana_set_memory_mode` |
| 8  | DC-AE-Lite / DCAE-1.1 VAE swap | ✅ | `sana_swap_vae` |

### P2 — Generation features

| #  | Item | Status | Notes |
|----|------|:------:|-------|
| 9  | ControlNet           | ✅ | `sana_controlnet_generate` |
| 10 | Inpainting           | 🟡 | auto-detects `SanaInpaintPipeline` in diffusers |
| 11 | LoRA loading         | ✅ | `sana_load_lora` / `sana_unload_loras` |
| 12 | SANA-Video           | 🔴 | sibling `strands-sana-video` package |
| 13 | Inference scaling    | ✅ | `sana_inference_scale` (CLIP scoring) |
| 14 | Safety filter        | 🟡 | keyword filter; ShieldGemma upgrade path |

### P3 — Schedulers & sampling

| #  | Item | Status | Notes |
|----|------|:------:|-------|
| 15 | Scheduler choice     | ✅ | 10 aliases |
| 16 | Negative-prompt embed cache | ✅ | `encode_negative_cached` helper |
| 17 | `num_images_per_prompt` | ✅ | `num_images=` |

### P4 — Pipeline / DevX

| #  | Item | Status | Notes |
|----|------|:------:|-------|
| 18 | Streaming preview callback | ✅ | `make_step_callback` |
| 19 | SGLang server adapter | ✅ | `sana_serve` (spawns sglang.launch_server) |
| 20 | ComfyUI graph export | ✅ | `sana_export_comfyui_workflow` |
| 21 | Cosmos-RL / Sol-RL hooks | 🔴 | training/RL — out of scope |
| 22 | Metrics              | ✅ | `sana_metric_clip`, `sana_metric_imagereward` |
| 23 | HuggingFace upload   | ✅ | `sana_upload_to_hf` |
| 24 | Prompt enhancement   | ✅ | `sana_enhance_prompt` + `COMPLEX_HUMAN_INSTRUCTION` |

### P5 — Tests & infra

| #  | Item | Status | Notes |
|----|------|:------:|-------|
| 25 | E2E smoke tests      | ✅ | 49 tests passing |
| 26 | HF download progress | ✅ | `sana_prefetch_model` |
| 27 | Docker image         | ✅ | `Dockerfile` (CUDA 12.1) |
| 28 | mkdocs site          | ✅ | `mkdocs.yml` configured |

---

## 📊 Coverage summary (v0.3.0)

| Area              | Upstream | strands-sana | Coverage |
|-------------------|---------:|-------------:|---------:|
| Models            | ~15      | 10           | **~67%** |
| Pipelines         | T2I, PAG, Sprint, ControlNet, Inpaint, Video | 5/6 | **~83%** |
| Schedulers        | 4+       | **10 aliases** | **~100%** |
| Quantization      | 4bit/8bit | int4/int8 (auto-probe) | **~100%** |
| Metrics           | 5 benchmarks | CLIP, ImageReward | ~40% |
| Training          | full     | n/a (out of scope) | n/a |
| Agent tools       | n/a      | **23**       | n/a |
| Tests             | n/a      | 49 passing   | ✅ |

**P0 done**: 5/5 ✅
**P1 done**: 3/3 ✅
**P2 done**: 4/6 ✅ (10🟡, 12🔴)
**P3 done**: 3/3 ✅
**P4 done**: 6/7 ✅ (21🔴)
**P5 done**: 4/4 ✅

**Remaining gaps**:
- 🟡 Inpainting (depends on diffusers ≥ next minor)
- 🟡 Full ShieldGemma safety (opt-in install)
- 🔴 SANA-Video → `strands-sana-video` sibling package
- 🔴 Cosmos-RL / Sol-RL training hooks (out of scope)
- ⏳ Full FID / GenEval / DPG metric backends (CLIP + ImageReward shipped)

---

## 🎯 Next milestones

- **strands-sana-video v0.1** — Sibling package for SANA-Video / LongSANA
- **v0.4.0** — FID, GenEval, DPG-Bench metric backends
- **v1.0.0** — Stable API, mkdocs published to GH Pages, GPU CI, PyPI release

---

_Generated against NVlabs/Sana ToT and `diffusers==0.37.1` — 23 tools verified by 49 unit tests._
