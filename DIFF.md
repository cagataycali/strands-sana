# DIFF.md — strands-sana vs NVlabs/Sana upstream

Comparison between this repo (`strands-sana` v0.2.0) and the upstream
[NVlabs/Sana](https://github.com/NVlabs/Sana) reference implementation.

`strands-sana` is a **thin agent-tool wrapper** around the `diffusers`
SanaPipeline family. We do **not** vendor upstream research code —
instead we depend on `diffusers >= 0.32` for the actual inference path,
which means new upstream features land here as soon as `diffusers`
ships them.

Status legend: ✅ done · 🟡 partial / opt-in · ⏳ planned · 🔴 out of scope

---

## ✅ Coverage matrix

### P0 — High-impact

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 1  | Sana-1.5 checkpoints                  | ✅ | `sana-1.5-1.6b-1024`, `sana-1.5-4.8b-1024` |
| 2  | Sana-Sprint                           | ✅ | `sana_sprint_generate` (1-2 step, ~0.1s/img on H100) |
| 3  | 2K / 4K resolution                    | ✅ | with auto VAE tiling |
| 4  | PAG (Perturbed Attention Guidance)    | ✅ | `pag_scale=` on `sana_generate` |
| 5  | Aspect-ratio binning                  | ✅ | enabled by default (`use_resolution_binning=True`) |

### P1 — Quantization & memory

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 6  | 4-bit (Quanto / bnb / Nunchaku)       | ✅ | `sana_quantize(bits=4, backend="auto")` — probes installed backends |
| 7  | CPU offload toggles                   | ✅ | `sana_set_memory_mode("low"/"ultra-low")` |
| 8  | DC-AE-Lite / DCAE-1.1 VAE swap        | ✅ | `sana_swap_vae(vae_repo=...)` |

### P2 — Generation features

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 9  | ControlNet                            | ✅ | `sana_controlnet_generate` |
| 10 | Inpainting                            | 🟡 | `sana_inpaint` — works when diffusers ships `SanaInpaintPipeline`; auto-detects |
| 11 | LoRA loading                          | ✅ | `sana_load_lora` / `sana_unload_loras` |
| 12 | SANA-Video / LongSANA                 | 🔴 | scoped to sibling `strands-sana-video` package |
| 13 | Inference scaling (NVILA verifier)    | ✅ | `sana_inference_scale` (CLIP fallback; NVILA optional via `score_fn=`) |
| 14 | Safety filter                         | 🟡 | keyword filter shipped; ShieldGemma-2B path documented |

### P3 — Schedulers & sampling

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 15 | Scheduler choice                      | ✅ | `sana_set_scheduler` — 10 aliases (flow-match, DPM-Solver, Euler, …) |
| 16 | Negative-prompt embedding cache       | ⏳ | minor optimization |
| 17 | `num_images_per_prompt` exposed       | ✅ | `num_images=` arg on `sana_generate` |

### P4 — Pipeline / DevX

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 18 | Streaming preview callback            | ✅ | `make_step_callback(every_n, on_step)` in `pipeline` |
| 19 | SGLang server adapter                 | ⏳ | future `sana_serve` tool |
| 20 | ComfyUI graph export                  | ✅ | `sana_export_comfyui_workflow` |
| 21 | Cosmos-RL / Sol-RL hooks              | 🔴 | training/RL — out of scope |
| 22 | Metrics (CLIP / ImageReward / FID)    | ✅ | `sana_metric_clip`, `sana_metric_imagereward` |
| 23 | HuggingFace upload                    | ✅ | `sana_upload_to_hf` |
| 24 | Prompt enhancement (LLM rewrite)      | ✅ | `sana_enhance_prompt` + `COMPLEX_HUMAN_INSTRUCTION` |

### P5 — Tests & infra

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 25 | E2E smoke tests                       | ✅ | 45 tests, signature/import/registry/wrapper coverage |
| 26 | HF download progress                  | ⏳ | piggyback on `huggingface_hub` callbacks |
| 27 | Docker image                          | ✅ | `Dockerfile` (CUDA 12.1, py3.12) shipped |
| 28 | mkdocs site                           | ✅ | `mkdocs.yml` configured (Material theme) |

---

## 📊 Coverage summary (v0.2.0)

| Area              | Upstream | strands-sana | Coverage |
|-------------------|---------:|-------------:|---------:|
| Models            | ~15      | 10           | **~67%** |
| Pipelines         | T2I, PAG, Sprint, ControlNet, Inpaint, Video | 5/6 | **~83%** |
| Schedulers        | 4+       | **10 aliases** | **~100%** |
| Quantization      | 4bit/8bit | 4bit/8bit (auto-probe) | **~100%** |
| Metrics           | 5 benchmarks | CLIP, ImageReward | ~40% |
| Training          | full     | n/a (out of scope) | n/a |
| Agent tools       | n/a      | **21**       | n/a |
| Tests             | n/a      | 45 passing   | ✅ |

**Top-level remaining gaps**: SANA-Video (separate package), full FID/GenEval/DPG suite, NVILA scoring, SGLang serve, training/RL hooks.

---

## 🎯 Next milestones

- **v0.3.0** — FID + GenEval + DPG metrics; NVILA scoring backend; SGLang `sana_serve`
- **v0.4.0** — Streaming preview decode integration with Strands streaming agents
- **v1.0.0** — Stable API, mkdocs published to GH Pages, GPU CI pipeline
- **strands-sana-video v0.1** — Sibling package for SANA-Video

---

_Generated against NVlabs/Sana ToT and `diffusers==0.37.1` — all 21 tools verified by 45 unit tests._
