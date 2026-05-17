# DIFF.md — strands-sana vs NVlabs/Sana upstream

Comparison between this repo (`strands-sana` v0.1.0) and the upstream
[NVlabs/Sana](https://github.com/NVlabs/Sana) reference implementation.

`strands-sana` is a **thin agent-tool wrapper** around the `diffusers`
SanaPipeline family, designed to expose the entire upstream feature set
through clean Strands `@tool` functions. We do **not** vendor the upstream
research code — we depend on `diffusers` for the actual inference path,
which means new upstream features land here as soon as `diffusers`
ships them.

Status legend: ✅ done · 🟡 partial · ⏳ planned · 🔴 blocked / out of scope

---

## ✅ What strands-sana already covers

| Capability                                                 | Status |
|------------------------------------------------------------|:------:|
| `diffusers.SanaPipeline` text-to-image                     | ✅ |
| `diffusers.SanaPAGPipeline` (PAG)                          | ✅ |
| `diffusers.SanaSprintPipeline` (1-2 step)                  | ✅ |
| `diffusers.SanaControlNetPipeline`                         | ✅ |
| `diffusers.SanaInpaintPipeline`                            | 🟡 (auto-detected; falls back gracefully if absent) |
| Lazy-loaded, module-level pipeline cache                   | ✅ |
| Auto device select (cuda / mps / cpu)                      | ✅ |
| 10 checkpoints (1.0, 1.5, Sprint, 2K, 4K, Multiling)       | ✅ |
| Per-model defaults (steps / guidance / dtype / tiling)     | ✅ |
| Aspect-ratio binning (via `use_resolution_binning=True`)   | ✅ |
| LoRA load / unload at runtime                              | ✅ |
| Memory modes (fast / balanced / low / ultra-low)           | ✅ |
| VAE tiling auto-enabled on 2K/4K                           | ✅ |
| ComfyUI workflow JSON export                               | ✅ |
| Prompt-enhancement helpers (5 styles + COMPLEX template)   | ✅ |
| Lightweight safety gate                                    | 🟡 (keyword-filter; ShieldGemma optional upgrade) |
| 36 unit tests, all passing without checkpoint downloads    | ✅ |
| Apache-2.0, pyproject, CI tests, 10 examples               | ✅ |

---

## 🚧 Remaining backlog

### P0 — High-impact (status after v0.1.0)

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 1  | Sana-1.5 checkpoints (1.6B & 4.8B)    | ✅ | `sana-1.5-1.6b-1024`, `sana-1.5-4.8b-1024` |
| 2  | Sana-Sprint (1-2 step distilled)      | ✅ | `sana_sprint_generate` tool |
| 3  | 2K / 4K resolution checkpoints        | ✅ | with auto vae tiling |
| 4  | PAG (Perturbed Attention Guidance)    | ✅ | `pag_scale=` on `sana_generate` |
| 5  | Aspect-ratio binning                  | ✅ | enabled by default |

### P1 — Quantization & memory

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 6  | 4-bit (Nunchaku / SVDQuant)           | ⏳ | requires `optimum-quanto` or `nunchaku` install path; tool stub via `sana_set_memory_mode("ultra-low")` |
| 7  | CPU offload toggles                   | ✅ | exposed via `sana_set_memory_mode` |
| 8  | DC-AE-Lite / DCAE-1.1 VAE swap        | ⏳ | hot-swap VAE in cache helper TBD |

### P2 — Generation features

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 9  | ControlNet                            | ✅ | `sana_controlnet_generate`; user supplies preprocessed control map |
| 10 | Inpainting                            | 🟡 | tool present; needs `SanaInpaintPipeline` in your diffusers (auto-detects) |
| 11 | LoRA loading                          | ✅ | `sana_load_lora` / `sana_unload_loras` |
| 12 | SANA-Video / LongSANA                 | 🔴 | scoped to sibling `strands-sana-video` package |
| 13 | Inference scaling (NVILA verifier)    | ⏳ | best implemented as Strands meta-agent loop |
| 14 | Safety filter (ShieldGemma-2B)        | 🟡 | keyword-filter shipped; ShieldGemma path documented |

### P3 — Schedulers & sampling

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 15 | Scheduler choice (DPMS / FlowEuler)   | ⏳ | requires explicit scheduler swap on the cached pipeline |
| 16 | Negative-prompt embedding cache       | ⏳ | minor optimization |
| 17 | `num_images_per_prompt` exposed       | ✅ | `num_images` arg on `sana_generate` |

### P4 — Pipeline / DevX

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 18 | Async / streaming generation          | ⏳ | needs `callback_on_step_end` plumbing |
| 19 | SGLang server adapter                 | ⏳ | `sana_serve` tool TBD |
| 20 | ComfyUI graph export                  | ✅ | `sana_export_comfyui_workflow` |
| 21 | Cosmos-RL / Sol-RL hooks              | 🔴 | training/RL — out of scope for inference wrapper |
| 22 | Metrics & evaluation tools            | ⏳ | optional: CLIP / FID / GenEval / DPG / ImageReward |
| 23 | HuggingFace upload helper             | ⏳ | trivial wrapper around `huggingface_hub` |
| 24 | Prompt enhancement (LLM rewrite)      | ✅ | client-side `sana_enhance_prompt` + `COMPLEX_HUMAN_INSTRUCTION` template |

### P5 — Tests & infra

| #  | Item                                  | Status | Notes |
|----|---------------------------------------|:------:|-------|
| 25 | End-to-end smoke test (CPU-tiny)      | 🟡 | sig + import tests cover API drift; full inference test gated on download |
| 26 | Model download progress               | ⏳ | piggyback on `huggingface_hub` callbacks |
| 27 | Docker / Apptainer image              | ⏳ | publish `strands-sana:cuda12.1` |
| 28 | mkdocs site                           | ⏳ | `[docs]` extra exists; need `mkdocs.yml` |

---

## 📊 Coverage summary (v0.1.0)

| Area              | Upstream features | strands-sana | Coverage |
|-------------------|------------------:|-------------:|---------:|
| Models            | ~15 checkpoints   | 10           | **~67%** |
| Pipelines         | T2I, PAG, Sprint, ControlNet, Inpaint, Video | 5/6 (no video) | **~83%** |
| Schedulers        | 4+                | default + per-model | ~50% |
| Quantization      | 4bit / 8bit       | offload only | ~33% |
| Training          | full (FSDP/DDP)   | n/a (out of scope) | n/a |
| Metrics/Eval      | 5 benchmarks      | 0            | 0%       |
| Agent tools       | n/a               | **13**       | n/a      |

---

## 🎯 Next milestones

- **v0.2.0** — P1 quantization (#6 4bit, #8 VAE swap), P3 scheduler choice (#15)
- **v0.3.0** — P4 streaming preview (#18), metrics tools (#22)
- **v0.4.0** — Inference scaling meta-agent (#13), HF upload (#23), Docker (#27)
- **v1.0.0** — Stable API, mkdocs site (#28), full test coverage with optional GPU CI
- **strands-sana-video v0.1** — Sibling package for SANA-Video (#12)

---

_Generated against NVlabs/Sana ToT and `diffusers==0.37.1`._
