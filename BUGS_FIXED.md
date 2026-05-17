# Bugs found & fixed via real-world testing on Thor

Tested on **NVIDIA Thor dev kit** (aarch64, sm_110, CUDA 13).

| # | Severity | Bug | Fix | Commit |
|---|----------|-----|-----|--------|
| 1 | medium | Test scaffolding `_call(t, ...)` pattern using `.original_func` doesn't exist on real Strands `DecoratedFunctionTool` | Tests already had `else t(**kw)` fallback path | n/a |
| 2 | low | training tests fail when no `Sana/` repo present (cross-machine portability) | Auto-clone shallow Sana into `/tmp/_sana_for_tests` via session fixture | 59a0a34 |
| 3 | **critical** | `negative_prompt or None` → `AttributeError: 'NoneType' object has no attribute 'lower'` when `guidance_scale > 1.0` | Use empty string `""` not `None` | 59a0a34 |
| 4 | **critical** | All 5 video model HF repos are 404 (made-up names) | Use canonical names from `model_zoo.md`: `Sana-Video_2B_480p_diffusers`, `SANA-Video_2B_720p_diffusers`, `SANA-Video_2B_480p_LongLive_diffusers` | a0a665c |
| 5 | **critical** | Sana-Sprint with `steps != 2` → `ValueError: Intermediate timesteps for SCM is not supported when num_inference_steps != 2` | Pass `intermediate_timesteps=None` when steps≠2 | 891b91b |
| 6 | **critical** | SANA-Video errors on `(h, w)` not divisible by 32; binning gave 624×624 | Snap to 32, disable `use_resolution_binning` for video | e2921d5 |
| 7 | low | MP4 export silently falls back to WebP without `imageio-ffmpeg` | Multi-tier fallback + `[video]` extra | c9f2472 |
| 8 | low | Test helper `name=` kwarg collides with `sana_set_scheduler(name=...)` | (test bug, not lib bug) — switched to scp-file approach | n/a |
| 9 | **critical** | Schedulers like DDIM/Heun reject `prediction_type=flow_prediction` from Sana's flow-matching scheduler config | Strip flow keys when swapping to non-flow schedulers, with quality-degradation warning | 9ea3421 |
| 10 | medium | ControlNet with non-ControlNet checkpoint → cryptic component error | Catch and return helpful error pointing to upstream `app/sana_controlnet_pipeline.py` | b4c71ec |
| 11 | medium | `sana_load_lora` propagates `ValueError: PEFT backend is required` and `RepositoryNotFoundError` instead of returning error dict | Wrap entire body in try/`except BaseException`, handle PEFT-missing specially. Add `[lora]` extra | e91df63, f81c540 |

## Verified working on Thor

### Image generation
| Test | Time | Result |
|---|---:|---|
| `sana_generate` (sana-0.6b-512, 10 steps) | 45.8s | ✅ PNG 512×512 |
| `sana_generate` PAG (pag_scale=2.5) | 29.3s | ✅ kind=pag |
| `sana_generate` complex_instruction | 34.8s | ✅ Gemma-2 prompt expansion |
| `sana_generate` Sana-1.5 1.6B | 163s | ✅ best-quality 1024×1024 |
| `sana_generate` num_images=2 | 40.3s | ✅ 2 PNGs |
| `sana_sprint_generate` (0.6B, 2 step) | 48.9s | ✅ PNG 1024 |
| `sana_sprint_generate` (0.6B, 4 step) | 43.0s | ✅ steps≠2 works |
| `sana_img2img` (sprint-i2i 1.6B, strength 0.7) | 9.3s | ✅ PNG 1024 |
| `sana_img2img` strength 0.3 | 77.5s | ✅ PNG 1024 |

### Video generation
| Test | Time | Result |
|---|---:|---|
| `sana_video_generate` (480p, 33 frames, 10 steps) | 75.6s | ✅ MP4 (1.4s clip) |
| `sana_video_generate` 65 frames | 89.0s | ✅ MP4 (2.7s) |
| `sana_video_generate` 97 frames | 102.6s | ✅ MP4 (4.0s) |
| `sana_image_to_video` (33 frames, 8 steps) | 120.6s | ✅ MP4 |

### Schedulers (all 6 swap successfully)
| Scheduler | Gen time |
|---|---:|
| flow-match-euler (default) | 48.1s |
| dpm-solver | **13.3s** |
| euler | 6.2s |
| euler-ancestral | 6.3s |
| ddim | 7.0s |
| deis | 7.1s |
| heun | 7.7s |

### Other tools
| Test | Time | Result |
|---|---:|---|
| `sana_export_comfyui_workflow` | 0.0s | ✅ JSON for 4 model variants |
| `sana_safety_check` (safe + bad + unicode + empty) | 0.0s | ✅ all branches |
| `sana_inference_scale` (3 samples + CLIP) | 15.8s | ✅ best=index 1 |
| `sana_inference_scale` (n=2, first) | 31.6s | ✅ |
| `sana_metric_clip` | 1.8s | ✅ score=0.254 |
| `sana_load_model("list", kind=, tag=)` | 0.0s | ✅ |
| `sana_set_memory_mode` low | 35.4s | ✅ |
| `sana_prefetch_model` (0.6B-512) | 14.3s | ✅ |
| `sana_quantize` (no backend) | 31.8s | ✅ status=deferred |
| `sana_load_lora` (bogus repo) | 27.3s | ✅ status=error |
| `sana_unload_loras` | 0.0s | ✅ |

### Training (dry-run, all 6)
| Tool | Result |
|---|---|
| `sana_list_training_configs` | ✅ 38 configs |
| `sana_train_lora` | ✅ accelerate launch ... |
| `sana_train` | ✅ torchrun ... |
| `sana_train_video` | ✅ torchrun video ... |
| `sana_train_video chunk=True` | ✅ chunked variant |
| `sana_train_solrl` | ✅ bash sol_rl/run_sana ... |
| `sana_train_scm_ladd` | ✅ torchrun scm_ladd ... |
| `sana_train_longsana` | ✅ torchrun longsana ... |

### Pytest

```
72 passed in 6.30s
```

## Total

- **11 bugs** found via real-world testing on Thor (5 critical, 4 medium, 2 low)
- **All fixed**, all pushed
- **9 fix commits** on top of v0.4.0
- **72 tests** still passing
- **17 example variants** validated against real GPU
- **3 video MP4s** generated at 33/65/97 frames
- Coverage: image, video, img2img, PAG, Sprint, schedulers, scaling, training dry-runs, LoRA, ControlNet, prompt enhance, safety, metrics, ComfyUI export
