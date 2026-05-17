# Bugs found & fixed via real-world testing on Thor

Tested on NVIDIA Thor dev kit (aarch64, sm_110, CUDA 13).

| # | Severity | Bug | Fix | Commit |
|---|----------|-----|-----|--------|
| 1 | medium | Test scaffolding `_call(t, ...)` pattern using `.original_func` doesn't exist on real Strands `DecoratedFunctionTool` | Tests fall through to direct call `t(**kw)` (already had fallback, just confirmed) | n/a |
| 2 | low | training tests fail when no `Sana/` repo present (cross-machine portability) | Auto-clone shallow Sana into `/tmp/_sana_for_tests` via session fixture | 59a0a34 |
| 3 | **critical** | `negative_prompt or None` → `AttributeError: 'NoneType' object has no attribute 'lower'` when `guidance_scale > 1.0` | Use empty string `""` not `None` so diffusers' `_text_preprocessing` is happy | 59a0a34 |
| 4 | **critical** | All 5 video model HF repos are 404 (made-up names) | Use canonical names from upstream `model_zoo.md`: `Sana-Video_2B_480p_diffusers`, `SANA-Video_2B_720p_diffusers`, `SANA-Video_2B_480p_LongLive_diffusers`. Drop bogus 256px. | a0a665c |
| 5 | **critical** | Sana-Sprint with `steps != 2` → `ValueError: Intermediate timesteps for SCM is not supported when num_inference_steps != 2` | Pass `intermediate_timesteps=None` when steps≠2 for both `sprint` and `sprint-i2i` kinds | 891b91b |
| 6 | **critical** | SANA-Video errors on `(h, w)` not divisible by 32; default binning produces 624×624 from 480×480 | Snap (h, w) to multiples of 32 + disable `use_resolution_binning` for video | e2921d5 |
| 7 | low | MP4 export silently falls back to WebP without `imageio-ffmpeg` | Multi-tier fallback: diffusers.export_to_video → imageio direct → animated WebP. Added `[video]` extra in pyproject. | c9f2472 |

## Verified working on Thor

| Test | Time | Result |
|---|---:|---|
| `sana_generate` (sana-0.6b-512, 10 steps) | 45.8s | ✅ PNG 512×512 |
| `sana_generate` PAG variant (pag_scale=2.0) | 29.2s | ✅ PNG 512×512 |
| `sana_sprint_generate` (0.6B, 2 step) | 48.9s | ✅ PNG 1024×1024 |
| `sana_sprint_generate` (0.6B, 4 step) | 43.0s | ✅ PNG 1024×1024 |
| `sana_img2img` (sprint-i2i 1.6B, 2 step) | 143.2s (incl 10GB DL) | ✅ PNG 1024×1024 |
| `sana_video_generate` (480p, 33 frames, 10 steps) | 75.6s | ✅ MP4 ~110 KB |
| `sana_export_comfyui_workflow` | 0.0s | ✅ JSON |
| `pytest tests/` | 6.8s | ✅ 72/72 |

## Total

- 7 bugs found via real-world testing (4 of them critical)
- All fixed, all pushed
- 5 commits on top of v0.4.0
- 72 tests still passing
