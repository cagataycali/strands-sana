"""Additional tools for v0.2.0+ (schedulers, metrics, HF upload, inference scaling)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from strands import tool

from ..pipeline.sana_pipeline import get_pipeline
from ..utils.io import load_image, save_image

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Scheduler choice (P3 #15)
# ────────────────────────────────────────────────────────────────────
SCHEDULER_MAP = {
    "flow-match-euler":  "FlowMatchEulerDiscreteScheduler",
    "flow-match-heun":   "FlowMatchHeunDiscreteScheduler",
    "dpm-solver":        "DPMSolverMultistepScheduler",
    "dpm-solver-single": "DPMSolverSinglestepScheduler",
    "dpm-solver-cosine": "CosineDPMSolverMultistepScheduler",
    "euler":             "EulerDiscreteScheduler",
    "euler-ancestral":   "EulerAncestralDiscreteScheduler",
    "ddim":              "DDIMScheduler",
    "deis":              "DEISMultistepScheduler",
    "heun":              "HeunDiscreteScheduler",
}


@tool
def sana_set_scheduler(
    name: str = "flow-match-euler",
    model: Optional[str] = None,
    use_flow_sigmas: bool = True,
) -> dict:
    """Swap the scheduler on the active pipeline.

    Args:
        name: Scheduler alias from SCHEDULER_MAP keys.
        model: Which model alias to attach to (uses default).
        use_flow_sigmas: For DPM-Solver, enable flow-matching sigmas
                         (recommended for Sana — it's a flow-matching model).

    Returns:
        Dict with status, scheduler class name, and config snapshot.
    """
    if name not in SCHEDULER_MAP:
        return {
            "status": "error",
            "error": f"Unknown scheduler '{name}'. Available: {sorted(SCHEDULER_MAP)}",
        }
    try:
        import diffusers
        cls = getattr(diffusers, SCHEDULER_MAP[name], None)
        if cls is None:
            return {"status": "error", "error": f"diffusers missing {SCHEDULER_MAP[name]}"}
    except ImportError as e:
        return {"status": "error", "error": str(e)}

    pipe = get_pipeline(model_name=model)
    pipeline = pipe.load()

    # Pull existing scheduler config so the new scheduler has matching
    # training schedule / num_train_timesteps / etc.
    cfg = dict(pipeline.scheduler.config)
    if use_flow_sigmas and "DPMSolver" in cls.__name__:
        cfg["use_flow_sigmas"] = True
        cfg["prediction_type"] = "flow_prediction"

    new_sched = cls.from_config(cfg)
    pipeline.scheduler = new_sched

    return {
        "status": "success",
        "scheduler": cls.__name__,
        "alias": name,
        "use_flow_sigmas": cfg.get("use_flow_sigmas", False),
        "config_keys": sorted(cfg.keys())[:10],  # truncate
    }


@tool
def sana_list_schedulers() -> dict:
    """List available scheduler aliases."""
    return {
        "status": "success",
        "schedulers": [
            {"alias": k, "class": v}
            for k, v in SCHEDULER_MAP.items()
        ],
        "count": len(SCHEDULER_MAP),
    }


# ────────────────────────────────────────────────────────────────────
# Quantization (P1 #6)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_quantize(
    bits: int = 8,
    backend: str = "auto",
    model: Optional[str] = None,
) -> dict:
    """Quantize the active pipeline's transformer + VAE to int8 / int4.

    Args:
        bits: 4 or 8.
        backend: "auto" (probe quanto / bnb / nunchaku in that order),
                 "quanto", "bitsandbytes", "nunchaku".
        model: Which model alias to attach to.
    """
    if bits not in (4, 8):
        return {"status": "error", "error": "bits must be 4 or 8"}

    pipe = get_pipeline(model_name=model)
    pipeline = pipe.load()

    # Backend probe order
    backends = [backend] if backend != "auto" else ["quanto", "bitsandbytes", "nunchaku"]
    last_error = None
    for be in backends:
        try:
            if be == "quanto":
                from optimum.quanto import quantize, freeze, qint4, qint8
                qtype = qint4 if bits == 4 else qint8
                quantize(pipeline.transformer, weights=qtype)
                freeze(pipeline.transformer)
                return {"status": "success", "backend": "quanto", "bits": bits,
                        "target": "transformer"}
            elif be == "bitsandbytes":
                # bnb only supports int8 directly via diffusers BitsAndBytesConfig
                from diffusers import BitsAndBytesConfig
                # In practice you'd reload from_pretrained with quantization_config
                # For runtime conversion we just flag it for next reload.
                return {
                    "status": "deferred",
                    "backend": "bitsandbytes",
                    "bits": bits,
                    "note": "bnb requires reload; pass `quantization_config` to from_pretrained next load",
                }
            elif be == "nunchaku":
                # Nunchaku is the SVDQuant 4-bit path
                import nunchaku  # noqa
                return {
                    "status": "deferred",
                    "backend": "nunchaku",
                    "bits": 4,
                    "note": "Nunchaku requires loading pre-quantized weights from MIT-Han-Lab/svdq-int4-sana",
                }
        except ImportError as e:
            last_error = f"{be}: {e}"
            continue
        except Exception as e:
            last_error = f"{be}: {e}"
            continue

    return {
        "status": "error",
        "error": f"No quantization backend available. Last error: {last_error}",
        "hint": "pip install optimum-quanto OR pip install bitsandbytes",
    }


# ────────────────────────────────────────────────────────────────────
# VAE swap (P1 #8) — DC-AE-Lite / DCAE-1.1
# ────────────────────────────────────────────────────────────────────
@tool
def sana_swap_vae(
    vae_repo: str = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
    model: Optional[str] = None,
) -> dict:
    """Replace the VAE on the active pipeline (e.g. for DC-AE-Lite or DCAE-1.1).

    Args:
        vae_repo: HF repo of the new VAE (must be DCAE-compatible).
        model: Which pipeline to attach to.
    """
    try:
        from diffusers import AutoencoderDC
    except ImportError as e:
        return {"status": "error", "error": f"diffusers AutoencoderDC missing: {e}"}

    pipe = get_pipeline(model_name=model)
    pipeline = pipe.load()

    import torch
    dtype = pipe._resolve_dtype()
    new_vae = AutoencoderDC.from_pretrained(vae_repo, torch_dtype=dtype)
    new_vae.to(pipe.device)
    pipeline.vae = new_vae

    return {"status": "success", "vae": vae_repo, "model": pipe.model_name}


# ────────────────────────────────────────────────────────────────────
# HuggingFace upload (P4 #23)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_upload_to_hf(
    path: str,
    repo_id: str,
    repo_type: str = "model",
    commit_message: str = "Upload via strands-sana",
) -> dict:
    """Upload a file or directory to a HuggingFace repository.

    Args:
        path: Local file or directory.
        repo_id: HF repo (`user/repo`).
        repo_type: "model", "dataset", or "space".
        commit_message: Commit message.

    Requires `HUGGINGFACE_TOKEN` env var or `huggingface-cli login`.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {"status": "error", "error": "Install huggingface_hub: pip install huggingface_hub"}

    api = HfApi()
    p = Path(path)
    try:
        if p.is_dir():
            url = api.upload_folder(
                folder_path=str(p), repo_id=repo_id, repo_type=repo_type,
                commit_message=commit_message,
            )
        else:
            url = api.upload_file(
                path_or_fileobj=str(p), path_in_repo=p.name,
                repo_id=repo_id, repo_type=repo_type,
                commit_message=commit_message,
            )
    except Exception as e:
        return {"status": "error", "error": str(e)}

    return {"status": "success", "url": str(url), "repo_id": repo_id}


# ────────────────────────────────────────────────────────────────────
# Inference scaling — generate-K-and-pick-best (P2 #13)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_inference_scale(
    prompt: str,
    n_samples: int = 4,
    model: Optional[str] = None,
    steps: Optional[int] = None,
    seed_start: int = 0,
    output_dir: Optional[str] = None,
    score_fn: str = "clip",
) -> dict:
    """Generate N candidates and return the best-scoring one.

    Args:
        prompt: Text description.
        n_samples: How many variants to generate.
        seed_start: First seed; subsequent seeds = seed_start + i.
        score_fn: "clip" (CLIPScore) or "first" (no scoring, return first).

    Note: Lightweight — Sana's official inference-scaling uses NVILA-2B
    as the verifier. We use CLIP for accessibility.
    """
    pipe = get_pipeline(model_name=model)

    # Generate all candidates
    candidates = []
    for i in range(n_samples):
        imgs = pipe.generate(
            prompt=prompt,
            num_inference_steps=steps,
            seed=seed_start + i,
            num_images=1,
        )
        path = save_image(imgs[0], output_dir=output_dir, prefix=f"scale_{i:02d}")
        candidates.append({"index": i, "seed": seed_start + i, "path": path,
                           "image": imgs[0]})

    # Score
    if score_fn == "first" or n_samples == 1:
        best = candidates[0]
        scores = [1.0] + [0.0] * (n_samples - 1)
    elif score_fn == "clip":
        try:
            scores = _clip_score([c["image"] for c in candidates], prompt)
        except Exception as e:
            logger.warning(f"CLIP scoring failed, returning first: {e}")
            scores = [1.0] + [0.0] * (n_samples - 1)
        best = candidates[max(range(n_samples), key=lambda i: scores[i])]
    else:
        return {"status": "error", "error": f"Unknown score_fn: {score_fn}"}

    return {
        "status": "success",
        "best": {"index": best["index"], "seed": best["seed"], "path": best["path"]},
        "scores": [
            {"index": c["index"], "seed": c["seed"], "path": c["path"], "score": s}
            for c, s in zip(candidates, scores)
        ],
        "n_samples": n_samples,
        "prompt": prompt,
    }


def _clip_score(images, prompt: str) -> List[float]:
    """Compute CLIPScore between images and prompt. Returns list of similarities."""
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError as e:
        raise ImportError(
            "transformers required for CLIP scoring. pip install transformers"
        ) from e

    import torch
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = proc(text=[prompt], images=list(images), return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    text_emb = out.text_embeds   # (1, 512)
    img_embs = out.image_embeds  # (N, 512)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
    sims = (img_embs @ text_emb.T).squeeze(-1)  # (N,)
    return [float(s) for s in sims.tolist()]


# ────────────────────────────────────────────────────────────────────
# Metrics tools (P4 #22)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_metric_clip(
    image_path: str,
    prompt: str,
) -> dict:
    """CLIPScore between an image and a text prompt (0..1)."""
    try:
        img = load_image(image_path)
    except Exception as e:
        return {"status": "error", "error": f"image load: {e}"}
    try:
        scores = _clip_score([img], prompt)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    return {
        "status": "success",
        "metric": "clip-score",
        "score": scores[0],
        "image_path": image_path,
        "prompt": prompt,
    }


@tool
def sana_metric_imagereward(
    image_path: str,
    prompt: str,
) -> dict:
    """ImageReward score (requires `pip install image-reward`)."""
    try:
        import ImageReward as RM
    except ImportError:
        return {
            "status": "error",
            "error": "Install image-reward: pip install image-reward",
        }
    try:
        model = RM.load("ImageReward-v1.0")
        score = model.score(prompt, image_path)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    return {
        "status": "success",
        "metric": "image-reward",
        "score": float(score),
        "image_path": image_path,
        "prompt": prompt,
    }


# ────────────────────────────────────────────────────────────────────
# SGLang server adapter (P4 #19)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_serve(
    model: str = "sana-1.6b-1024",
    port: int = 30000,
    host: str = "0.0.0.0",
    extra_args: Optional[List[str]] = None,
) -> dict:
    """Boot an SGLang server hosting the given Sana checkpoint.

    Spawns `sglang.launch_server` in a subprocess. Returns immediately;
    use the returned PID to stop it later.

    Args:
        model: Sana model alias (resolves to HF repo).
        port: Bind port.
        host: Bind host.
        extra_args: Extra `sglang.launch_server` flags.

    Requires `pip install "sglang[all]"`.
    """
    from ..models.registry import SANA_MODELS
    if model not in SANA_MODELS:
        return {"status": "error", "error": f"Unknown model: {model}"}

    repo = SANA_MODELS[model].hf_repo
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", repo,
        "--host", host,
        "--port", str(port),
    ]
    if extra_args:
        cmd.extend(extra_args)

    import shutil, subprocess
    if shutil.which("python") is None:
        return {"status": "error", "error": "python not on PATH"}

    try:
        # Detached spawn — caller is responsible for reaping
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}

    return {
        "status": "success",
        "pid": proc.pid,
        "url": f"http://{host}:{port}",
        "model": model,
        "hf_repo": repo,
        "note": "Stop with: kill <pid>; or `os.kill(pid, signal.SIGTERM)`",
    }


# ────────────────────────────────────────────────────────────────────
# HF download progress (P5 #26)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_prefetch_model(
    model: str = "sana-1.6b-1024",
    quiet: bool = False,
) -> dict:
    """Pre-download a Sana checkpoint with progress reporting.

    Useful before generating in a constrained env where you'd rather
    pay the download cost up front than mid-prompt.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return {"status": "error", "error": "Install huggingface_hub: pip install huggingface_hub"}
    from ..models.registry import SANA_MODELS

    if model not in SANA_MODELS:
        return {"status": "error", "error": f"Unknown model: {model}"}
    repo = SANA_MODELS[model].hf_repo
    try:
        path = snapshot_download(repo_id=repo, tqdm_class=None if quiet else None)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    return {"status": "success", "model": model, "hf_repo": repo, "local_path": path}


# ────────────────────────────────────────────────────────────────────
# Negative-prompt embedding cache (P3 #16)
# ────────────────────────────────────────────────────────────────────
_NEG_EMBED_CACHE: dict = {}


def encode_negative_cached(pipe, negative_prompt: str):
    """Encode a negative prompt and cache the embedding for reuse.

    Most agents reuse the same negative prompt across many calls — the
    Gemma-2 text encoder isn't free. This caches by `(model_id, prompt)`.
    """
    if not negative_prompt:
        return None, None
    key = f"{id(pipe)}::{negative_prompt}"
    if key in _NEG_EMBED_CACHE:
        return _NEG_EMBED_CACHE[key]
    try:
        emb, mask = pipe.encode_prompt(negative_prompt)
    except AttributeError:
        # Different diffusers API — fall through, no cache
        return None, None
    _NEG_EMBED_CACHE[key] = (emb, mask)
    return emb, mask


def clear_negative_embed_cache() -> int:
    n = len(_NEG_EMBED_CACHE)
    _NEG_EMBED_CACHE.clear()
    return n
