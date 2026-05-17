"""Training tools — wraps upstream NVlabs/Sana training scripts (v0.4.0).

These tools shell out to the official Sana training scripts in the cloned
upstream repo. They auto-detect a `Sana/` directory in the project root,
or accept a `sana_root=` override.

Supported jobs:
- LoRA / DreamBooth fine-tuning      (`sana_train_lora`)
- Full pretrain / finetune           (`sana_train`)
- sCM-LADD distillation (Sana-Sprint)(`sana_train_scm_ladd`)
- Sol-RL post-training               (`sana_train_solrl`)
- SANA-Video FSDP training           (`sana_train_video`)
- LongSANA training                  (`sana_train_longsana`)

All tools default to `--dry-run` printing the command. Pass
`dry_run=False` to actually launch (requires GPUs + dependencies).
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

from strands import tool

logger = logging.getLogger(__name__)


def _resolve_sana_root(sana_root: Optional[str] = None) -> Path:
    """Find the upstream Sana repo. Priority:
    1. explicit `sana_root` arg
    2. `SANA_ROOT` env var
    3. `./Sana` relative to current working dir
    4. `../Sana` relative to package
    """
    candidates = []
    if sana_root:
        candidates.append(Path(sana_root))
    if os.getenv("SANA_ROOT"):
        candidates.append(Path(os.environ["SANA_ROOT"]))
    candidates.append(Path.cwd() / "Sana")
    pkg_dir = Path(__file__).resolve().parent.parent.parent
    candidates.append(pkg_dir / "Sana")

    for c in candidates:
        if c.exists() and (c / "train_scripts").exists():
            return c
    raise FileNotFoundError(
        "Sana upstream repo not found. Either:\n"
        "  - clone via `git clone https://github.com/NVlabs/Sana.git`\n"
        "  - pass `sana_root=/path/to/Sana`\n"
        "  - set env `SANA_ROOT=/path/to/Sana`"
    )


def _run_or_describe(cmd: List[str], cwd: Path, dry_run: bool, env: Optional[dict] = None) -> dict:
    """Either dry-run or actually execute `cmd`. Returns status dict."""
    pretty = " ".join(shlex.quote(c) for c in cmd)
    if dry_run:
        return {
            "status": "success",
            "dry_run": True,
            "command": pretty,
            "cwd": str(cwd),
            "note": "Pass dry_run=False to actually launch this training job.",
        }
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=full_env, start_new_session=True,
        )
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e), "command": pretty}
    return {
        "status": "success",
        "dry_run": False,
        "pid": proc.pid,
        "command": pretty,
        "cwd": str(cwd),
        "note": f"Launched training. Tail logs / kill via PID {proc.pid}.",
    }


# ────────────────────────────────────────────────────────────────────
# LoRA / DreamBooth
# ────────────────────────────────────────────────────────────────────
@tool
def sana_train_lora(
    instance_data_dir: str,
    output_dir: str = "./trained-sana-lora",
    pretrained_model: str = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    instance_prompt: str = "a photo of sks",
    validation_prompt: Optional[str] = None,
    resolution: int = 1024,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    max_train_steps: int = 500,
    use_8bit_adam: bool = True,
    mixed_precision: str = "bf16",
    seed: int = 0,
    num_processes: int = 1,
    push_to_hub: bool = False,
    hub_repo: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    sana_root: Optional[str] = None,
    dry_run: bool = True,
) -> dict:
    """Fine-tune Sana with LoRA / DreamBooth.

    Wraps `train_scripts/train_dreambooth_lora_sana.py`.
    """
    try:
        root = _resolve_sana_root(sana_root)
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}

    cmd = [
        "accelerate", "launch",
        f"--num_processes={num_processes}",
        "train_scripts/train_dreambooth_lora_sana.py",
        f"--pretrained_model_name_or_path={pretrained_model}",
        f"--instance_data_dir={instance_data_dir}",
        f"--output_dir={output_dir}",
        f"--mixed_precision={mixed_precision}",
        f"--instance_prompt={instance_prompt}",
        f"--resolution={resolution}",
        f"--train_batch_size={train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--learning_rate={learning_rate}",
        f"--max_train_steps={max_train_steps}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--seed={seed}",
    ]
    if use_8bit_adam:
        cmd.append("--use_8bit_adam")
    if validation_prompt:
        cmd.append(f"--validation_prompt={validation_prompt}")
        cmd.append("--validation_epochs=25")
    if push_to_hub:
        cmd.append("--push_to_hub")
        if hub_repo:
            cmd.append(f"--hub_model_id={hub_repo}")
    if extra_args:
        cmd.extend(extra_args)

    return _run_or_describe(cmd, cwd=root, dry_run=dry_run)


# ────────────────────────────────────────────────────────────────────
# Full pretrain / finetune
# ────────────────────────────────────────────────────────────────────
@tool
def sana_train(
    config_path: str = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml",
    work_dir: str = "./output/sana_train",
    num_processes: int = 1,
    name: str = "tmp",
    resume_from: str = "latest",
    report_to: str = "tensorboard",
    debug: bool = False,
    extra_args: Optional[List[str]] = None,
    sana_root: Optional[str] = None,
    dry_run: bool = True,
) -> dict:
    """Full Sana training (`train_scripts/train.py`)."""
    try:
        root = _resolve_sana_root(sana_root)
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_processes}",
        f"--master_port={20000 + os.getpid() % 10000}",
        "train_scripts/train.py",
        f"--config_path={config_path}",
        f"--work_dir={work_dir}",
        f"--name={name}",
        f"--resume_from={resume_from}",
        f"--report_to={report_to}",
    ]
    if debug:
        cmd.append("--debug=true")
    if extra_args:
        cmd.extend(extra_args)
    return _run_or_describe(cmd, cwd=root, dry_run=dry_run)


# ────────────────────────────────────────────────────────────────────
# sCM-LADD distillation (Sana-Sprint)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_train_scm_ladd(
    config_path: str = "configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml",
    work_dir: str = "./output/sCM_ladd",
    num_processes: int = 2,
    name: str = "tmp",
    debug: bool = False,
    extra_args: Optional[List[str]] = None,
    sana_root: Optional[str] = None,
    dry_run: bool = True,
) -> dict:
    """Distill Sana-Sprint via sCM-LADD (`train_scripts/train_scm_ladd.py`)."""
    try:
        root = _resolve_sana_root(sana_root)
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_processes}",
        f"--master_port={20000 + os.getpid() % 10000}",
        "train_scripts/train_scm_ladd.py",
        f"--config_path={config_path}",
        f"--work_dir={work_dir}",
        f"--name={name}",
        "--resume_from=latest",
        "--report_to=tensorboard",
    ]
    if debug:
        cmd.append("--debug=true")
    if extra_args:
        cmd.extend(extra_args)
    env = {"TRITON_PRINT_AUTOTUNING": "1"}
    return _run_or_describe(cmd, cwd=root, dry_run=dry_run, env=env)


# ────────────────────────────────────────────────────────────────────
# Sol-RL post-training
# ────────────────────────────────────────────────────────────────────
@tool
def sana_train_solrl(
    config_spec: str = "configs/sol_rl/sana.py:sana_diffusionnft_pickscore",
    nproc_per_node: int = 8,
    cuda_visible_devices: str = "0,1,2,3,4,5,6,7",
    sana_native_model_path: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    sana_root: Optional[str] = None,
    dry_run: bool = True,
) -> dict:
    """Sol-RL post-training for Sana (`train_scripts/sol_rl/run_sana_single_node_8gpu.sh`).

    NVFP4 rollout + BF16 training. Requires the Sana native checkpoint
    (auto-downloaded to `output/pretrained_models/SANA_LinearFFN.pth`).
    """
    try:
        root = _resolve_sana_root(sana_root)
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}

    cmd = ["bash", "train_scripts/sol_rl/run_sana_single_node_8gpu.sh"]
    if extra_args:
        cmd.extend(extra_args)

    env = {
        "NPROC_PER_NODE": str(nproc_per_node),
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "CONFIG_SPEC": config_spec,
        "DISABLE_XFORMERS": "1",
    }
    if sana_native_model_path:
        env["SANA_NATIVE_MODEL_PATH"] = sana_native_model_path
    return _run_or_describe(cmd, cwd=root, dry_run=dry_run, env=env)


# ────────────────────────────────────────────────────────────────────
# SANA-Video training (FSDP)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_train_video(
    config_path: str = "configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml",
    work_dir: str = "./output/sana_video",
    num_processes: int = 1,
    name: str = "tmp",
    chunk: bool = False,
    extra_args: Optional[List[str]] = None,
    sana_root: Optional[str] = None,
    dry_run: bool = True,
) -> dict:
    """SANA-Video FSDP training.

    Wraps `train_video_scripts/train_video_ivjoint.py` (or `_chunk` for
    long-context training).
    """
    try:
        root = _resolve_sana_root(sana_root)
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}

    script = (
        "train_video_scripts/train_video_ivjoint_chunk.py" if chunk
        else "train_video_scripts/train_video_ivjoint.py"
    )

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_processes}",
        f"--master_port={20000 + os.getpid() % 10000}",
        script,
        f"--config_path={config_path}",
        f"--work_dir={work_dir}",
        f"--name={name}",
        "--resume_from=latest",
        "--report_to=tensorboard",
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = {"DISABLE_XFORMERS": "1"}
    return _run_or_describe(cmd, cwd=root, dry_run=dry_run, env=env)


# ────────────────────────────────────────────────────────────────────
# LongSANA training (real-time minute-long video)
# ────────────────────────────────────────────────────────────────────
@tool
def sana_train_longsana(
    config_path: str = "configs/sana_video_config/Sana_2000M_480px_adamW_fsdp_longsana.yaml",
    work_dir: str = "./output/longsana",
    num_processes: int = 1,
    name: str = "tmp",
    extra_args: Optional[List[str]] = None,
    sana_root: Optional[str] = None,
    dry_run: bool = True,
) -> dict:
    """LongSANA training — minute-long, real-time video generation.

    Wraps `train_video_scripts/train_longsana.py`.
    """
    try:
        root = _resolve_sana_root(sana_root)
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_processes}",
        f"--master_port={20000 + os.getpid() % 10000}",
        "train_video_scripts/train_longsana.py",
        f"--config_path={config_path}",
        f"--work_dir={work_dir}",
        f"--name={name}",
        "--resume_from=latest",
        "--report_to=tensorboard",
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = {"DISABLE_XFORMERS": "1"}
    return _run_or_describe(cmd, cwd=root, dry_run=dry_run, env=env)


# ────────────────────────────────────────────────────────────────────
# Listing / introspection
# ────────────────────────────────────────────────────────────────────
@tool
def sana_list_training_configs(
    sana_root: Optional[str] = None,
) -> dict:
    """Enumerate all `*.yaml`/`*.py` training configs in upstream Sana."""
    try:
        root = _resolve_sana_root(sana_root)
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}

    configs = []
    for cfg_dir in (root / "configs").rglob("*"):
        if cfg_dir.suffix in (".yaml", ".py") and cfg_dir.is_file():
            rel = cfg_dir.relative_to(root)
            configs.append(str(rel))
    return {
        "status": "success",
        "sana_root": str(root),
        "count": len(configs),
        "configs": sorted(configs),
    }
