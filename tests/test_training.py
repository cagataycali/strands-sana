"""Tests for training tools (all dry-run, no GPU required).

Auto-clones upstream NVlabs/Sana into /tmp if not already on disk so
training-script paths can resolve.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from strands_sana.tools.training import (
    sana_train_lora, sana_train, sana_train_scm_ladd,
    sana_train_solrl, sana_train_video, sana_train_longsana,
    sana_list_training_configs, _resolve_sana_root,
)


@pytest.fixture(scope="session", autouse=True)
def _ensure_sana_root(tmp_path_factory):
    """Auto-clone Sana once per test session if not on disk."""
    # If already resolvable, skip.
    try:
        _resolve_sana_root()
        return
    except FileNotFoundError:
        pass
    # Use SANA_ROOT or clone shallow into a known location.
    sana_root = os.environ.get("SANA_ROOT")
    if sana_root and Path(sana_root).exists():
        os.environ["SANA_ROOT"] = sana_root
        return
    target = Path("/tmp/_sana_for_tests")
    if not (target / "train_scripts").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none",
                 "https://github.com/NVlabs/Sana.git", str(target)],
                check=True, timeout=180,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pytest.skip("Could not clone NVlabs/Sana — set SANA_ROOT to skip clone")
    os.environ["SANA_ROOT"] = str(target)


def _call(t, **kw):
    """Strands @tool now decorated as DecoratedFunctionTool — call directly."""
    if hasattr(t, "original_func"):
        return t.original_func(**kw)
    return t(**kw)
def test_resolve_sana_root_explicit(tmp_path):
    """Explicit sana_root with a fake `train_scripts` dir resolves."""
    fake = tmp_path / "MyFakeSana"
    (fake / "train_scripts").mkdir(parents=True)
    root = _resolve_sana_root(str(fake))
    assert root == fake


def test_resolve_sana_root_missing(tmp_path, monkeypatch):
    """Missing path should raise FileNotFoundError when no fallback exists."""
    # Move to an empty dir so cwd/Sana doesn't satisfy fallback
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SANA_ROOT", raising=False)
    import strands_sana.tools.training as t
    monkeypatch.setattr(t, "__file__", str(tmp_path / "fake" / "tools" / "training.py"))
    with pytest.raises(FileNotFoundError):
        # call the function FROM the patched module so __file__ is read correctly
        t._resolve_sana_root(str(tmp_path / "doesnotexist"))


def test_resolve_sana_root_default():
    """The repo we cloned at ./Sana should resolve."""
    root = _resolve_sana_root()
    assert root.exists()
    assert (root / "train_scripts").exists()


def test_lora_dry_run():
    r = _call(
        sana_train_lora,
        instance_data_dir="./data/sks-dog",
        instance_prompt="a sks dog",
        max_train_steps=100,
        dry_run=True,
    )
    assert r["status"] == "success"
    assert r["dry_run"] is True
    assert "train_dreambooth_lora_sana.py" in r["command"]
    assert "--instance_data_dir=./data/sks-dog" in r["command"]
    assert "--max_train_steps=100" in r["command"]


def test_train_dry_run():
    r = _call(sana_train, dry_run=True)
    assert r["status"] == "success"
    assert "train_scripts/train.py" in r["command"]


def test_scm_ladd_dry_run():
    r = _call(sana_train_scm_ladd, dry_run=True)
    assert r["status"] == "success"
    assert "train_scm_ladd.py" in r["command"]


def test_solrl_dry_run():
    r = _call(sana_train_solrl, dry_run=True)
    assert r["status"] == "success"
    assert "run_sana_single_node_8gpu.sh" in r["command"]


def test_video_train_dry_run():
    r = _call(sana_train_video, dry_run=True)
    assert r["status"] == "success"
    assert "train_video_ivjoint.py" in r["command"]


def test_video_train_chunk_dry_run():
    r = _call(sana_train_video, chunk=True, dry_run=True)
    assert "train_video_ivjoint_chunk.py" in r["command"]


def test_longsana_dry_run():
    r = _call(sana_train_longsana, dry_run=True)
    assert r["status"] == "success"
    assert "train_longsana.py" in r["command"]


def test_list_configs():
    r = _call(sana_list_training_configs)
    assert r["status"] == "success"
    assert r["count"] > 0
    # Should find at least the canonical configs
    cfgs = r["configs"]
    assert any("sana_config" in c for c in cfgs)
    assert any("sana_video_config" in c for c in cfgs)
    assert any("sana_sprint_config" in c for c in cfgs)
    assert any("sol_rl" in c for c in cfgs)
