"""Sol-RL post-training (NVFP4 rollout + BF16) for Sana."""
from strands_sana import sana_train_solrl


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    r = _call(
        sana_train_solrl,
        config_spec="configs/sol_rl/sana.py:sana_diffusionnft_pickscore",
        nproc_per_node=8,
        dry_run=True,
    )
    print(r)
