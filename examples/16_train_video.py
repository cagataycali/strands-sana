"""Train SANA-Video / LongSANA from scratch or fine-tune."""
from strands_sana import sana_train_video, sana_train_longsana


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    print("=== SANA-Video FSDP ===")
    print(_call(sana_train_video, num_processes=8, dry_run=True)["command"])

    print("\n=== LongSANA ===")
    print(_call(sana_train_longsana, num_processes=8, dry_run=True)["command"])
