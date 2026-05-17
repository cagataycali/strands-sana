"""DreamBooth + LoRA fine-tuning on your own data."""
from strands_sana import sana_train_lora


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    # Dry-run first — see the command
    r = _call(
        sana_train_lora,
        instance_data_dir="./data/sks-dog",
        instance_prompt="a photo of sks dog",
        validation_prompt="a photo of sks dog in a yarn art style",
        max_train_steps=500,
        num_processes=4,
        dry_run=True,
    )
    print("Dry-run command:")
    print(r["command"])
    print("\nTo launch for real, set dry_run=False")
