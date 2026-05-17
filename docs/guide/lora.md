# LoRA Adapters

Load and unload LoRA weights at runtime to specialize a base Sana model.

```python
from strands_sana import sana_load_lora, sana_generate, sana_unload_loras

sana_load_lora(
    repo_or_path="user/sana-yarnart-lora",
    scale=0.8,
    model="sana-1.6b-1024",
)

sana_generate(prompt="a duck in yarn art style",
              model="sana-1.6b-1024", steps=15)

sana_unload_loras(model="sana-1.6b-1024")
```

## Requires PEFT

```bash
pip install 'strands-sana[lora]'
# OR: pip install peft
```

## Stacking multiple LoRAs

```python
sana_load_lora("user/style-lora", scale=0.6, adapter_name="style")
sana_load_lora("user/character-lora", scale=0.7, adapter_name="char")
sana_generate(prompt="...")
```

The wrapper tracks active adapters in `pipe._loaded_loras`.

## Sources

- HuggingFace Hub: `user/repo` format
- Local path: `/path/to/lora.safetensors`
- Train your own: see [Training](training.md)

## Train your own LoRA

```python
from strands_sana import sana_train_lora

sana_train_lora(
    instance_data_dir="./my-photos",
    instance_prompt="a photo of sks dog",
    max_train_steps=500,
    num_processes=4,    # 4 GPUs
    dry_run=False,      # actually launch
)
```

→ Full training guide: **[Training](training.md)**
