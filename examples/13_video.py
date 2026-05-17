"""Generate hero assets for README — same script that produced the README images.

Run on a CUDA box (downloads ~10GB for sana-1.5 + ~10GB for sana-video).
"""
import os
from strands_sana import (
    sana_generate, sana_sprint_generate, sana_video_generate, sana_clear_cache,
)

OUT = "./generated/hero"
os.makedirs(OUT, exist_ok=True)

# 1. Best-quality image
sana_generate(
    prompt=("a majestic rubber duck wearing aviator goggles, perched on a "
            "futuristic motorcycle in a cyberpunk neon-lit city, raining, "
            "photorealistic, 8k, dramatic cinematic lighting"),
    model="sana-1.5-1.6b-1024",
    steps=20, seed=42, output_dir=OUT,
)

# 2. Sprint comparison — same seed
sana_clear_cache()
sana_sprint_generate(
    prompt=("a majestic rubber duck wearing aviator goggles, perched on a "
            "futuristic motorcycle in a cyberpunk neon-lit city, raining"),
    model="sana-sprint-1.6b-1024",
    steps=2, seed=42, output_dir=OUT,
)

# 3. 5-second video
sana_clear_cache()
sana_video_generate(
    prompt=("a serene sunrise over misty mountain peaks, warm golden light "
            "slowly illuminating ancient pine trees"),
    model="sana-video-2b-480",
    steps=15, frames=121, seed=7,
    output_dir=OUT,
)
