"""Prompt-engineering helpers (P4 #24).

Sana uses a Gemma-2-2B decoder-only LLM as the text encoder. Upstream
provides a 'complex human instruction' template that significantly improves
text-image alignment when prepended to user prompts.

Source: NVlabs/Sana — diffusion/model/builder.py / app/sana_pipeline.py
"""
from __future__ import annotations

# This is the upstream "complex_human_instruction" used in app/app_sana.py.
# diffusers ≥ 0.32 accepts this as a list of strings to prepend.
COMPLEX_HUMAN_INSTRUCTION = [
    "Given a user prompt, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round ball, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]


def enhance_prompt(prompt: str, style: str = "photorealistic") -> str:
    """Lightweight client-side prompt enhancement (no LLM call).

    Adds a style hint suffix. For full upstream-style enhancement, pass
    `complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION` to the pipeline
    so Gemma-2 expands the prompt server-side.
    """
    style_map = {
        "photorealistic": "photorealistic, high detail, sharp focus, 4k",
        "anime":          "anime style, cel shaded, vibrant colors",
        "oil-painting":   "oil painting, thick brush strokes, masterpiece",
        "cinematic":      "cinematic lighting, dramatic, film grain, 35mm",
        "minimalist":     "minimalist, clean, simple composition",
    }
    suffix = style_map.get(style, style)
    return f"{prompt}, {suffix}" if suffix else prompt
