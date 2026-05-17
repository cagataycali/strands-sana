# Quickstart

Install the package, run a script, get a PNG. ~2 minutes once weights are cached.

```bash
pip install strands-sana
```

## As a Strands Agent

```python
from strands import Agent
from strands_sana import sana_generate

agent = Agent(tools=[sana_generate])
result = agent("Generate a cyberpunk duck on a motorcycle, 1024x1024")
```

The agent reads the prompt, picks `sana-1.6b-1024`, downloads weights on first use, runs inference, and returns the saved path.

## Direct call (no agent)

```python
from strands_sana import sana_generate

result = sana_generate(
    prompt="a cyberpunk duck on a motorcycle",
    model="sana-1.6b-1024",
    steps=20,
    seed=42,
    output_dir="./out",
)
print(result["path"])
```

## Pick a faster model

```python
# Sana-Sprint: 1-2 step distilled, ~0.1s per image on H100
from strands_sana import sana_sprint_generate

sana_sprint_generate(
    prompt="a cyberpunk duck on a motorcycle",
    model="sana-sprint-1.6b-1024",
    steps=2,
)
```

## Generate a video

```python
from strands_sana import sana_video_generate

sana_video_generate(
    prompt="a serene sunrise over misty mountains",
    model="sana-video-2b-480",
    steps=15,
    frames=121,  # 5s @ 24fps
    seed=7,
)
```

→ Continue: **[First image](first-image.md)** · **[First video](first-video.md)**
