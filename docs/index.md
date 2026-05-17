# strands-sana

NVIDIA Sana text-to-image diffusion for Strands Agents.

## Install

```bash
pip install strands-sana
```

## Use

```python
from strands import Agent
from strands_sana import sana_generate

agent = Agent(tools=[sana_generate])
agent("Generate a cyberpunk cityscape at night")
```

See [API reference](api.md) for more.
