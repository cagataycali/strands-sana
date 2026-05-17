# Schedulers

Swap denoising schedulers at runtime. **10 aliases** mapped to diffusers classes.

```python
from strands_sana import sana_set_scheduler, sana_list_schedulers

print(sana_list_schedulers())

sana_set_scheduler(name="dpm-solver", model="sana-1.6b-1024")
```

## Available schedulers

| Alias | Diffusers class | Notes |
|---|---|---|
| `flow-match-euler` | `FlowMatchEulerDiscreteScheduler` | **Default — best for Sana** |
| `flow-match-heun` | `FlowMatchHeunDiscreteScheduler` | Higher fidelity, 2× cost |
| `dpm-solver` | `DPMSolverMultistepScheduler` | Fast, supports flow_sigmas |
| `dpm-solver-single` | `DPMSolverSinglestepScheduler` | Even faster |
| `dpm-solver-cosine` | `CosineDPMSolverMultistepScheduler` | Cosine schedule |
| `euler` | `EulerDiscreteScheduler` | Lossy on flow models |
| `euler-ancestral` | `EulerAncestralDiscreteScheduler` | Adds noise per step |
| `ddim` | `DDIMScheduler` | Lossy on flow models |
| `deis` | `DEISMultistepScheduler` | Fast |
| `heun` | `HeunDiscreteScheduler` | Lossy on flow models |

## Demo — same prompt, 4 schedulers

<div class="compare-grid" markdown>
<figure markdown>
  ![FME](../assets/tool_scheduler_flow_match.png)
  <figcaption>flow-match-euler<br/>(default)</figcaption>
</figure>
<figure markdown>
  ![DPM](../assets/tool_scheduler_dpm_solver.png)
  <figcaption>dpm-solver<br/>(13× faster)</figcaption>
</figure>
<figure markdown>
  ![Euler](../assets/tool_scheduler_euler.png)
  <figcaption>euler<br/>(lossy)</figcaption>
</figure>
<figure markdown>
  ![DDIM](../assets/tool_scheduler_ddim.png)
  <figcaption>ddim<br/>(lossy)</figcaption>
</figure>
</div>

## Flow-matching compatibility

Sana is a flow-matching model. Its scheduler config has `prediction_type=flow_prediction`.

- ✅ `FlowMatch*` schedulers — native fit
- ✅ `DPMSolver*` schedulers — accept `use_flow_sigmas=True`
- ⚠️ Other schedulers — work but quality degrades (we strip `flow_prediction` automatically and warn)

This was [BUG #9](../bugs-fixed.md) — caught on Thor.

## Speed comparison (Thor, sana-0.6b-1024, 10 steps)

| Scheduler | Time |
|---|---:|
| flow-match-euler | 48s (first run, ~12s warm) |
| **dpm-solver** | **13.3s** |
| euler | 6.2s |
| euler-ancestral | 6.3s |
| ddim | 7.0s |
| deis | 7.1s |
| heun | 7.7s |
