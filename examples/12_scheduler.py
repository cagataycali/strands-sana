"""Swap the scheduler on the active Sana pipeline."""
from strands_sana import sana_set_scheduler, sana_list_schedulers, sana_generate


def _call(t, **kw):
    return t.original_func(**kw) if hasattr(t, "original_func") else t(**kw)


if __name__ == "__main__":
    print("Available:", _call(sana_list_schedulers))
    print(_call(sana_set_scheduler, name="dpm-solver", use_flow_sigmas=True))
    print(_call(sana_generate, prompt="a serene koi pond", steps=15))
