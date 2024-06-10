import torch
from tqdm.auto import trange


from .substep_sampling import SamplerState, History, ModelCallCache, NoiseSamplerCache
from .substep_merging import MERGE_SUBSTEPS_CLASSES


def composable_sampler(
    model,
    x,
    sigmas,
    *,
    s_noise=1.0,
    eta=1.0,
    composable_sampler_options,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    **kwargs,
):
    copts = composable_sampler_options.copy()
    if extra_args is None:
        extra_args = {}
    if noise_sampler is None:

        def noise_sampler(_s, _sn):
            return torch.randn_like(x)

    ss = SamplerState(
        ModelCallCache(
            model,
            x,
            x.new_ones((x.shape[0],)),
            extra_args,
            size=copts.get("model_call_cache", 0),
            max_use=copts.get("model_call_cache_max_use", 1000000),
            threshold=copts.get("model_call_cache_threshold", 0),
        ),
        sigmas,
        0,
        History(x, 3),
        History(x, 2),
        extra_args,
        noise_sampler=noise_sampler,
        callback=callback,
        eta=eta if eta != 1.0 else copts["eta"],
        s_noise=s_noise if s_noise != 1.0 else copts["s_noise"],
        reta=copts.get("reta", 1.0),
    )
    groups = copts["_groups"]
    merge_samplers = tuple(
        MERGE_SUBSTEPS_CLASSES[g.merge_method](ss, g.items, **g.options)
        for g in groups.items
    )
    nsc = NoiseSamplerCache(
        x,
        extra_args.get("seed", 42),
        sigmas[-1],
        sigmas[0],
        **copts.get("noise", {}),
    )
    ss.noise = nsc
    step_count = len(sigmas) - 1
    for idx in trange(step_count, disable=disable):
        print(f"STEP {idx + 1}")
        ss.update(idx)
        ss.model.reset_cache()
        nsc.update_x(x)
        ms_idx = groups.find_match(ss.sigma, idx, step_count)
        if ms_idx is None:
            raise RuntimeError(f"No matching sampler group for step {idx + 1}")
        merge_sampler = merge_samplers[ms_idx]
        x = merge_sampler.step(x)
        if (idx + 1) % nsc.cache_reset_interval == 0:
            nsc.reset_cache()
    return x
