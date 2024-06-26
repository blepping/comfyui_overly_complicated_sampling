import torch
from tqdm.auto import trange


from .substep_samplers import STEP_SAMPLERS
from .substep_sampling import SamplerState, History, ModelCallCache
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

    samplers = []
    substeps = 0
    for sitem in copts["chain"].items:
        custom_noise = sitem.get("custom_noise_opt")
        if custom_noise is None:
            curr_ns = noise_sampler
        else:
            curr_ns = custom_noise.make_noise_sampler(
                x, sigmas[-1], sigmas[0], normalized=True
            )
        ssampler = STEP_SAMPLERS[sitem["step_method"]](noise_sampler=curr_ns, **sitem)
        samplers.append(ssampler)
        # samplers += (ssampler,) * sitem["substeps"]
        substeps += ssampler.substeps
    msitem = copts["merge_sampler"]
    if copts["merge_method"] in ("sample", "sample_uncached"):
        custom_noise = msitem.get("custom_noise_opt")
        if custom_noise is None:
            curr_ns = noise_sampler
        else:
            curr_ns = custom_noise.make_noise_sampler(
                x, sigmas[-1], sigmas[0], normalized=True
            )
        merge_sampler = STEP_SAMPLERS[msitem["step_method"]](
            noise_sampler=curr_ns, **msitem
        )
        pass
    else:
        merge_sampler = None
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
    merge_sampler = MERGE_SUBSTEPS_CLASSES[copts["merge_method"]](
        ss,
        samplers,
        **(copts | {"merge_sampler": merge_sampler}),
    )
    for idx in trange(len(sigmas) - 1, disable=disable):
        print(f"STEP {idx+1}")
        ss.update(idx)
        ss.model.reset_cache()
        x = merge_sampler.step(x)
    return x
