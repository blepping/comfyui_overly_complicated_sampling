import torch
from tqdm.auto import trange


from .substep_sampling import SamplerState, History, ModelCallCache, NoiseSamplerCache
from .substep_merging import MERGE_SUBSTEPS_CLASSES


def restart_get_segment(sigmas: torch.Tensor) -> torch.Tensor:
    last_sigma = sigmas[0]
    for idx in range(1, len(sigmas)):
        sigma = sigmas[idx]
        if sigma > last_sigma:
            return sigmas[:idx]
        last_sigma = sigma
    return sigmas


def restart_split_sigmas(sigmas):
    prev_seg = None
    while len(sigmas) > 1:
        seg = restart_get_segment(sigmas)
        sigmas = sigmas[len(seg) :]
        if prev_seg is not None and seg[0] > prev_seg[-1]:
            s_min, s_max = prev_seg[-1], seg[0]
            noise_scale = ((s_max**2 - s_min**2) ** 0.5).item()
        else:
            noise_scale = 0.0
        prev_seg = seg
        yield (noise_scale, seg)


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

    restart_custom_noise = copts.get("restart_custom_noise")

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
    sigma_chunks = tuple(restart_split_sigmas(sigmas))
    step_count = sum(len(chunk) - 1 for _noise, chunk in sigma_chunks)
    step = 0
    with trange(step_count, disable=disable) as pbar:
        for noise_scale, chunk_sigmas in sigma_chunks:
            ss.sigmas = chunk_sigmas
            nsc.reset_cache()
            ss.dhist.reset()
            ss.xhist.reset()
            nsc.min_sigma, nsc.max_sigma = chunk_sigmas[-1], chunk_sigmas[0]
            if noise_scale != 0:
                restart_ns = nsc.make_caching_noise_sampler(
                    restart_custom_noise, 1, nsc.max_sigma, nsc.min_sigma
                )
                x += nsc.scale_noise(restart_ns(), noise_scale)
                del restart_ns
            for idx in range(len(chunk_sigmas) - 1):
                ss.update(idx, step=step)
                print(
                    f"STEP {step + 1}: {ss.sigma.item():.3} -> {ss.sigma_next.item():.3}"
                )
                ss.model.reset_cache()
                nsc.update_x(x)
                ms_idx = groups.find_match(ss.sigma, step, step_count)
                if ms_idx is None:
                    raise RuntimeError(f"No matching sampler group for step {step + 1}")
                merge_sampler = merge_samplers[ms_idx]
                x = merge_sampler.step(x)
                if (idx + 1) % nsc.cache_reset_interval == 0:
                    nsc.reset_cache()
                step += 1
                pbar.update(1)
    return x
