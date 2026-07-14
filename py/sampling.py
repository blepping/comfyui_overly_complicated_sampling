import torch
from tqdm.auto import trange

from .filtering import FILTER_HANDLERS, FilterRefs
from .model import OCSModel
from .noise import NoiseSamplerCache
from .restart import Restart
from .substep_merging import MERGE_SUBSTEPS_CLASSES
from .substep_sampling import SamplerState


def find_merge_sampler(merge_samplers, ss) -> object | None:
    handlers = None
    for merge_sampler in merge_samplers:
        if merge_sampler.when is not None and handlers is None:
            handlers = FILTER_HANDLERS.clone(constants=ss.refs)
            # handlers = FILTER_HANDLERS.clone_with_refs(ss.refs)
        if merge_sampler.check_match(handlers, ss=ss):
            return merge_sampler
    return None


def composable_sampler(
    model,
    x,
    sigmas,
    *,
    s_noise=1.0,
    eta=1.0,
    overly_complicated_options,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    **kwargs,
):
    copts = overly_complicated_options.copy()
    if extra_args is None:
        extra_args = {}
    if noise_sampler is None:

        def noise_sampler(_s, _sn):
            return torch.randn_like(x)

    restart_params = copts.get("restart", {})
    restart_enabled = restart_params.get("enabled", True)
    restart_custom_noise = copts.get("restart_custom_noise")
    if isinstance(restart_custom_noise, str):
        restart_custom_noise = copts.get(f"restart_custom_noise_{restart_custom_noise}")

    ss = SamplerState(
        OCSModel(
            model,
            x,
            x.new_ones((x.shape[0],)),
            extra_args,
            **copts.get("model", {}),
        ),
        sigmas,
        0,
        extra_args,
        noise_sampler=noise_sampler,
        callback=callback,
        eta=eta if eta != 1.0 else copts.get("eta", 1.0),
        s_noise=s_noise if s_noise != 1.0 else copts.get("s_noise", 1.0),
        reta=copts.get("reta", 1.0),
        disable_status=disable,
    )

    restart = Restart(
        s_noise=restart_params.get("s_noise", 1.0),
        custom_noise=restart_custom_noise,
        immiscible=restart_params.get("immiscible", False),
        normalized=restart_params.get("normalized", True),
        normalize_dims=restart_params.get("normalize_dims"),
        is_flow=ss.model.is_rectified_flow,
    )

    groups = copts["_groups"]
    merge_samplers = tuple(
        MERGE_SUBSTEPS_CLASSES[g.merge_method](ss, g) for g in groups.items
    )
    nsc = NoiseSamplerCache(
        x,
        extra_args.get("seed", 42),
        sigmas[sigmas > 0].min(),
        sigmas.max(),
        **copts.get("noise", {}),
    )
    ss.noise = nsc
    sigma_chunks = (
        tuple(restart.split_sigmas(sigmas)) if restart_enabled else ((None, sigmas),)
    )
    step_count = sum(len(chunk) - 1 for _noise, chunk in sigma_chunks)
    ss.total_steps = step_count
    step = 0
    with trange(step_count, disable=ss.disable_status) as pbar:
        for chunk_idx, (scale_factors, chunk_sigmas) in enumerate(sigma_chunks):
            if step != 0 and scale_factors is not None:
                prev_refs = FilterRefs(
                    {f"pre_restart_{k}": v for k, v in ss.refs.items()}
                )
            ss.sigmas = chunk_sigmas
            ss.update(0, step=step, substep=0)
            if step != 0:
                nsc.reset_cache()
                nsc.update_x(x)
                ss.hist.reset()
                for ms in merge_samplers:
                    ms.reset()
            nsc.min_sigma, nsc.max_sigma = (
                chunk_sigmas[-1].clone(),
                chunk_sigmas[0].clone(),
            )
            if step != 0 and scale_factors is not None:
                x = restart.add_noise(
                    x,
                    sigma_from=sigma_chunks[chunk_idx - 1][1][-1].item(),
                    sigma_to=chunk_sigmas[0].item(),
                    scale_factors=scale_factors,
                    nsc=nsc,
                    refs=prev_refs | ss.refs,
                    in_place=True,
                )
                del prev_refs
            for idx in range(len(chunk_sigmas) - 1):
                if idx > 0:
                    ss.update(idx, step=step, substep=0)
                    nsc.update_x(x)
                nsc.update_x(x)
                merge_sampler = find_merge_sampler(merge_samplers, ss)
                if merge_sampler is None:
                    raise RuntimeError(f"No matching sampler group for step {step + 1}")
                pbar.set_description(
                    f"{merge_sampler.name}: {ss.sigma.item():.03} -> {ss.sigma_next.item():.03}"
                )
                x = merge_sampler(x)
                if (idx + 1) % nsc.cache_reset_interval == 0:
                    nsc.reset_cache()
                step += 1
                pbar.update(1)
    return x
