import torch
from tqdm.auto import trange


from .model import ModelCallCache
from .noise import NoiseSamplerCache
from .substep_sampling import SamplerState
from .substep_merging import MERGE_SUBSTEPS_CLASSES
from .utils import Restart


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
    restart = Restart(
        s_noise=restart_params.get("s_noise", 1.0),
        custom_noise=copts.get("restart_custom_noise"),
        immiscible=restart_params.get("immiscible", False),
    )

    ss = SamplerState(
        ModelCallCache(
            model,
            x,
            x.new_ones((x.shape[0],)),
            extra_args,
            **copts.get("model_call_cache", {}),
        ),
        sigmas,
        0,
        extra_args,
        noise_sampler=noise_sampler,
        callback=callback,
        eta=eta if eta != 1.0 else copts["eta"],
        s_noise=s_noise if s_noise != 1.0 else copts["s_noise"],
        reta=copts.get("reta", 1.0),
        disable_status=disable,
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
    sigma_chunks = tuple(restart.split_sigmas(sigmas))
    step_count = sum(len(chunk) - 1 for _noise, chunk in sigma_chunks)
    step = 0
    restart_snoise = copts.get("restart_s_noise", 1.0)
    with trange(step_count, disable=ss.disable_status) as pbar:
        for noise_scale, chunk_sigmas in sigma_chunks:
            ss.sigmas = chunk_sigmas
            ss.update(0, step=step)
            if step != 0:
                hcur = ss.hcur
                nsc.reset_cache()
                ss.hist.reset()
                for ms in merge_samplers:
                    ms.reset()
            nsc.min_sigma, nsc.max_sigma = chunk_sigmas[-1], chunk_sigmas[0]
            if step != 0 and noise_scale != 0:
                restart_ns = restart.get_noise_sampler(nsc)
                x += nsc.scale_noise(
                    restart_ns(
                        refs={
                            "x": x,
                            "denoised": hcur.denoised,
                            "sigma": chunk_sigmas[0],
                            "sigma_next": chunk_sigmas[1],
                        }
                    ),
                    noise_scale * restart_snoise,
                )
                del restart_ns
            for idx in range(len(chunk_sigmas) - 1):
                if idx > 0:
                    ss.update(idx, step=step)
                # print(
                #     f"STEP {step + 1:>3}: {ss.sigma.item():.03} -> {ss.sigma_next.item():.03} || up={ss.sigma_up.item():.03}, down={ss.sigma_down.item():.03}"
                # )
                ss.model.reset_cache()
                nsc.update_x(x)
                ms_idx = groups.find_match(ss.sigma, step, step_count)
                if ms_idx is None:
                    raise RuntimeError(f"No matching sampler group for step {step + 1}")
                merge_sampler = merge_samplers[ms_idx]
                pbar.set_description(
                    f"{merge_sampler.name}: {ss.sigma.item():.03} -> {ss.sigma_next.item():.03}"
                )
                x = merge_sampler.step(x)
                if (idx + 1) % nsc.cache_reset_interval == 0:
                    nsc.reset_cache()
                step += 1
                pbar.update(1)
    return x
