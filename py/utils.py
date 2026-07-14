from __future__ import annotations

import contextlib
import math
from functools import partial

import torch
from comfy.k_diffusion.sampling import to_d

# def scale_noise_(
#     noise,
#     factor=1.0,
#     *,
#     normalized=True,
#     normalize_dims=(-3, -2, -1),
# ):
#     if not normalized or noise.numel() == 0:
#         return noise.mul_(factor) if factor != 1 else noise
#     mean, std = (
#         noise.mean(dim=normalize_dims, keepdim=True),
#         noise.std(dim=normalize_dims, keepdim=True),
#     )
#     return latent.normalize_to_scale(
#         noise.sub_(mean).div_(std).clamp(-1, 1), -1.0, 1.0, dim=normalize_dims
#     ).mul_(factor)


# def scale_noise(
#     noise,
#     factor=1.0,
#     *,
#     normalized=True,
#     normalize_dims=(-3, -2, -1),
# ):
#     if not normalized or noise.numel() == 0:
#         return noise * factor if factor != 1 else noise
#     mean, std = (
#         noise.mean(dim=normalize_dims, keepdim=True),
#         noise.std(dim=normalize_dims, keepdim=True),
#     )
#     return (noise - mean).div_(std).mul_(factor)


def scale_noise(
    noise: torch.Tensor,
    factor: float = 1.0,
    *,
    normalized: bool = True,
    normalize_dims: tuple[int, ...] = (-3, -2, -1),
    eps: float = 1e-08,
) -> torch.Tensor:
    if not normalized or noise.numel() == 0:
        return noise * factor if factor != 1 else noise
    std = noise.std(dim=normalize_dims, keepdim=True)
    noise = noise / torch.where(std != 0.0, std, eps)
    noise -= noise.mean(dim=normalize_dims, keepdim=True)
    return noise if factor == 1.0 else noise.mul_(factor)


def range_wrap(
    x: torch.Tensor,
    min_val: float | torch.Tensor,
    max_val: float | torch.Tensor,
) -> torch.Tensor:
    return min_val + (x - min_val).remainder_(max_val - min_val)


def _quantile_norm_scaledown(
    noise: torch.Tensor,
    nq: torch.Tensor,
    *,
    dim,
    **_kwargs: dict,
) -> torch.Tensor:
    noiseabs = noise.abs()
    mv = noiseabs.max(dim=dim, keepdim=True).values.clamp(min=1e-06)
    return (
        noise
        if mv.sum().item() == 0
        else torch.where(noiseabs > nq, noise * (nq / mv), noise)
    )


def _quantile_norm_wave(
    noise: torch.Tensor,
    nq: torch.Tensor,
    *,
    preserve_sign: bool = False,
    wave_function=torch.sin,
    pi_factor: float = 0.5,
    wrong_mode: bool = False,
    **_kwargs: dict,
) -> torch.Tensor:
    if wrong_mode:
        multiplier = 1.0 / ((math.pi * pi_factor) / nq)
    else:
        multiplier = 1.0 / (nq / (math.pi * pi_factor))
    pos_mask = noise >= 0
    neg_mask = ~pos_mask
    result = torch.zeros_like(noise)
    result[pos_mask] = wave_function(noise.mul(multiplier))[pos_mask]
    result[neg_mask] = wave_function(noise.mul(multiplier))[neg_mask]
    result *= nq
    return result.copysign(noise) if preserve_sign else result


def _quantile_norm_mode(
    noise: torch.Tensor,
    nq: torch.Tensor,
    *,
    dim: int | None,
    decimals=1,
    **_kwargs: dict,
) -> torch.Tensor:
    return torch.where(
        noise.abs() > nq,
        noise.round(decimals=decimals).mode(dim=dim, keepdim=True).values,
        noise,
    )


def _quantile_norm_replace(
    noise: torch.Tensor,
    nq: torch.Tensor,
    *,
    keep_sign: bool = False,
    avoid_sign: bool = False,
    count: int = 1,
    count_flipping: bool = False,
    **_kwargs: dict,
) -> torch.Tensor:
    mask = noise.abs() <= nq
    candidates = noise[mask].flatten()
    n_candidates = candidates.numel()
    idxs = torch.arange(noise.numel()) % n_candidates
    cresult = candidates[idxs]
    if count < 2:
        candidates = cresult
    else:
        multiplier = 1.0 / count
        cresult = cresult * multiplier  # noqa: PLR6104
        for i in range(1, count):
            cresult += (
                candidates[
                    torch.roll(
                        idxs,
                        i if not count_flipping or (i % 2) == 0 else -i,
                        dims=(-1,),
                    )
                ]
                * multiplier
            )
    candidates = cresult.reshape(noise.shape)
    if keep_sign or avoid_sign:
        candidates = candidates.copysign_(noise.neg() if avoid_sign else noise)
    return torch.where(mask, noise, candidates)


quantile_handlers = {
    "clamp": lambda noise, nq, **_kwargs: noise.clamp(-nq, nq),
    "scale_down": _quantile_norm_scaledown,
    "tanh": lambda noise, nq, **_kwargs: noise.tanh().mul_(nq.abs()),
    "tanh_outliers": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        noise.tanh().mul_(nq.abs()),
        noise,
    ),
    "sigmoid_keepsign": lambda noise, nq, **_kwargs: (
        noise.sigmoid().mul_(nq.abs()).copysign(noise)
    ),
    "sigmoid": lambda noise, nq, **_kwargs: (
        noise.sigmoid().mul_(nq.abs() * 2).sub_(nq.abs())
    ),
    "sigmoid_outliers": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        noise.sigmoid().mul_(nq.abs()).copysign(noise),
        noise,
    ),
    "sin": partial(_quantile_norm_wave, wave_function=torch.sin),
    "sin_wholepi": partial(
        _quantile_norm_wave,
        wave_function=torch.sin,
        pi_factor=1.0,
    ),
    "sin_keepsign": partial(
        _quantile_norm_wave,
        wave_function=torch.sin,
        preserve_sign=True,
    ),
    "sin_wrong": partial(_quantile_norm_wave, wave_function=torch.sin, wrong_mode=True),
    "sin_wrong_wholepi": partial(
        _quantile_norm_wave,
        wave_function=torch.sin,
        pi_factor=1.0,
        wrong_mode=True,
    ),
    "sin_wrong_keepsign": partial(
        _quantile_norm_wave,
        wave_function=torch.sin,
        preserve_sign=True,
        wrong_mode=True,
    ),
    "cos": partial(_quantile_norm_wave, wave_function=torch.cos),
    "cos_wholepi": partial(
        _quantile_norm_wave,
        wave_function=torch.cos,
        pi_factor=1.0,
    ),
    "cos_keepsign": partial(
        _quantile_norm_wave,
        wave_function=torch.cos,
        preserve_sign=True,
    ),
    "cos_wrong": partial(_quantile_norm_wave, wave_function=torch.cos, wrong_mode=True),
    "cos_wrong_wholepi": partial(
        _quantile_norm_wave,
        wave_function=torch.cos,
        pi_factor=1.0,
        wrong_mode=True,
    ),
    "cos_wrong_keepsign": partial(
        _quantile_norm_wave,
        wave_function=torch.cos,
        preserve_sign=True,
        wrong_mode=True,
    ),
    "atan": lambda noise, nq, **_kwargs: noise.atan().mul_(nq.abs() / (math.pi / 2)),
    "tenth": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        noise * 0.1,
        noise,
    ),
    "half": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        noise * 0.5,
        noise,
    ),
    "zero": lambda noise, nq, **_kwargs: torch.where(noise.abs() > nq, 0, noise),
    "reverse_zero": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() >= nq,
        noise,
        0,
    ),
    "mean": lambda noise, nq, *, dim, **_kwargs: torch.where(
        noise.abs() > nq,
        noise.mean(dim=dim, keepdim=True),
        noise,
    ),
    "median": lambda noise, nq, *, dim, **_kwargs: torch.where(
        noise.abs() > nq,
        noise.median(dim=dim, keepdim=True).values,
        noise,
    ),
    "mode_1dec": partial(_quantile_norm_mode, decimals=1),
    "mode_2dec": partial(_quantile_norm_mode, decimals=2),
    "replace": _quantile_norm_replace,
    "replace_keepsign": partial(_quantile_norm_replace, keep_sign=True),
    "replace_avoidsign": partial(_quantile_norm_replace, avoid_sign=True),
    "replace_2pt": partial(_quantile_norm_replace, count=2),
    "replace_3pt": partial(_quantile_norm_replace, count=3),
    "replace_2pt_flip": partial(_quantile_norm_replace, count=2, count_flipping=True),
    "replace_3pt_flip": partial(_quantile_norm_replace, count=3, count_flipping=True),
    "replace_2pt_keepsign": partial(
        _quantile_norm_replace,
        count=2,
        keep_sign=True,
    ),
    "replace_3pt_keepsign": partial(
        _quantile_norm_replace,
        count=3,
        keep_sign=True,
    ),
    "replace_2pt_flip_keepsign": partial(
        _quantile_norm_replace,
        count=2,
        count_flipping=True,
        keep_sign=True,
    ),
    "replace_3pt_flip_keepsign": partial(
        _quantile_norm_replace,
        count=3,
        count_flipping=True,
        keep_sign=True,
    ),
    "replace_2pt_avoidsign": partial(
        _quantile_norm_replace,
        count=2,
        avoid_sign=True,
    ),
    "replace_3pt_avoidsign": partial(
        _quantile_norm_replace,
        count=3,
        avoid_sign=True,
    ),
    "replace_2pt_flip_avoidsign": partial(
        _quantile_norm_replace,
        count=2,
        count_flipping=True,
        avoid_sign=True,
    ),
    "replace_3pt_flip_avoidsign": partial(
        _quantile_norm_replace,
        count=3,
        count_flipping=True,
        avoid_sign=True,
    ),
    "wrap": lambda noise, nq, **_kwargs: range_wrap(noise, -nq, nq),
    "wrap_keepsign": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        range_wrap(noise, -nq, nq).copysign_(noise),
        noise,
    ),
    "wrap_avoidsign": lambda noise, nq, **_kwargs: torch.where(
        noise.abs() > nq,
        range_wrap(noise, -nq, nq).copysign_(noise.neg()),
        noise,
    ),
}


# Initial version based on Studentt distribution normalizatino from https://github.com/Clybius/ComfyUI-Extra-Samplers/
def quantile_normalize(
    noise: torch.Tensor,
    *,
    quantile: float | tuple | list = 0.75,
    dim: int | None = 1,
    flatten: bool = True,
    nq_fac: float = 1.0,
    pow_fac: float = 0.5,
    strategy: str = "clamp",
    strategy_handler=None,
    eps=1e-08,
) -> torch.Tensor:
    if noise.numel() == 0:
        return noise
    if isinstance(quantile, (tuple, list)):
        for q in quantile:
            noise = quantile_normalize(
                noise=noise,
                quantile=q,
                dim=dim,
                flatten=flatten,
                nq_fac=nq_fac,
                pow_fac=pow_fac,
                strategy=strategy,
                strategy_handler=strategy_handler,
            )
        return noise
    if quantile is None or quantile >= 1 or quantile <= -1:
        return noise
    centered = quantile < 0
    absquantile = abs(quantile)
    orig_shape = noise.shape
    if noise.ndim > 1 and flatten:
        flatnoise = noise.flatten(start_dim=dim)
    else:
        flatten = False
        flatnoise = noise
    handler = (
        quantile_handlers.get(strategy)
        if strategy_handler is None
        else strategy_handler
    )
    if handler is None:
        raise ValueError("Unknown strategy")
    if not centered:
        nq = torch.quantile(
            flatnoise.abs(),
            quantile,
            dim=-1 if flatten else dim,
            keepdim=True,
        )
        nq = nq.mul_(nq_fac).add_(eps)
        # print(f"\nNQ: {nq}")
        noise = handler(
            flatnoise,
            nq,
            orig_noise=noise,
            dim=dim,
            flatten=flatten,
        )
    else:
        absnoise = flatnoise.abs()
        maxabs = absnoise.amax(dim=-1 if flatten else dim, keepdim=True)
        proxy = flatnoise.sign().mul_(maxabs - absnoise)
        nq_proxy = torch.quantile(
            proxy.abs(),
            absquantile,
            dim=-1 if flatten else dim,
            keepdim=True,
        )
        nq_proxy = nq_proxy.mul_(nq_fac).add_(eps)
        # print(f"\nNQ proxy: {nq_proxy}")
        out_proxy = handler(
            proxy,
            nq_proxy,
            orig_noise=noise,
            dim=dim,
            flatten=flatten,
        )
        noise = out_proxy.sign().mul_(maxabs - out_proxy.abs())
    if pow_fac not in {0.0, 1.0}:
        noise = noise.abs().pow_(pow_fac).copysign(noise)
    return noise if noise.shape == orig_shape else noise.reshape(orig_shape)


# def scale_noise(
#     noise,
#     factor=1.0,
#     *,
#     normalized=True,
#     normalize_dims=(-3, -2, -1),
# ):
#     if not normalized or noise.numel() == 0:
#         return noise.mul_(factor) if factor != 1 else noise
#     n = (
#         torch.nn.LayerNorm(noise.shape[1:])
#         if normalize_dims == (-3, -2, -1)
#         else torch.nn.InstanceNorm2d(noise.shape[1])
#     ).to(noise)
#     return n(noise) * factor
#     return latent.normalize_to_scale(
#         n(noise).clamp_(-1, 1), -1, 1, dim=normalize_dims
#     ).mul_(factor)


def find_first_unsorted(tensor, desc=True):
    if not (len(tensor.shape) and tensor.shape[0]):
        return None
    fun = torch.gt if desc else torch.lt
    first_unsorted = fun(tensor[1:], tensor[:-1]).nonzero().flatten()[:1].add_(1)
    return None if not len(first_unsorted) else first_unsorted.item()


def fallback(val, default, exclude=None):
    return val if val is not exclude else default


def step_generator(gen, *, get_next, initial=None):
    next_val = initial
    with contextlib.suppress(StopIteration):
        while True:
            result = gen.send(next_val)
            next_val = get_next(result)
            yield result


# From Gaeros. Thanks!
def extract_pred(x_before, x_after, sigma_before, sigma_after):
    if sigma_after == 0:
        return x_after, torch.zeros_like(x_after)
    alpha = sigma_after / sigma_before
    denoised = (x_after - alpha * x_before) / (1 - alpha)
    return denoised, to_d(x_after, sigma_after, denoised)


def resolve_value(keys, obj):
    if not len(keys):
        raise ValueError("Cannot resolve empty key list")
    result = obj

    class Empty:
        pass

    for idx, key in enumerate(keys):
        if not (hasattr(result, "__getattr__") or hasattr(obj, "__getattribute__")):
            raise ValueError(
                f"Cannot access key {key}: value does not support attribute access"
            )
        result = getattr(result, key, Empty)
        if result is Empty:
            raise AttributeError(f"Key {key} from path {'.'.join(keys)} does not exist")


def check_time(time_mode, time_start, time_end, sigma, step, steps):
    step_pct = step / steps if steps != 0 else 0.0
    if time_mode == "step":
        return time_start <= step <= time_end
    if time_mode == "step_pct":
        return time_start <= step_pct <= time_end
    if time_mode == "sigma":
        return time_start >= sigma >= time_end
    raise ValueError("Bad time mode")
