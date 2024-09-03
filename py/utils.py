import contextlib

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
    noise,
    factor=1.0,
    *,
    normalized=True,
    normalize_dims=(-3, -2, -1),
):
    if not normalized or noise.numel() == 0:
        return noise * factor if factor != 1 else noise
    noise = noise / noise.std(dim=normalize_dims, keepdim=True)
    return noise.sub_(noise.mean(dim=normalize_dims, keepdim=True)).mul_(factor)


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
