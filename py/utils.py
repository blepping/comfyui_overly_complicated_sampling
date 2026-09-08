from __future__ import annotations

import contextlib

import torch
from comfy.k_diffusion.sampling import to_d

F = torch.nn.functional


def scale_noise(
    noise: torch.Tensor,
    factor: float = 1.0,
    *,
    normalized: bool = True,
    normalize_dims: tuple[int, ...] | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    if factor == 0:
        return torch.zeros_like(noise)
    if not normalized or noise.numel() == 0:
        return noise * factor if factor != 1 else noise
    if eps is None:
        eps = torch.finfo(noise.dtype).eps * 1.25
    if normalize_dims is None:
        normalize_dims = tuple(
            range(
                max(0, min(1, noise.ndim - 1)),
                noise.ndim,
            )
        )
    std, mean = torch.std_mean(noise, dim=normalize_dims, keepdim=True)
    noise = noise - mean
    if factor != 1:
        std /= factor
    return noise.div_(std.clamp_min_(eps) if factor >= 0 else std.clamp_max_(-eps))


def range_wrap(
    x: torch.Tensor,
    min_val: float | torch.Tensor,
    max_val: float | torch.Tensor,
) -> torch.Tensor:
    return min_val + (x - min_val).remainder_(max_val - min_val)


def softplus_soft_clamp(
    t: torch.Tensor,
    min_val: torch.Tensor | float = 0.0,
    max_val: torch.Tensor | float = 1.0,
    *,
    # We define stiffness as a multiplier (beta) for the softplus function.
    # Higher stiffness = sharper transition.
    stiffness: float = 1.0,
    safe: bool = True,
) -> torch.Tensor:
    if isinstance(min_val, (float, int)):
        min_val = t.new_tensor(min_val)
    if isinstance(max_val, (float, int)):
        max_val = t.new_tensor(max_val)

    if stiffness < 1e-04:
        return t.clamp(min_val, max_val)

    # Calculate how much we are exceeding the Max
    # softplus(beta * x) / beta
    upper_overshoot = F.softplus((t - max_val).mul_(stiffness)).div_(-stiffness)

    # Calculate how much we are falling short of the Min
    lower_undershoot = F.softplus((min_val - t).mul_(stiffness)).div_(stiffness)

    # Apply corrections:
    # Original - (Amount over max) + (Amount under min)
    t = upper_overshoot.add_(t).add_(lower_undershoot)
    if safe:
        t = t.clamp(min_val, max_val)
    return t


def flip_tensor_range(
    x: torch.Tensor,
    *,
    min_neg: torch.Tensor | None = None,
    max_pos: torch.Tensor | None = None,
    return_ranges: bool = False,
    dim: int = -1,
    eps: float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if eps is None:
        eps = torch.finfo(x.dtype).eps * 1.25
    # 1. Use the provided maximum positive values, or calculate them dynamically
    if max_pos is None:
        max_pos = (
            torch.clamp_min(x, 0.0).max(dim=dim, keepdim=True).values.clamp_min_(eps)
        )

    # 2. Use the provided minimum negative values, or calculate them dynamically
    if min_neg is None:
        min_neg = (
            torch.clamp_max(x, 0.0).min(dim=dim, keepdim=True).values.clamp_max_(-eps)
        )

    # 3. Separate positive and negative elements
    is_pos = x >= 0

    # 4. Flip positive side: [0, max_pos] -> [eps, max_pos + eps]
    x_pos = x.clamp_min(eps)
    flipped_pos = (max_pos + eps) - x_pos

    # 5. Flip negative side: [min_neg, 0] -> [min_neg - eps, -eps]
    x_neg = x.clamp_max(-eps)
    flipped_neg = (min_neg - eps) - x_neg

    # 6. Recombine the domains
    result = torch.where(is_pos, flipped_pos, flipped_neg)
    return (result, max_pos, min_neg) if return_ranges else result


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
