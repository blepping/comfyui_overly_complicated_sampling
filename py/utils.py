import torch

from comfy.k_diffusion.sampling import to_d


def scale_noise(
    noise,
    factor=1.0,
    *,
    normalized=True,
    normalize_dims=(-3, -2, -1),
):
    if not normalized or noise.numel() == 0:
        return noise.mul_(factor) if factor != 1 else noise
    mean, std = (
        noise.mean(dim=normalize_dims, keepdim=True),
        noise.std(dim=normalize_dims, keepdim=True),
    )
    return noise.sub_(mean).div_(std).mul_(factor)


def find_first_unsorted(tensor, desc=True):
    if not (len(tensor.shape) and tensor.shape[0]):
        return None
    fun = torch.gt if desc else torch.lt
    first_unsorted = fun(tensor[1:], tensor[:-1]).nonzero().flatten()[:1].add_(1)
    return None if not len(first_unsorted) else first_unsorted.item()


def fallback(val, default, exclude=None):
    return val if val is not exclude else default


# From Gaeros. Thanks!
def extract_pred(x_before, x_after, sigma_before, sigma_after):
    if sigma_after == 0:
        return x_after, torch.zeros_like(x_after)
    alpha = sigma_after / sigma_before
    denoised = (x_after - alpha * x_before) / (1 - alpha)
    return denoised, to_d(x_after, sigma_after, denoised)
