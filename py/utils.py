import math
import torch


def scale_noise(
    noise,
    factor=1.0,
    *,
    normalized=True,
    threshold_std_devs=2.5,
    normalize_dims=(-3, -2, -1),
):
    if not normalized or noise.numel() == 0:
        return noise.mul_(factor) if factor != 1 else noise
    mean, std = (
        noise.mean(dim=normalize_dims, keepdim=True),
        noise.std(dim=normalize_dims, keepdim=True),
    )
    # threshold = threshold_std_devs / math.sqrt(noise.numel())
    # noise[mean.abs() > threshold] -= mean
    # noise[(1.0 - std).abs() > threshold] /= std
    noise -= mean
    noise /= std
    # if abs(mean) > threshold:
    #     noise -= mean
    # if abs(1.0 - std) > threshold:
    #     noise /= std
    return noise.mul_(factor) if factor != 1 else noise


# def scale_noise(noise, factor=1.0, *, normalized=True, threshold_std_devs=2.5):
#     if not normalized or noise.numel() == 0:
#         return noise.mul_(factor) if factor != 1 else noise
#     mean, std = noise.mean().item(), noise.std().item()
#     threshold = threshold_std_devs / math.sqrt(noise.numel())
#     if abs(mean) > threshold:
#         noise -= mean
#     if abs(1.0 - std) > threshold:
#         noise /= std
#     return noise.mul_(factor) if factor != 1 else noise


def find_first_unsorted(tensor, desc=True):
    if not (len(tensor.shape) and tensor.shape[0]):
        return None
    fun = torch.gt if desc else torch.lt
    first_unsorted = fun(tensor[1:], tensor[:-1]).nonzero().flatten()[:1].add_(1)
    return None if not len(first_unsorted) else first_unsorted.item()
