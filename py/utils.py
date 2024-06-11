import torch


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
