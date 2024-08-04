import torch
import torch.nn.functional as F

from comfy.utils import bislerp

from .external import MODULES as EXT


# The following is modified to work with latent images of ~0 mean from https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/tree/main.
def contrast_adaptive_sharpening(x, amount=0.8, *, epsilon=1e-06):
    """
    Performs a contrast adaptive sharpening on the batch of images x.
    The algorithm is directly implemented from FidelityFX's source code,
    that can be found here
    https://github.com/GPUOpen-Effects/FidelityFX-CAS/blob/master/ffx-cas/ffx_cas.h

    Parameters
    ----------
    x : Tensor
        Image or stack of images, of shape [batch, channels, ny, nx].
        Batch and channel dimensions can be ommited.
    amount : int [0, 1]
        Amount of sharpening to do, 0 being minimum and 1 maximum

    Returns
    -------
    Tensor
        Processed stack of images.

    """

    def on_abs_stacked(tensor_list, f, *args, **kwargs):
        return f(torch.abs(torch.stack(tensor_list)), *args, **kwargs)[0]

    x_padded = F.pad(x, pad=(1, 1, 1, 1))
    x_padded = torch.complex(x_padded, torch.zeros_like(x_padded))
    # each side gets padded with 1 pixel
    # padding = same by default

    # Extracting the 3x3 neighborhood around each pixel
    # a b c
    # d e f
    # g h i

    a = x_padded[..., :-2, :-2]
    b = x_padded[..., :-2, 1:-1]
    c = x_padded[..., :-2, 2:]
    d = x_padded[..., 1:-1, :-2]
    e = x_padded[..., 1:-1, 1:-1]
    f = x_padded[..., 1:-1, 2:]
    g = x_padded[..., 2:, :-2]
    h = x_padded[..., 2:, 1:-1]
    i = x_padded[..., 2:, 2:]

    # Computing contrast
    cross = (b, d, e, f, h)
    mn = on_abs_stacked(cross, torch.min, axis=0)
    mx = on_abs_stacked(cross, torch.max, axis=0)

    diag = (a, c, g, i)
    mn2 = on_abs_stacked(diag, torch.min, axis=0)
    mx2 = on_abs_stacked(diag, torch.max, axis=0)

    mx = mx + mx2
    mn = mn + mn2

    # Computing local weight
    inv_mx = torch.reciprocal(mx + epsilon)  # 1/mx

    amp = inv_mx * mn

    # scaling
    amp = torch.sqrt(amp)

    w = -amp * (amount * (1 / 5 - 1 / 8) + 1 / 8)
    # w scales from 0 when amp=0 to K for amp=1
    # K scales from -1/5 when amount=1 to -1/8 for amount=0

    # The local conv filter is
    # 0 w 0
    # w 1 w
    # 0 w 0
    div = torch.reciprocal(1 + 4 * w)
    output = ((b + d + f + h) * w + e) * div

    return output.real.clamp(x.min(), x.max())


if "bleh" in EXT:
    scale_samples = EXT["bleh"].latent_utils.scale_samples
    UPSCALE_METHODS = EXT["bleh"].latent_utils.UPSCALE_METHODS
else:
    UPSCALE_METHODS = ("bicubic", "bislerp", "bilinear", "nearest-exact", "area")

    def scale_samples(
        samples,
        width,
        height,
        mode="bicubic",
        sigma=None,  # noqa: ARG001
    ):
        if mode == "bislerp":
            return bislerp(samples, width, height)
        return F.interpolate(samples, size=(height, width), mode=mode)


if "sonar" in EXT:
    get_noise_sampler = EXT["sonar"].noise.get_noise_sampler
else:

    def get_noise_sampler(noise_type, x, *_args: list, **_kwargs: dict):
        if noise_type != "gaussian":
            raise ValueError("Only gaussian noise supported")
        return lambda _s, _sn: torch.randn_like(x)
