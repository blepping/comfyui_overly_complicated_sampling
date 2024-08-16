import numpy as np
import torch
import torch.nn.functional as F

import folder_paths
import latent_preview

from comfy.taesd.taesd import TAESD
from comfy.utils import bislerp
from comfy import latent_formats

from .external import MODULES as EXT


def normalize_to_scale(latent, target_min, target_max, *, dim=(-3, -2, -1)):
    min_val, max_val = (
        latent.amin(dim=dim, keepdim=True),
        latent.amax(dim=dim, keepdim=True),
    )
    normalized = (latent - min_val).div_(max_val - min_val)
    return (
        normalized.mul_(target_max - target_min)
        .add_(target_min)
        .clamp_(target_min, target_max)
    )


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


class ImageBatch(tuple):
    __slots__ = ()


class OCSTAESD:
    latent_formats = {
        "sd15": latent_formats.SD15(),
        "sdxl": latent_formats.SDXL(),
    }

    @classmethod
    def get_decoder_name(cls, fmt):
        return cls.latent_formats[fmt].taesd_decoder_name

    @classmethod
    def get_encoder_name(cls, fmt):
        result = cls.get_decoder_name(fmt)
        if not result.endswith("_decoder"):
            raise RuntimeError(
                f"Could not determine TAESD encoder name from {result!r}"
            )
        return f"{result[:-7]}encoder"

    @classmethod
    def get_taesd_path(cls, name):
        taesd_path = next(
            (
                fn
                for fn in folder_paths.get_filename_list("vae_approx")
                if fn.startswith(name)
            ),
            "",
        )
        if taesd_path == "":
            raise RuntimeError(f"Could not get TAESD path for {name!r}")
        return folder_paths.get_full_path("vae_approx", taesd_path)

    @classmethod
    def decode(cls, fmt, latent):
        latent_format = cls.latent_formats[fmt]
        # rv = latent_format.process_out(1.0)
        filename = cls.get_taesd_path(cls.get_decoder_name(fmt))
        model = TAESD(
            decoder_path=filename, latent_channels=latent_format.latent_channels
        ).to(latent.device)
        # print("DEC INPUT ORIG", latent.min(), latent.max())
        # if torch.any(latent.max() > rv) or torch.any(latent.min() < -rv):
        #     sv = latent.new((-rv, rv))
        #     latent = normalize_to_scale(
        #         latent,
        #         latent.amin(dim=(-3, -2, -1), keepdim=True).maximum(sv[0]),
        #         latent.amax(dim=(-3, -2, -1), keepdim=True).minimum(sv[1]),
        #         dim=(-3, -2, -1),
        #     )
        # print("DEC INPUT", latent.min(), latent.max())
        # result = model.decode(latent.clamp(-rv, rv)).movedim(1, 3)
        result = model.decode(latent).movedim(1, 3)
        # print("DEC RESULT", result.shape, result.isnan().any().item())
        return ImageBatch(
            latent_preview.preview_to_image(result[batch_idx])
            for batch_idx in range(result.shape[0])
        )

    @staticmethod
    def img_to_encoder_input(imgbatch):
        return torch.stack(
            tuple(
                torch.tensor(np.array(img), dtype=torch.float32)
                .div_(127)
                .sub_(1.0)
                .clamp_(-1, 1)
                for img in imgbatch
            ),
            dim=0,
        ).movedim(-1, 1)

    @classmethod
    def encode(cls, fmt, imgbatch, latent, *, normalize_output=False):
        latent_format = cls.latent_formats[fmt]
        rv = latent_format.process_out(1.0)
        filename = cls.get_taesd_path(cls.get_encoder_name(fmt))
        model = TAESD(
            encoder_path=filename, latent_channels=latent_format.latent_channels
        ).to(device=latent.device)
        result = model.encode(cls.img_to_encoder_input(imgbatch).to(latent.device))
        # print(
        #     "ENC RESULT ORIG",
        #     result.min(),
        #     result.max(),
        # )
        # if torch.any(result.max() > rv) or torch.any(result.min() < -rv):
        #     sv = result.new((-rv, rv))
        #     result = normalize_to_scale(
        #         result,
        #         result.amin(dim=(-3, -2, -1), keepdim=True).maximum(sv[0]),
        #         result.amax(dim=(-3, -2, -1), keepdim=True).minimum(sv[1]),
        #         dim=(-3, -2, -1),
        #     )
        # print(
        #     "ENC RESULT",
        #     result.shape,
        #     result.isnan().any().item(),
        #     result.min(),
        #     result.max(),
        # )
        return result.to(latent.dtype).clamp(-rv, rv)


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


if "nnlatentupscale" in EXT:

    def scale_nnlatentupscale(
        mode,
        latent,
        scale=2.0,
        *,
        scale_factor=0.13025,
        __nlu_module=EXT["nnlatentupscale"],
    ):
        module = __nlu_module
        mode = {"sdxl": "SDXL", "sd1": "SD 1.x"}.get(mode)
        if mode is None:
            raise ValueError("Bad mode")
        node = module.NNLatentUpscale()
        model = module.latent_resizer.LatentResizer.load_model(
            node.weight_path[mode], latent.device, latent.dtype
        ).to(device=latent.device)
        result = (
            model(scale_factor * latent, scale=scale).to(
                dtype=latent.dtype, device=latent.device
            )
            / scale_factor
        )
        del model
        return result
