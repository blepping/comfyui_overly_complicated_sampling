import numpy as np
import torch
import torch.nn.functional as F

import folder_paths
import latent_preview

from comfy.taesd.taesd import TAESD
from comfy.utils import bislerp
from comfy import latent_formats

from .external import MODULES as EXT

EXT_NNLATENTUPSCALE = None


def init_integrations():
    global get_noise_sampler, EXT_NNLATENTUPSCALE

    ext_sonar = EXT.sonar
    if ext_sonar is not None:
        get_noise_sampler = ext_sonar.noise.get_noise_sampler
    EXT_NNLATENTUPSCALE = EXT.nnlatentupscale


EXT.register_init_handler(init_integrations)


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


def get_noise_sampler(noise_type, x, *_args: list, **_kwargs: dict):  # noqa: F811
    if noise_type != "gaussian":
        raise ValueError("Only gaussian noise supported")
    return lambda _s, _sn: torch.randn_like(x)


def scale_nnlatentupscale(mode, latent, scale=2.0, *, scale_factor=0.13025):
    if EXT_NNLATENTUPSCALE is None:
        raise RuntimeError("nnlatentupscale integration not available")
    mode = {"sdxl": "SDXL", "sd1": "SD 1.x"}.get(mode)
    if mode is None:
        raise ValueError("Bad mode")
    node = EXT_NNLATENTUPSCALE.NNLatentUpscale()
    model = EXT_NNLATENTUPSCALE.latent_resizer.LatentResizer.load_model(
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


# Gaussian blur
def gaussian_blur_2d(img, kernel_size, sigma):
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = torch.nn.functional.pad(img, padding, mode="reflect")
    img = torch.nn.functional.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


# Saliency-adaptive Noise Fusion based on High-fidelity Person-centric Subject-to-Image Synthesis (Wang et al.)
# https://github.com/CodeGoat24/Face-diffuser/blob/edff1a5178ac9984879d9f5e542c1d0f0059ca5f/facediffuser/pipeline.py#L535-L562
def snf_guidance(
    t_guidance: torch.Tensor,
    s_guidance: torch.Tensor,
    t_kernel_size=3,
    t_sigma=1,
    s_kernel_size=3,
    s_sigma=1,
):
    b, c, h, w = shape = t_guidance.shape

    t_softmax, s_softmax = (
        torch.softmax(
            gaussian_blur_2d(torch.abs(t), ks, sig).reshape(b * c, h * w),
            dim=1,
        ).reshape(*shape)
        for t, ks, sig in (
            (t_guidance, t_kernel_size, t_sigma),
            (s_guidance, s_kernel_size, s_sigma),
        )
    )
    guidance_stacked = torch.stack((t_guidance, s_guidance), dim=0)
    argeps = torch.argmax(
        torch.stack((t_softmax, s_softmax), dim=0), dim=0, keepdim=True
    )
    return torch.gather(guidance_stacked, dim=0, index=argeps).squeeze(0)


class OCSLatentFormat:
    def __init__(self, device, latent_format):
        self.rgb_factors = torch.tensor(
            latent_format.latent_rgb_factors, device=device, dtype=torch.float
        ).t()
        # Thanks for Joviax for the help implementing this!
        self.rgb_factors_inv = torch.linalg.pinv(self.rgb_factors)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        self.rgb_factors_bias = (
            None
            if bias is None
            else torch.tensor(bias, device=device, dtype=torch.float)
        )

    def latent_to_rgb(self, latent: torch.Tensor) -> torch.Tensor:
        # NCHW -> NHWC
        return torch.nn.functional.linear(
            latent.movedim(1, -1), self.rgb_factors, bias=self.rgb_factors_bias
        )

    def rgb_to_latent(self, img: torch.Tensor) -> torch.Tensor:
        # NHWC
        if self.rgb_factors_bias is not None:
            img = img - self.rgb_factors_bias
        return torch.nn.functional.linear(img, self.rgb_factors_inv)
