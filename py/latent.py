from typing import Any, NamedTuple, Self

import folder_paths
import latent_preview
import numpy as np
import torch
import torch.nn.functional as F
from comfy import latent_formats
from comfy.taesd.taesd import TAESD
from comfy.utils import bislerp

from .external import MODULES as EXT

EXT_NNLATENTUPSCALE = None


def init_integrations(integrations):
    global get_noise_sampler, EXT_NNLATENTUPSCALE

    ext_sonar = integrations.sonar
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


# Improvements by https://github.com/Clybius
# The following is modified to work with latent images of ~0 mean from https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/tree/main.
# The algorithm is directly implemented from FidelityFX's source code that can be found here: https://github.com/GPUOpen-Effects/FidelityFX-CAS/blob/master/ffx-cas/ffx_cas.h.
def contrast_adaptive_sharpening(
    x,
    amount=0.8,
    *,
    normalize=True,
    epsilon=1e-06,
):
    orig_shape = x.shape
    if x.ndim == 5:
        x = x.reshape(orig_shape[0], orig_shape[1] * orig_shape[2], *orig_shape[-2:])
    elif x.ndim != 4:
        raise ValueError(
            "Contrast-adaptive sharpening requires a tensor with 4 or 5 dimensions",
        )

    def on_abs_stacked(tensor_list, f, *args: list, **kwargs: dict):
        return f(torch.abs(torch.stack(tensor_list)), *args, **kwargs)[0]

    if normalize:
        luminance = torch.linalg.vector_norm(x, dim=1, keepdim=True).add_(1e-08)
        x = x / luminance
        orig_mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        x -= orig_mean

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

    output = output.real
    for ob, xb in zip(x, output):
        ob.clamp_(*xb.aminmax())
    if normalize:
        output = output.add_(orig_mean).mul_(luminance)
    return output.reshape(*orig_shape)


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
        filename = cls.get_taesd_path(cls.get_decoder_name(fmt))
        model = TAESD(
            decoder_path=filename, latent_channels=latent_format.latent_channels
        ).to(latent.device)
        result = model.decode(latent).movedim(1, 3)
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
        return result.to(latent.dtype).clamp(-rv, rv)


bleh_scale_samples = None
UPSCALE_METHODS = ("bicubic", "bislerp", "bilinear", "nearest-exact", "area")


def scale_samples(
    samples,
    width,
    height,
    mode="bicubic",
    sigma=None,  # noqa: ARG001
):
    global bleh_scale_samples, UPSCALE_METHODS
    if bleh_scale_samples is None:
        bleh = EXT.get("bleh")
        if bleh is not None:
            bleh_scale_samples = bleh.latent_utils.scale_samples
            UPSCALE_METHODS = bleh.latent_utils.UPSCALE_METHODS
        else:
            bleh_scale_samples = False
    if bleh_scale_samples:
        return bleh_scale_samples(samples, width, height, mode=mode, sigma=sigma)
    if mode == "bislerp":
        return bislerp(samples, width, height)
    return F.interpolate(samples, size=(height, width), mode=mode)


def get_noise_sampler(noise_type, x, *_args: list, **_kwargs: dict):  # noqa: F811
    if noise_type != "gaussian":
        raise ValueError("Only gaussian noise supported unless you have ComfyUI-sonar")
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
        if latent_format.latent_rgb_factors is None:
            self.rgb_factors = None
            return
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
        if self.rgb_factors is None:
            raise ValueError("No RGB factors for latent type!")
        return torch.nn.functional.linear(
            latent.movedim(1, -1), self.rgb_factors, bias=self.rgb_factors_bias
        )

    def rgb_to_latent(self, img: torch.Tensor) -> torch.Tensor:
        # NHWC
        if self.rgb_factors is None:
            raise ValueError("No RGB factors for latent type!")
        if self.rgb_factors_bias is not None:
            img = img - self.rgb_factors_bias
        return torch.nn.functional.linear(img, self.rgb_factors_inv)


def randomized_svd(
    m: torch.Tensor,
    *,
    rank: int | None = None,
    n_iter: int = 6,
    ortho_interval: int = 3,
    oversample: int = 10,
    noise_sampler: Callable | None = None,
    y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c = m.shape[-2], m.shape[-1]
    if rank is None:
        rank = n
    if y is None:
        if noise_sampler is None:
            noise_sampler = torch.randn
        k = min(rank + oversample, n, c)
        y_shape = (*m.shape[:-2], c, k)
        y = noise_sampler(y_shape, device=m.device, dtype=m.dtype)
    elif y.shape == m.shape:
        y = y.mT
    elif y.shape != m.mT.shape:
        raise ValueError("Bad initial y shape")
    y = m @ y

    ortho = False
    for idx in range(n_iter):
        y = m @ (m.mT @ y)
        ortho = ortho_interval > 0 and (idx % ortho_interval) == 0 and n_iter - idx != 2
        if ortho:
            y = torch.linalg.qr(y)[0]

    q = y if ortho else torch.linalg.qr(y)[0]
    u, s, vh = torch.linalg.svd(q.mT @ m, full_matrices=False)
    u = q @ u
    if rank < n:
        return u[..., :rank], s[..., :rank], vh[..., :rank, :]
    return u, s, vh


class DimCorrelationOrder(NamedTuple):
    perm: torch.Tensor
    dim: int
    leave: bool = False

    def reorder(
        self,
        x: torch.Tensor,
        *,
        invert: bool = False,
    ) -> torch.Tensor:
        dim, perm = self.dim, self.perm
        if invert:
            perm = perm.argsort(dim=-1)
        if dim < 0:
            dim = x.ndim + self.dim
        shape = [1] * x.ndim
        if dim != 0:
            shape[0] = x.shape[0]
        shape[dim] = x.shape[dim]
        perm = perm.view(*shape).expand_as(x)
        return x.gather(dim=dim, index=perm)


class DimCorrelationConfig(NamedTuple):
    dim: int = 1
    flip: bool = False
    cross: bool = False
    leave: bool = False
    preserve_first: bool = False
    center_strength: float = 1.0
    center_dim: int = -1
    # None - disabled, otherwise controls whether abs occurs before or after centering.
    abs_before: bool | None = None
    # 0 - disabled, positive value - enabled, negative value - enabled with sign flipped.
    fix_sign: int = 1
    align_to_peak: bool = False
    expansion_factor: int = 1  # NYI
    # Only applies to cross mode. One of: svd, randomized_svd
    decomp_mode: str = "svd"
    low_rank: int = 0
    low_rank_niter: int = 6
    work_dtype: torch.dtype | None = None
    pc: int = 0

    @classmethod
    def build(cls, **kwargs: Any) -> Self:
        wd = kwargs.get("work_dtype")
        if isinstance(wd, str):
            dtype_map = {
                "float64": torch.float64,
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            wd = dtype_map.get(wd)
            if wd is None:
                raise ValueError("Bad dtype")
            kwargs["work_dtype"] = wd
        if kwargs.get("decomp_mode") not in {None, "svd", "randomized_svd"}:
            raise ValueError(
                "Bad decomp mode, must be unset or one of: svd, randomized_svd",
            )
        fs = frozenset(cls._fields)
        kwargs = {k: v for k, v in kwargs.items() if k in fs}
        return cls(**kwargs)

    @classmethod
    def from_str(cls, s: str) -> Self:
        parts = tuple(p.strip() for p in s.split(":", 3))
        plen = len(parts)
        if plen > 2:
            raise ValueError("Dim correlations only support up to two parts.")
        dim = int(parts[0])
        result = cls(dim=dim)
        if plen < 2 or not parts[1]:
            return result
        p1 = parts[1]
        p1len = len(p1)
        offs = 0
        while offs < p1len:
            pflag = p1[offs]
            offs += 1
            if pflag == "f":
                result = result._replace(flip=True)
            elif pflag == "x":
                result = result._replace(cross=True)
            elif pflag == "l":
                result = result._replace(leave=True)
            elif pflag == "u":
                result = result._replace(center_strength=0.0)
            elif pflag == "c":
                result = result._replace(center_dim=-2)
            elif pflag in "aA":
                result = result._replace(abs_before=pflag == "a")
            elif pflag == "s":
                result = result._replace(fix_sign=0)
            elif pflag == "S":
                result = result._replace(fix_sign=-1)
            elif pflag == "p":
                result = result._replace(align_to_peak=True)
            elif pflag == "i":
                result = result._replace(preserve_first=True)
            else:
                offs -= 1
                break
        p1 = p1[offs:].strip()
        pc = int(p1) if p1 else 0
        return result._replace(pc=pc)

    @classmethod
    def _fix_sign_ambiguity(
        cls,
        x: torch.Tensor,
        *,
        ref: torch.Tensor | None = None,
        dim: int = -1,
        in_place: bool = True,
        neg: bool = False,
    ) -> torch.Tensor:
        if ref is not None:
            x = cls._fix_sign_ambiguity(x, dim=dim, in_place=in_place)
        else:
            ref = x
        signs = ref.gather(dim, ref.abs().argmax(dim=dim, keepdim=True)).sign_()
        signs = signs.masked_fill_(signs == 0, 1.0)
        if neg:
            signs = signs.neg_()
        return x.mul_(signs) if in_place else x * signs

    @staticmethod
    def _preserve_first_index(perm: torch.Tensor) -> torch.Tensor:
        c = perm.shape[-1]
        shift = (perm == 0).to(dtype=torch.int64).argmax(dim=-1, keepdim=True)
        shift = torch.arange(c, device=perm.device, dtype=shift.dtype) + shift
        shift %= c
        return perm.gather(dim=-1, index=shift)

    @staticmethod
    def _align_ref(
        *,
        x: torch.Tensor,
        ref: torch.Tensor,
        skip_dim: int,
    ) -> torch.Tensor:
        if skip_dim < 0:
            skip_dim = x.ndim + skip_dim
        reps = tuple(
            None if szr == 0 or szx % szr != 0 else (d, szx // szr)
            for d, (szx, szr) in enumerate(zip(x.shape, ref.shape, strict=True))
            if d != skip_dim and szx != szr
        )
        if not all(reps):
            raise ValueError("Bad shape")
        for d, r in reps:
            ref = ref.repeat_interleave(r, d)
        return ref

    def _get_correlation_order(
        self,
        cov: torch.Tensor,
        *,
        pc_idx: int = 0,
        cross_mode: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if cross_mode:
            if self.decomp_mode == "randomized_svd":
                pcs = randomized_svd(cov, n_iter=self.low_rank_niter, **kwargs)[0]
            elif self.low_rank < 1:
                pcs = torch.linalg.svd(cov, full_matrices=False).U
            else:
                pcs = torch.svd_lowrank(
                    cov,
                    q=self.low_rank,
                    niter=self.low_rank_niter,
                )[0]
            n_pcs = pcs.shape[-1]
            if pc_idx < 0:
                pc_idx = n_pcs + pc_idx
        else:
            pcs = torch.linalg.eigh(cov).eigenvectors
            n_pcs = pcs.shape[-1]
            pc_idx = n_pcs - pc_idx - 1 if pc_idx >= 0 else pc_idx + 1
        pc_idx = max(0, min(n_pcs - 1, pc_idx))
        return pcs[..., pc_idx]

    def _preprocess(
        self,
        x: torch.Tensor,
        *,
        dim: int,
        allow_post_abs: bool = True,
    ) -> torch.Tensor:
        x_flat = (x.unsqueeze(0) if dim == 0 else x.movedim(dim, 1)).flatten(
            start_dim=2
        )
        if self.abs_before is True:
            x_flat = x_flat.abs()
        if self.center_strength == 0:
            return x_flat
        xm = x_flat.mean(dim=self.center_dim, keepdim=True)
        if self.center_strength != 1:
            xm *= self.center_strength
        x_flat = x_flat.sub_(xm) if self.abs_before else x_flat - xm
        if allow_post_abs and self.abs_before is False:
            x_flat = x_flat.abs_()
        return x_flat

    def get_correlation_order(
        self,
        x: torch.Tensor,
        *,
        ref: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> DimCorrelationOrder:
        if self.work_dtype is not None:
            if x.dtype != self.work_dtype:
                x = x.to(dtype=self.work_dtype)
            if ref is not None and ref.dtype != self.work_dtype:
                ref = ref.to(dtype=self.work_dtype)
        dim, pc_idx = self.dim, self.pc
        if dim < 0:
            dim = x.ndim + dim
        allow_post_abs = self.abs_before is not False or not self.align_to_peak
        x_flat = self._preprocess(
            x,
            dim=dim,
            allow_post_abs=ref is not None or allow_post_abs,
        )
        if ref is None or not self.cross:
            # if ref is None:
            y_flat = x_flat
            if not allow_post_abs:
                sign_ref = y_flat.clone()
                y_flat = y_flat.abs_()
            else:
                sign_ref = y_flat if self.align_to_peak else None
        else:
            if ref.shape != x.shape:
                ref = self._align_ref(x=x, ref=ref, skip_dim=dim)
            y_flat = self._preprocess(ref, dim=dim, allow_post_abs=allow_post_abs)
            if not allow_post_abs:
                sign_ref = y_flat.clone()
                y_flat = y_flat.abs_()
            else:
                sign_ref = y_flat if self.align_to_peak else None
        pc = self._get_correlation_order(
            x_flat @ y_flat.mT,
            pc_idx=pc_idx,
            cross_mode=ref is not None,
            **kwargs,
        )
        if self.fix_sign:
            pc = self._fix_sign_ambiguity(
                pc,
                neg=self.fix_sign < 0,
                ref=None
                if sign_ref is None
                else sign_ref.flatten(start_dim=0 if dim == 0 else 1),
            )
        perm = pc.argsort(dim=-1, descending=self.flip)
        if self.preserve_first:
            perm = self._preserve_first_index(perm)
        return DimCorrelationOrder(dim=dim, perm=perm, leave=self.leave)
