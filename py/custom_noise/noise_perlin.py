# Initial revision based on Perlin generation routines from https://github.com/Extraltodeus/noise_latent_perlinpinpin which was based on https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57 which was based on https://github.com/pvigier/perlin-numpy
import math
from typing import Any, Callable, NamedTuple, Sequence

import torch
from comfy import model_management
from tqdm import tqdm

from .. import filtering
from ..latent import normalize_to_scale
from ..noise import scale_noise
from .base import CustomNoiseItemBase, NormalizeNoiseNodeMixin


def smoothstep_function(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


class BlendFunction(NamedTuple):
    name: str = "lerp"
    blend_function: Callable[..., torch.Tensor] = torch.lerp

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.blend_function(*args, **kwargs)


class Perlin(NamedTuple):
    depth: int = 16
    res: tuple[tuple[float, ...], ...] = ((1.0,), (1.0,), (1.0,))
    octaves: int = 2
    persistence: tuple[float, ...] = (1.0,)
    lacunarity: tuple[tuple[float, ...], ...] = ((2,), (2,), (2,))
    initial_amplitude: float = 1.0
    initial_frequency: tuple[float, ...] = (1.0, 1.0, 1.0)
    break_pattern: float = 0.99
    break_pattern_multiplier: float = 100000.0
    break_pattern_use_frac: bool = True
    detail_level: float = 0.0
    ridge_blend: BlendFunction = BlendFunction()
    ridge_weight: float = 0.0
    ridge_scale: float = 1.0
    warp_strength: float = 0.0
    octave_shift: float = 0.0
    curl_strength: float = 0.0
    curl_dims: tuple[int, int] = (0, 1)
    tileable: tuple[bool, ...] = (False, False, False)
    fade: Callable[[torch.Tensor], torch.Tensor] = smoothstep_function
    blend: BlendFunction = BlendFunction()
    pattern_break_blend: BlendFunction = BlendFunction()
    depth_over_channels: bool = False
    initial_depth: int = 0
    wrap_depth: int = 0
    max_depth: int = -1
    pad: tuple[int, ...] = (0, 0, 0)
    pad_mode: str = "replicate"
    generator: torch.Generator | None = None
    device: str | torch.device = "default"
    dtype: torch.dtype | None = None

    @classmethod
    def build(cls, **kwargs: Any) -> "Perlin":
        dfl = cls()
        depth = kwargs.get("depth", dfl.depth)
        for bk in ("blend", "pattern_break_blend", "ridge_blend"):
            bv = kwargs.pop(bk, getattr(dfl, bk))
            kwargs[bk] = (
                BlendFunction(bv, filtering.BLENDING_MODES[bv])
                if isinstance(bv, str)
                else bv
            )
        lacunarity = kwargs.pop("lacunarity", None)
        kwargs["lacunarity"] = (
            cls.maybe_parse_dhw_triple(
                (
                    kwargs.pop("lacunarity_depth", dfl.lacunarity[0]),
                    kwargs.pop("lacunarity_height", dfl.lacunarity[1]),
                    kwargs.pop("lacunarity_width", dfl.lacunarity[2]),
                ),
                depth,
            )
            if lacunarity is None
            else tuple(lacunarity)
        )
        res = kwargs.pop("res", None)
        kwargs["res"] = (
            cls.maybe_parse_dhw_triple(
                (
                    kwargs.pop("res_depth", dfl.res[0]),
                    kwargs.pop("res_height", dfl.res[1]),
                    kwargs.pop("res_width", dfl.res[2]),
                ),
                depth,
            )
            if res is None
            else tuple(res)
        )
        pad = kwargs.pop("pad", None)
        kwargs["pad"] = (
            (
                kwargs.pop("pad_depth", dfl.pad[0]),
                kwargs.pop("pad_height", dfl.pad[1]),
                kwargs.pop("pad_width", dfl.pad[2]),
            )
            if pad is None
            else tuple(pad)
        )
        initial_frequency = kwargs.pop("initial_frequency", None)
        kwargs["initial_frequency"] = (
            (
                kwargs.pop("initial_frequency_depth", dfl.initial_frequency[0]),
                kwargs.pop("initial_frequency_height", dfl.initial_frequency[1]),
                kwargs.pop("initial_frequency_width", dfl.initial_frequency[2]),
            )[int(depth == 0) :]
            if initial_frequency is None
            else tuple(initial_frequency)
        )
        tileable = kwargs.pop("tileable", None)
        kwargs["tileable"] = (
            (
                kwargs.pop("tileable_depth", dfl.tileable[0]),
                kwargs.pop("tileable_height", dfl.tileable[1]),
                kwargs.pop("tileable_width", dfl.tileable[2]),
            )[int(depth == 0) :]
            if tileable is None
            else tuple(tileable)
        )
        persistence = kwargs.pop("persistence", None)
        if persistence is not None:
            kwargs["persistence"] = (
                cls.maybe_parse_commasep_list(persistence)
                if isinstance(persistence, str)
                else tuple(persistence)
            )
        curl_dims = kwargs.pop("curl_dims", None)
        if curl_dims is not None:
            kwargs["curl_dims"] = tuple(
                int(v)
                for v in (
                    cls.maybe_parse_commasep_list(curl_dims)
                    if isinstance(curl_dims, str)
                    else curl_dims
                )
            )
        fs = frozenset(cls._fields)
        kwargs = {k: v for k, v in kwargs.items() if k in fs}
        return cls(**kwargs)

    @classmethod
    def maybe_parse_dhw_triple(cls, val, depth, convert=float):
        return tuple(cls.maybe_parse_commasep_list(v) for v in val)[int(depth == 0) :]

    @classmethod
    def maybe_parse_commasep_list(cls, val, convert=float):
        if not isinstance(val, str):
            return val
        return tuple(convert(v) for v in val.strip().split(",") if v.strip())

    def get_commasep(self, key, idx=None):
        val = getattr(self, key)
        if idx is not None:
            val = val[idx]
        return ", ".join(repr(v) for v in val)

    def octave(
        self,
        shape: Sequence[int],
        res: Sequence[float],
        *,
        batch_size: int = 1,
        channels: int = 1,
        octave: int = 0,
        base_noise: torch.Tensor | None = None,
        warp: torch.Tensor | None = None,
    ):
        shape = tuple(shape)
        res = tuple(res)
        dims = len(res)
        didxs = tuple(range(dims))

        coords = tuple(
            torch.linspace(
                0,
                res[i],
                shape[i] + 1,
                device=self.device,
                dtype=self.dtype,
            )[:-1]
            for i in didxs
        )
        grid_coords = torch.meshgrid(*coords, indexing="ij")
        p = torch.stack(grid_coords, dim=-1)

        # Expand `p` to include Batch and Channel dimensions
        p = (
            p.unsqueeze(0)
            .unsqueeze(0)
            .expand(
                batch_size,
                channels,
                *((-1,) * (dims + 1)),
            )
        )

        # Domain Warping: Apply warp before calculating p0 and grid.
        if warp is not None and self.warp_strength != 0.0:
            p += warp.unsqueeze(-1) * self.warp_strength

        #  Now calculate indices and bounds safely
        p0 = p.floor().long()
        grid = p - p0
        grad_shape = tuple(int(math.ceil(res[i])) + 1 for i in didxs)

        gradients = torch.randn(
            batch_size,
            channels,
            *grad_shape,
            dims,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        gradients = torch.nn.functional.normalize(gradients, dim=-1)

        # Modulate Perlin amplitude using base noise
        if base_noise is not None:
            gradients = gradients * base_noise.unsqueeze(-1).to(gradients)

        octave_shift = round((1.0 + octave) * self.octave_shift)
        if octave_shift != 0:
            gradients = gradients.roll(dims=-1, shifts=octave_shift)

        if dims > 1 and self.curl_strength != 0:
            # Generate a random spin angle for every single gradient point on the grid
            angles = torch.randn(
                batch_size,
                channels,
                *grad_shape,
                generator=self.generator,
                device=gradients.device,
                dtype=gradients.dtype,
            ).mul_(self.curl_strength)

            cos_a = angles.cos()
            sin_a = angles.sin_()

            # Grab the first two axes by default (e.g., Depth/Height, or Height/Width)
            d1, d2 = self.curl_dims[:2]
            g0 = gradients[..., d1].clone()
            g1 = gradients[..., d2].clone()

            # Apply 2D Rotation Matrix to twist the vectors
            gradients[..., d1] = (g0 * cos_a).sub_(g1 * sin_a)
            gradients[..., d2] = (g0 * sin_a).add_(g1 * cos_a)

        def get_shift(n, dims, *, on_value, off_value):
            return tuple(
                on_value if n & (1 << bitidx) else off_value for bitidx in range(dims)
            )

        def blend_reduce(vals, t, depth=0):
            curr_t = t[..., depth]
            if len(vals) == 2:
                return self.blend(*vals, curr_t)
            pairs = zip(vals[0::2], vals[1::2])
            return blend_reduce(
                tuple(self.blend(v1, v2, curr_t) for v1, v2 in pairs), t, depth + 1
            )

        ns = []

        # Pre-calculate batched indices for tensor indexing
        b_idx = torch.arange(batch_size, device=self.device).view(
            batch_size, 1, *[1] * dims
        )
        c_idx = torch.arange(channels, device=self.device).view(
            1, channels, *[1] * dims
        )

        for i in range(1 << dims):
            shift = get_shift(i, dims, off_value=0, on_value=1)

            idx = p0.clone()
            for dim in range(dims):
                idx[..., dim] += shift[dim]
                idx[..., dim] %= grad_shape[dim] - int(self.tileable[dim])

            spatial_indices = tuple(idx[..., dim] for dim in range(dims))
            grad = gradients[(b_idx, c_idx) + spatial_indices]

            grid_shift = get_shift(i, dims, off_value=0, on_value=-1)
            grid_shift_tensor = torch.tensor(
                grid_shift, dtype=self.dtype, device=self.device
            )

            d = ((grid + grid_shift_tensor) * grad).sum(dim=-1)
            ns.append(d)

        return blend_reduce(ns, self.fade(grid)).mul_(2.0**0.5)

    @staticmethod
    def get_wrap_dim(val, *dims):
        for dim in dims:
            nelem = len(val) if not isinstance(val, torch.Tensor) else val.shape[0]
            val = val[dim % nelem]
        return val

    def get_unwrapped_octaves_dims(self, val, ndim: int) -> torch.Tensor:
        return torch.tensor(
            tuple(
                self.get_wrap_dim(val, didx, oidx)
                for oidx in range(self.octaves)
                for didx in range(ndim)
            ),
            dtype=self.dtype,
            device=self.device,
        ).reshape(self.octaves, ndim)

    def generate_octaves(
        self,
        shape: Sequence[int],
        *,
        batch_size: int = 1,
        channels: int = 1,
        base_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = tuple(shape)
        ndim = len(shape)

        amplitude = self.initial_amplitude
        res = self.get_unwrapped_octaves_dims(self.res, ndim)
        lacunarity = self.get_unwrapped_octaves_dims(self.lacunarity, ndim)
        persistence = self.persistence[: self.octaves]
        previous_octave = None

        initial_frequency = self.initial_frequency[-ndim:]
        frequency = torch.ones(ndim, dtype=self.dtype, device=self.device)
        frequency[: len(initial_frequency)] = frequency.new(initial_frequency)
        noise = torch.zeros(
            (batch_size, channels, *shape),
            dtype=self.dtype,
            device=self.device,
        )

        for octave in range(self.octaves):
            octave_res = tuple(
                frequency[didx].item() * res[octave][didx].item()
                for didx in range(ndim)
            )

            grad_shape = tuple(int(math.ceil(octave_res[i])) + 1 for i in range(ndim))

            if base_noise is None:
                octave_base_noise = None
            else:
                if base_noise.shape[-len(grad_shape) :] != grad_shape:
                    mode = (
                        "bilinear"
                        if ndim == 2
                        else ("trilinear" if ndim == 3 else "nearest")
                    )
                    octave_base_noise = torch.nn.functional.interpolate(
                        base_noise,
                        size=grad_shape,
                        mode=mode,
                        **(
                            {"align_corners": False}
                            if mode not in ("nearest", "area")
                            else {}
                        ),
                    )
                else:
                    octave_base_noise = base_noise

            octave_output = self.octave(
                shape,
                octave_res,
                batch_size=batch_size,
                channels=channels,
                octave=octave,
                base_noise=octave_base_noise,
                warp=previous_octave,
            )

            if self.ridge_weight != 0.0:
                ridge = (
                    1.0
                    - octave_output.div(
                        octave_output.abs().max().clamp_min_(1e-07)
                    ).abs_()
                )
                ridge -= 0.5
                ridge *= 2.0 * self.ridge_scale
                octave_output = self.ridge_blend(
                    octave_output,
                    ridge,
                    self.ridge_weight,
                )

            noise += amplitude * octave_output
            previous_octave = octave_output

            frequency *= lacunarity[octave]
            amplitude *= self.get_wrap_dim(persistence, octave)

        return noise

    # Based on approach from https://github.com/Extraltodeus/noise_latent_perlinpinpin
    @staticmethod
    def break_pattern_func(
        t: torch.Tensor,
        detail: float = 0.0,
        *,
        multiplier: float = 1000000.0,
        use_frac: bool = False,
        clamp_low: float = -5.0,
        clamp_high: float = 5.0,
    ) -> torch.Tensor:
        detail_factor = (1 + detail * 0.1) * 2.0**0.5 * 0.2
        result = t.abs().mul_(multiplier)
        result = result.frac_() if use_frac else result.remainer_(11).div_(11)
        return (
            result.mul_(2)
            .sub_(1)
            .erfinv_()
            .mul_(detail_factor)
            .clamp_(clamp_low, clamp_high)
        )

    def __call__(
        self,
        width: int,
        height: int,
        *,
        batch_size: int = 1,
        channels: int = 4,
        base_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        depth = self.depth
        pad_depth, pad_height, pad_width = self.pad[:3]
        depth_over_channels = self.depth_over_channels

        if depth < 1:
            depth_over_channels = False
            pad_depth = 0
            eff_shape = (height + pad_height * 2, width + pad_width * 2)
            eff_channels = channels
            eff_depth = 0
            noise_dims = 2
        else:
            eff_channels = channels if not depth_over_channels else 1
            eff_depth = depth if not depth_over_channels else depth * channels
            eff_shape = (
                eff_depth + pad_depth * 2,
                height + pad_height * 2,
                width + pad_width * 2,
            )
            noise_dims = 3

        bn = base_noise
        if bn is not None:
            if depth_over_channels and depth > 0:
                # Flattens 5D base_noise into a contiguous sequential depth format matching outputs
                # (B, C, D, H, W) -> (B, 1, D*C, H, W)
                bn = bn.movedim(1, 2).reshape(batch_size, 1, eff_depth, height, width)

            if pad_width > 0 or pad_height > 0 or pad_depth > 0:
                if noise_dims == 3:
                    pad_tuple = (
                        pad_width,
                        pad_width,
                        pad_height,
                        pad_height,
                        pad_depth,
                        pad_depth,
                    )
                else:
                    pad_tuple = (pad_width, pad_width, pad_height, pad_height)
                bn = torch.nn.functional.pad(bn, pad_tuple, mode=self.pad_mode)

        noise_values = self.generate_octaves(
            eff_shape,
            batch_size=batch_size,
            channels=eff_channels,
            base_noise=bn,
        )

        # Apply normalization to the spatial dimensions individually per batch and channel
        norm_dims = tuple(range(-len(eff_shape), 0))
        noise_values = normalize_to_scale(noise_values, -1.0, 1.0, dim=norm_dims)

        if self.break_pattern != 0.0:
            result = self.pattern_break_blend(
                noise_values,
                self.break_pattern_func(
                    noise_values,
                    detail=self.detail_level,
                    use_frac=self.break_pattern_use_frac,
                    multiplier=self.break_pattern_multiplier,
                ),
                self.break_pattern,
            )
        else:
            result = noise_values

        if sum(self.pad[:3]) > 0:
            if noise_dims == 3:
                result = result[
                    :,
                    :,
                    pad_depth : eff_depth + pad_depth,
                    pad_height : height + pad_height,
                    pad_width : width + pad_width,
                ]
            else:
                result = result[
                    :,
                    :,
                    pad_height : height + pad_height,
                    pad_width : width + pad_width,
                ]

        if depth_over_channels and depth > 0:
            # Map the contiguous sequential flattened depth back identically matching (D0_C0, D0_C1, ...) interleaving
            # (B, 1, D*C, H, W) -> (B, C, D, H, W)
            result = result.reshape(
                batch_size,
                depth,
                channels,
                height,
                width,
            ).movedim(1, 2)

        if noise_dims == 3:
            # Shift Depth index backwards ahead of Batch (per expectation of the make_noise_sampler)
            # (B, C, D, H, W) -> (D, B, C, H, W)
            result = result.movedim(2, 0)

        return result.contiguous()


class PerlinItem(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        perlin: Perlin | None = None,
        device=None,
        normalized=None,
        base_noise_opt=None,
        **kwargs: Any,
    ):
        if perlin is None:
            perlin = Perlin.build(**kwargs)
        super().__init__(
            factor,
            perlin=perlin,
            device=device,
            normalized=normalized
            if not isinstance(normalized, str)
            else NormalizeNoiseNodeMixin.get_normalize(normalized),
            base_noise_opt=base_noise_opt.clone()
            if base_noise_opt is not None
            else None,
            # **kwargs,
        )

    def make_noise_sampler(
        self,
        x: torch.Tensor,
        sigma_min: float | None,
        sigma_max: float | None,
        seed: int | None,
        cpu: bool = True,
        normalized=True,
    ) -> torch.Tensor:  # ty:ignore[invalid-method-override]
        normalized = self.get_normalize("normalized", normalized)
        cpu = cpu if self.device == "default" and cpu else self.device == "cpu"
        device = torch.device("cpu") if cpu else model_management.get_torch_device()
        perlin: Perlin = self.perlin._replace(device=device, dtype=x.dtype)
        noise_chunk = None
        noise_index = perlin.initial_depth
        max_idx = None
        if x.ndim < 4:
            raise ValueError("Can only handle latents with 4+ dimensions")
        orig_shape = x.shape
        b = orig_shape[0]
        c = math.prod(orig_shape[1:-2])  # Hack to deal with video models
        h, w = orig_shape[-2:]

        base_noise_sampler = (
            self.base_noise_opt.make_noise_sampler(
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cpu,
                normalized=False,
            )
            if self.base_noise_opt
            else None
        )

        x_device, x_dtype = x.device, x.dtype
        del x

        def noise_sampler(s, sn):
            nonlocal noise_chunk, noise_index, max_idx
            if noise_chunk is None:
                base_noise = None
                if base_noise_sampler:
                    bn_tuple = tuple(
                        base_noise_sampler(s, sn).reshape(b, c, h, w)
                        for _ in range(max(1, perlin.depth))
                    )
                    base_noise = (
                        bn_tuple[0]
                        if perlin.depth < 1
                        else torch.stack(bn_tuple, dim=0).movedim(0, 2)
                    )
                    del bn_tuple

                noise_chunk = perlin(
                    w,
                    h,
                    batch_size=b,
                    channels=c,
                    base_noise=base_noise,
                ).to(device=x_device, dtype=x_dtype)

                if perlin.depth < 1:
                    noise = noise_chunk
                    noise_chunk = None
                    return scale_noise(noise, self.factor, normalized=normalized)
                if perlin.max_depth != 0 and perlin.max_depth != -1:
                    noise_chunk = noise_chunk[: perlin.max_depth]
                chunk_shape = noise_chunk.shape
                max_idx = (
                    chunk_shape[0] - 1
                    if perlin.wrap_depth == 0
                    else min(perlin.wrap_depth, chunk_shape[0] - 1)
                )
                if max_idx < 0:
                    max_idx += chunk_shape[0]

            noise = noise_chunk[noise_index]
            noise_index += 1
            if noise_index > max_idx:
                noise_index = 0
                if not perlin.wrap_depth:
                    noise_chunk = None
            result = scale_noise(noise, self.factor, normalized=normalized)
            return result.reshape(orig_shape) if result.shape != orig_shape else result

        return noise_sampler
