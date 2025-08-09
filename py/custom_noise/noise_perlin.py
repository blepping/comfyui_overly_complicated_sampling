import itertools
import math

import torch

from comfy import model_management

from .. import filtering
from ..latent import normalize_to_scale
from ..noise import scale_noise
from .base import CustomNoiseItemBase, NormalizeNoiseNodeMixin

# Perlin generation routines based on https://github.com/Extraltodeus/noise_latent_perlinpinpin which was based on https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57 which was based on https://github.com/pvigier/perlin-numpy


def smoothstep_function(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


class DEFAULTS:
    depth = 16
    res = ((1,), (1,), (1,))
    octaves = 2
    persistence = (1.0,)
    lacunarity = ((2,), (2,), (2,))
    initial_amplitude = 1.0
    initial_frequency = (1.0, 1.0, 1.0)
    break_pattern = 1.0
    detail_level = 0.0
    tileable = (False, False, False)
    fade = smoothstep_function
    blend = "lerp"
    pattern_break_blend = "lerp"
    depth_over_channels = False
    initial_depth = 0
    wrap_depth = 0
    max_depth = -1
    pad = (0, 0, 0)
    generator = None
    device = "default"

    @classmethod
    def get_commasep(cls, key, idx=None):
        val = getattr(cls, key)
        if idx is not None:
            val = val[idx]
        return ", ".join(repr(v) for v in val)


def rand_perlin(
    shape,
    res,
    *,
    tileable=DEFAULTS.tileable,
    fade=DEFAULTS.fade,
    blend=torch.lerp,
    generator=DEFAULTS.generator,
    device=DEFAULTS.device,
):
    dims = len(res)
    didxs = tuple(range(dims))
    delta, d = zip(*((res[i] / shape[i], int(round(shape[i] / res[i]))) for i in didxs))

    grid = (
        torch.stack(
            torch.meshgrid(*(torch.arange(0, res[i], delta[i]) for i in didxs)),
            dim=-1,
        )
        % 1
    ).to(device=device)

    noise = (
        2
        * math.pi
        * torch.rand(
            max(1, dims - 1),
            *(round(res[i]) + 1 for i in didxs),
            generator=generator,
            device=device,
        )
    )
    if dims == 1:
        gradients = torch.cos(noise[0])
    elif dims == 2:
        gradients = torch.stack((torch.cos(noise[0]), torch.sin(noise[0])), dim=-1)
    elif dims == 3:
        gradients = torch.stack(
            (
                torch.sin(noise[0]) * torch.cos(noise[1]),
                torch.sin(noise[0]) * torch.sin(noise[1]),
                torch.cos(noise[0]),
            ),
            dim=-1,
        )
    elif dims == 4:
        # No idea if this makes sense.
        gradients = torch.stack(
            (
                torch.sin(noise[0]) * torch.cos(noise[1]),
                torch.sin(noise[0]) * torch.sin(noise[1]),
                torch.sin(noise[1]) * torch.cos(noise[2]),
                torch.sin(noise[1]) * torch.sin(noise[2]),
            ),
            dim=-1,
        )
    else:
        raise ValueError("Currently only dimensions up to 4 are supported")
    del noise

    for tidx, tile in enumerate(tileable[:dims]):
        if not tile:
            continue
        gradients[tuple(-1 if didx == tidx else None for didx in didxs)] = gradients[
            tuple(0 if didx == tidx else None for didx in didxs)
        ]

    shape_slices = tuple(slice(0, shape[i]) for i in didxs)

    def tile_grads(slices):
        result = gradients[tuple(slice(*slices[i]) for i in didxs)]
        for i in didxs:
            result = result.repeat_interleave(d[i], i)
        return result

    def dot(grad, shift):
        return (
            torch.stack(
                tuple(grid[(*shape_slices, i)] + shift[i] for i in didxs), dim=-1
            )
            * grad[shape_slices]
        ).sum(dim=-1)

    # It's just binary with the bits reversed and -1 for enabled columns.
    def get_shift(n, dims, *, on_value, off_value):
        return tuple(
            on_value if n & (1 << bitidx) else off_value for bitidx in range(dims)
        )

    def blend_reduce(vals, t, depth=0):
        curr_t = t[..., depth]
        if len(vals) == 2:
            return blend(*vals, curr_t)
        return blend_reduce(
            tuple(blend(v1, v2, curr_t) for v1, v2 in itertools.batched(vals, 2)),
            t,
            depth + 1,
        )

    ns = tuple(
        dot(
            tile_grads(get_shift(i, dims, off_value=(None, -1), on_value=(1, None))),
            get_shift(i, dims, off_value=0, on_value=-1),
        )
        for i in range(1 << dims)
    )
    return math.sqrt(2) * blend_reduce(ns, fade(grid[shape_slices]))


def generate_fractal_noise(
    shape,
    res=DEFAULTS.res,
    octaves=DEFAULTS.octaves,
    persistence=DEFAULTS.persistence,
    lacunarity=DEFAULTS.lacunarity,
    initial_amplitude=DEFAULTS.initial_amplitude,
    initial_frequency=DEFAULTS.initial_frequency,
    tileable=DEFAULTS.tileable,
    fade=DEFAULTS.fade,
    blend=torch.lerp,
    generator=DEFAULTS.generator,
    device=DEFAULTS.device,
):
    ndim = len(shape)

    def get_wrap_dim(val, *dims):
        for dim in dims:
            nelem = len(val) if not isinstance(val, torch.Tensor) else val.shape[0]
            val = val[dim % nelem]
        return val

    def get_unwrapped_octaves_dims(val):
        return torch.tensor(
            tuple(
                get_wrap_dim(val, didx, oidx)
                for oidx in range(octaves)
                for didx in range(ndim)
            ),
            dtype=torch.float,
            device="cpu",
        ).view(octaves, ndim)

    res = get_unwrapped_octaves_dims(res)
    lacunarity = get_unwrapped_octaves_dims(lacunarity)
    initial_frequency = initial_frequency[-ndim:]
    persistence = persistence[:octaves]
    noise = torch.zeros(shape, dtype=torch.float32, device=device)
    frequency = torch.ones(ndim, dtype=torch.float, device="cpu")
    frequency[: len(initial_frequency)] = frequency.new(initial_frequency)
    amplitude = initial_amplitude

    for octave in range(octaves):
        noise += amplitude * rand_perlin(
            shape,
            tuple(
                frequency[didx].item() * res[octave][didx].item()
                for didx in range(ndim)
            ),
            tileable=tileable,
            fade=fade,
            blend=blend,
            generator=generator,
            device=device,
        )
        # print(
        #     f"Octave {octave}: freq={frequency}, amp={amplitude}, lac={lacunarity[octave]}, pers={get_wrap_dim(persistence, octave)}"
        # )
        frequency *= lacunarity[octave]
        amplitude *= get_wrap_dim(persistence, octave)
        # print(f"Octave {octave}: POST: freq={frequency}, amp={amplitude}")
    return noise


def create_noisy_latents_perlin(
    width,
    height,
    depth,
    *,
    batch_size=1,
    detail_level=DEFAULTS.detail_level,
    octaves=DEFAULTS.octaves,
    persistence=DEFAULTS.persistence,
    lacunarity=DEFAULTS.lacunarity,
    tileable=DEFAULTS.tileable,
    res=DEFAULTS.res,
    break_pattern=DEFAULTS.break_pattern,
    channels=4,
    blend=torch.lerp,
    pattern_break_blend=torch.lerp,
    depth_over_channels=DEFAULTS.depth_over_channels,
    pad=DEFAULTS.pad,
    initial_frequency=DEFAULTS.initial_frequency,
    initial_amplitude=DEFAULTS.initial_amplitude,
    generator=DEFAULTS.generator,
    device=DEFAULTS.device,
):
    pad_depth, pad_height, pad_width = pad
    if depth < 1:
        depth_over_channels = False
        pad_depth = 0
        shape = (height, width)
        eff_shape = (
            height + pad_height * 2,
            width + pad_width * 2,
        )
    eff_channels = channels if not depth_over_channels else 1
    eff_depth = depth if not depth_over_channels else depth * channels
    if depth > 0:
        shape = (depth, height, width)
        eff_shape = (
            eff_depth + pad_depth * 2,
            height + pad_height * 2,
            width + pad_width * 2,
        )
    noise = torch.zeros(
        (batch_size, channels, *shape),
        dtype=torch.float32,
        device=device,
    )
    noise_dims = len(shape)
    for i in range(batch_size):
        for j in range(eff_channels):
            noise_values = generate_fractal_noise(
                eff_shape,
                res=res,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                tileable=tileable,
                blend=blend,
                initial_frequency=initial_frequency,
                initial_amplitude=initial_amplitude,
                generator=generator,
                device=device,
            )
            noise_values = normalize_to_scale(noise_values, -1.0, 1.0, dim=())
            if break_pattern != 0:
                result = torch.remainder(torch.abs(noise_values) * 1000000, 11) / 11
                result = (
                    ((1 + detail_level / 10) * torch.erfinv(2 * result - 1) * (2**0.5))
                    .mul_(0.2)
                    .clamp_(-1, 1)
                )
                result = pattern_break_blend(noise_values, result, break_pattern)
            else:
                result = noise_values
            if pad_width + pad_height + pad_depth > 0:
                result = (
                    result[
                        ...,
                        pad_depth : eff_depth + pad_depth,
                        pad_height : height + pad_height,
                        pad_width : width + pad_width,
                    ]
                    if noise_dims == 3
                    else result[
                        ...,
                        pad_height : height + pad_height,
                        pad_width : width + pad_width,
                    ]
                )
            if not depth_over_channels:
                noise[i, j, ...] = result
                continue
            noise[i, ...] = result.view(depth, channels, height, width).movedim(0, 1)
    return noise.movedim(-3, 0) if noise_dims == 3 else noise


class PerlinItem(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        depth=20,
        detail_level=DEFAULTS.detail_level,
        octaves=DEFAULTS.octaves,
        persistence=DEFAULTS.persistence,
        lacunarity_depth=DEFAULTS.lacunarity[0],
        lacunarity_height=DEFAULTS.lacunarity[1],
        lacunarity_width=DEFAULTS.lacunarity[2],
        lacunarity=None,
        tileable_depth=DEFAULTS.tileable[0],
        tileable_height=DEFAULTS.tileable[1],
        tileable_width=DEFAULTS.tileable[2],
        tileable=None,
        res_depth=DEFAULTS.res[0],
        res_height=DEFAULTS.res[1],
        res_width=DEFAULTS.res[2],
        res=None,
        initial_frequency_depth=DEFAULTS.initial_frequency[0],
        initial_frequency_height=DEFAULTS.initial_frequency[1],
        initial_frequency_width=DEFAULTS.initial_frequency[2],
        initial_frequency=None,
        initial_amplitude=DEFAULTS.initial_amplitude,
        wrap_depth=DEFAULTS.wrap_depth,
        initial_depth=DEFAULTS.initial_depth,
        max_depth=DEFAULTS.max_depth,
        break_pattern=DEFAULTS.break_pattern,
        blend=DEFAULTS.blend,
        pattern_break_blend=DEFAULTS.pattern_break_blend,
        depth_over_channels=DEFAULTS.depth_over_channels,
        pad=None,
        pad_depth=DEFAULTS.pad[0],
        pad_height=DEFAULTS.pad[1],
        pad_width=DEFAULTS.pad[2],
        device=None,
        normalized=None,
        **kwargs,
    ):
        if tileable is None:
            tileable = (tileable_depth, tileable_height, tileable_width)[
                int(depth == 0) :
            ]
        if res is None:
            res = self.maybe_parse_dhw_triple(
                (res_depth, res_height, res_width), depth, int
            )
        if lacunarity is None:
            lacunarity = self.maybe_parse_dhw_triple(
                (
                    lacunarity_depth,
                    lacunarity_height,
                    lacunarity_width,
                ),
                depth,
            )
        if pad is None:
            pad = (pad_depth, pad_height, pad_width)
        if initial_frequency is None:
            initial_frequency = (
                initial_frequency_depth,
                initial_frequency_height,
                initial_frequency_width,
            )[int(depth == 0) :]
        persistence = self.maybe_parse_commasep_list(persistence)
        super().__init__(
            factor,
            depth=depth,
            detail_level=detail_level,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            tileable=tileable,
            res=res,
            initial_frequency=initial_frequency,
            initial_amplitude=initial_amplitude,
            initial_depth=initial_depth,
            wrap_depth=wrap_depth,
            max_depth=max_depth,
            break_pattern=break_pattern,
            blend=blend,
            pattern_break_blend=pattern_break_blend,
            depth_over_channels=depth_over_channels,
            pad=pad,
            device=device,
            normalized=normalized
            if not isinstance(normalized, str)
            else NormalizeNoiseNodeMixin.get_normalize(normalized),
            **kwargs,
        )

    @classmethod
    def maybe_parse_dhw_triple(cls, val, depth, convert=float):
        return tuple(cls.maybe_parse_commasep_list(v) for v in val)[int(depth == 0) :]

    @classmethod
    def maybe_parse_commasep_list(cls, val, convert=float):
        if not isinstance(val, str):
            return val
        return tuple(convert(v) for v in val.strip().split(",") if v.strip())

    def make_noise_sampler(
        self,
        x: torch.Tensor,
        sigma_min: float | None,
        sigma_max: float | None,
        seed: int | None,
        cpu: bool = True,
        normalized=True,
    ):
        normalized = self.get_normalize("normalized", normalized)
        cpu = cpu if self.device == "default" else self.device == "cpu"
        device = "cpu" if cpu else model_management.get_torch_device()
        noise_chunk = None
        noise_index = self.initial_depth
        max_idx = None
        if x.ndim < 4:
            raise ValueError("Can only handle latents with 4+ dimensions")
        orig_shape = x.shape
        b = x.shape[0]
        c = math.prod(x.shape[1:-2])  # Hack to deal with video models.
        h, w = x.shape[-2:]
        x_device, x_dtype = x.device, x.dtype
        del x
        blend = filtering.BLENDING_MODES[self.blend]
        pattern_break_blend = filtering.BLENDING_MODES[self.pattern_break_blend]

        def noise_sampler(_s, _sn):
            nonlocal noise_chunk, noise_index, max_idx
            if noise_chunk is None:
                # print("-->", noise_index, self.depth)
                noise_chunk = create_noisy_latents_perlin(
                    w,
                    h,
                    self.depth,
                    batch_size=b,
                    channels=c,
                    detail_level=self.detail_level,
                    octaves=self.octaves,
                    persistence=self.persistence,
                    lacunarity=self.lacunarity,
                    initial_frequency=self.initial_frequency,
                    initial_amplitude=self.initial_amplitude,
                    break_pattern=self.break_pattern,
                    res=self.res,
                    tileable=self.tileable,
                    blend=blend,
                    pattern_break_blend=pattern_break_blend,
                    depth_over_channels=self.depth_over_channels,
                    pad=self.pad,
                    device=device,
                ).to(device=x_device, dtype=x_dtype)
                if self.depth < 1:  # 2D mode
                    noise = noise_chunk
                    noise_chunk = None
                    return scale_noise(noise, self.factor, normalized=normalized)
                if self.max_depth != 0 and self.max_depth != -1:
                    noise_chunk = noise_chunk[: self.max_depth]
                chunk_shape = noise_chunk.shape
                max_idx = (
                    chunk_shape[0] - 1
                    if self.wrap_depth == 0
                    else min(self.wrap_depth, chunk_shape[0] - 1)
                )
                if max_idx < 0:
                    max_idx += chunk_shape[0]
            noise = noise_chunk[noise_index]
            noise_index += 1
            if noise_index > max_idx:
                noise_index = 0
                if not self.wrap_depth:
                    noise_chunk = None
            result = scale_noise(noise, self.factor, normalized=normalized)
            return result.reshape(orig_shape) if result.shape != orig_shape else result

        return noise_sampler
