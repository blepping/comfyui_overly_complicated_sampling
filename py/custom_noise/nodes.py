import inspect
import math
import torch
import yaml

from comfy.model_management import get_torch_device
from comfy.model_patcher import set_model_options_post_cfg_function
import comfy.samplers

from .base import CustomNoiseNodeBase, NormalizeNoiseNodeMixin
from .noise_perlin import DEFAULTS as PERLIN_DEFAULTS
from .noise_perlin import PerlinItem
from .noise_immiscibleref import ImmiscibleReferenceItem

from ..external import MODULES, IntegratedNode
from ..filtering import BLENDING_MODES
from ..nodes import WILDCARD_NOISE, NOISE_INPUT_TYPES_HINT
from ..utils import scale_noise
from ..noise import ImmiscibleNoise


class ToSonarNode:
    RETURN_TYPES = ("SONAR_CUSTOM_NOISE",)
    CATEGORY = "OveryComplicatedSampling/noise"
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ocs_noise": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": NOISE_INPUT_TYPES_HINT,
                        "forceInput": True,
                    },
                ),
            },
        }

    @classmethod
    def go(cls, ocs_noise):
        MODULES.initialize()
        return (ocs_noise,)


class PerlinAdvancedNode(CustomNoiseNodeBase, NormalizeNoiseNodeMixin):
    DESCRIPTION = "Advanced Perlin noise generator, allows generating 2D or 3D Perlin noise. See the OCSNoise PerlinSimple node for less tuneable parameters."

    @classmethod
    def INPUT_TYPES(cls):
        MODULES.initialize()
        result = super().INPUT_TYPES()
        result["required"] |= {
            "depth": (
                "INT",
                {
                    "default": PERLIN_DEFAULTS.depth,
                    "tooltip": "When non-zero, 3D perlin noise will be generated.",
                },
            ),
            "detail_level": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.detail_level,
                    "tooltip": "Controls the detail level of the noise when break_pattern is non-zero. No effect when using 100% raw Perlin noise.",
                },
            ),
            "octaves": (
                "INT",
                {
                    "default": PERLIN_DEFAULTS.octaves,
                    "tooltip": "Generally controls the detail level of the noise. Each octave involves generating a layer of noise so there is a performance cost to increasing octaves.",
                },
            ),
            "persistence": (
                "STRING",
                {
                    "default": PERLIN_DEFAULTS.get_commasep("persistence"),
                    "tooltip": "Controls how rough the generated noise is. Lower values will result in smoother noise, higher values will look more like Gaussian noise. Comma-separated list, multiple items will apply to octaves in sequence.",
                },
            ),
            "lacunarity_height": (
                "STRING",
                {
                    "default": PERLIN_DEFAULTS.get_commasep("lacunarity", 0),
                    "tooltip": "Lacunarity controls the frequency multiplier between successive octaves. Only has an effect when octaves is greater than one. Comma-separated list, multiple items will apply to octaves in sequence.",
                },
            ),
            "lacunarity_width": (
                "STRING",
                {
                    "default": PERLIN_DEFAULTS.get_commasep("lacunarity", 1),
                    "tooltip": "Lacunarity controls the frequency multiplier between successive octaves. Only has an effect when octaves is greater than one. Comma-separated list, multiple items will apply to octaves in sequence.",
                },
            ),
            "lacunarity_depth": (
                "STRING",
                {
                    "default": PERLIN_DEFAULTS.get_commasep("lacunarity", 2),
                    "tooltip": "Lacunarity controls the frequency multiplier between successive octaves. Only has an effect when depth is non-zero and octaves is greater than one. Comma-separated list, multiple items will apply to octaves in sequence.",
                },
            ),
            "res_height": (
                "STRING",
                {
                    "default": PERLIN_DEFAULTS.get_commasep("res", 0),
                    "tooltip": "Number of periods of noise to generate along an axis. Comma-separated list, multiple items will apply to octaves in sequence.",
                },
            ),
            "res_width": (
                "STRING",
                {
                    "default": PERLIN_DEFAULTS.get_commasep("res", 1),
                    "tooltip": "Number of periods of noise to generate along an axis. Comma-separated list, multiple items will apply to octaves in sequence.",
                },
            ),
            "res_depth": (
                "STRING",
                {
                    "default": PERLIN_DEFAULTS.get_commasep("res", 2),
                    "tooltip": "Number of periods of noise to generate along an axis. Only has an effect when depth is non-zero. Comma-separated list, multiple items will apply to octaves in sequence.",
                },
            ),
            "break_pattern": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.break_pattern,
                    "tooltip": "Applies a function to break the Perlin pattern, making it more like normal noise. The value is the blend strength, where 1.0 indicates 100% pattern broken noise and 0.5 indicates 50% raw noise and 50% pattern broken noise. Generally should be at least 0.9 unless you want to generate colorful blobs.",
                },
            ),
            "initial_depth": (
                "INT",
                {
                    "default": PERLIN_DEFAULTS.initial_depth,
                    "tooltip": "First zero-based depth index the noise generator will return. Only has an effect when depth is non-zero.",
                },
            ),
            "wrap_depth": (
                "INT",
                {
                    "default": PERLIN_DEFAULTS.wrap_depth,
                    "tooltip": "If non-zero, instead of generating a new chunk of noise when the last slice is used will instead jump back to the specified zero-based depth index. Only has an effect when depth is non-zero.",
                },
            ),
            "max_depth": (
                "INT",
                {
                    "default": PERLIN_DEFAULTS.max_depth,
                    "tooltip": "Basically crops the depth dimension to the specified value (inclusive). Negative values start from the end, the default of -1 does no cropping. Only has an effect when depth is non-zero.",
                },
            ),
            "tileable_height": (
                "BOOLEAN",
                {
                    "default": PERLIN_DEFAULTS.tileable[0],
                    "tooltip": "Makes the specified dimension tileable.",
                },
            ),
            "tileable_width": (
                "BOOLEAN",
                {
                    "default": PERLIN_DEFAULTS.tileable[1],
                    "tooltip": "Makes the specified dimension tileable.",
                },
            ),
            "tileable_depth": (
                "BOOLEAN",
                {
                    "default": PERLIN_DEFAULTS.tileable[2],
                    "tooltip": "Makes the specified dimension tileable. Only has an effect when depth is non-zero.",
                },
            ),
            "blend": (
                tuple(BLENDING_MODES.keys()),
                {
                    "default": "lerp",
                    "tooltip": "Blending function used when generating Perlin noise. When set to values other than LERP may not work at all or may not actually generate Perlin noise.",
                },
            ),
            "pattern_break_blend": (
                tuple(BLENDING_MODES.keys()),
                {
                    "default": "lerp",
                    "tooltip": "Blending function used to blend pattern broken noise with raw noise.",
                },
            ),
            "depth_over_channels": (
                "BOOLEAN",
                {
                    "default": PERLIN_DEFAULTS.depth_over_channels,
                    "tooltip": "When disabled, each channel will have its own separate 3D noise pattern. When enabled, depth is multiplied by the number of channels and each channel is a slice of depth. Only has an effect when depth is non-zero.",
                },
            ),
            "pad_height": (
                "INT",
                {
                    "default": PERLIN_DEFAULTS.pad[0],
                    "min": 0,
                    "tooltip": "Pads the specified dimension by the size. Equal padding will be added on both sides and cropped out after generation.",
                },
            ),
            "pad_width": (
                "INT",
                {
                    "default": PERLIN_DEFAULTS.pad[1],
                    "min": 0,
                    "tooltip": "Pads the specified dimension by the size. Equal padding will be added on both sides and cropped out after generation.",
                },
            ),
            "pad_depth": (
                "INT",
                {
                    "default": PERLIN_DEFAULTS.pad[2],
                    "min": 0,
                    "tooltip": "Pads the specified dimension by the size. Equal padding will be added on both sides and cropped out after generation. Only has an effect when depth is non-zero.",
                },
            ),
            "initial_amplitude": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.initial_amplitude,
                    "tooltip": "Controls the amplitude for the first octave.",
                },
            ),
            "initial_frequency_height": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.initial_frequency[0],
                    "tooltip": "Controls the frequency for the first octave for the this axis.",
                },
            ),
            "initial_frequency_width": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.initial_frequency[1],
                    "tooltip": "Controls the frequency for the first octave for the this axis.",
                },
            ),
            "initial_frequency_depth": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.initial_frequency[2],
                    "tooltip": "Controls the frequency for the first octave for the this axis.",
                },
            ),
            "normalize": (
                ("default", "forced", "off"),
                {
                    "tooltip": "Controls whether the output noise is normalized after generation.",
                },
            ),
            "device": (
                ("default", "cpu", "gpu"),
                {
                    "default": "default",
                    "tooltip": "Controls what device is used to generate the noise. GPU noise may be slightly faster but you will get different results on different GPUs.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return PerlinItem


class PerlinSimpleNode(PerlinAdvancedNode):
    DESCRIPTION = "Simplified Perlin noise generator, allows generating 2D or 3D Perlin noise. See the OCSNoise PerlinAdvanced node for more tuneable parameters."

    _COPY_KEYS = {
        "factor",
        "rescale",
        "depth",
        "detail_level",
        "octaves",
        "persistence",
        "break_pattern",
    }

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        orig_reqs = result["required"]
        reqs = {k: v for k, v in orig_reqs.items() if k in cls._COPY_KEYS}
        reqs["lacunarity"] = orig_reqs["lacunarity_height"]
        reqs["res"] = orig_reqs["res_height"]
        result["required"] = reqs
        return result

    @classmethod
    def get_item_class(cls):
        def wrapper(factor, *, lacunarity, res, **kwargs):
            return PerlinItem(
                factor,
                lacunarity_height=lacunarity,
                lacunarity_width=lacunarity,
                lacunarity_depth=lacunarity,
                res_height=res,
                res_width=res,
                res_depth=res,
                **kwargs,
            )

        return wrapper


class ImmiscibleReferenceNoiseNode(CustomNoiseNodeBase, NormalizeNoiseNodeMixin):
    DESCRIPTION = "Immiscible noise that uses a latent reference."

    @classmethod
    def INPUT_TYPES(cls):
        MODULES.initialize()
        result = super().INPUT_TYPES(include_rescale=False, include_chain=False)
        result["required"] |= {
            "size": (
                "INT",
                {
                    "default": 64,
                    "min": 0,
                    "tooltip": "Number of batch repeats to use when generating Immiscible noise. Setting this to 0 disables immiscible noise. If the batching type is batch, then Immiscible noise is also disabled unless the size is 2 or higher. Note that this size is in batch repeats regardless of the batching mode. For example, if you are generating a batch of 2 and you set this to 2, then you will generate noise with batch size 4.",
                },
            ),
            "batching": (
                (
                    "channel",
                    "batch",
                    "row",
                    "column",
                    "frame",
                ),
                {
                    "default": "channel",
                    "tooltip": "Dimension to maximize (or minimize) the noise with. Column mode requires reshaping the input and may require a lot of VRAM. Row mode is also fairly slow, but not as bad as column mode. Row and column modes have a very strong effect.",
                },
            ),
            "normalize_ref_scale": (
                "FLOAT",
                {
                    "default": 0.0,
                    "tooltip": "Controls whether the reference gets normalized. If set to 0, no normalization is done.",
                },
            ),
            "normalize_noise_scale": (
                "FLOAT",
                {
                    "default": 0.0,
                    "tooltip": "Controls whether the noise used as an input for immiscible noise is gets normalized first. If set to 0, no normalization is done.",
                },
            ),
            "maximize": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "When enabled, maximizes the distance between the noise and the reference rather than trying to minimize it.",
                },
            ),
            "distance_scale": (
                "FLOAT",
                {
                    "default": 0.1,
                    "tooltip": "Multiplier on the input noise for v2 Immiscible noise. Set to 0 to use v1 Immiscible noise.",
                },
            ),
            "distance_scale_ref": (
                "FLOAT",
                {
                    "default": 0.1,
                    "tooltip": "Multiplier on the refence for v2 Immiscible noise. No effect if distance_scale is 0.",
                },
            ),
            "blend": (
                "FLOAT",
                {
                    "default": 1.0,
                    "tooltip": "Percentage of immiscible noise to use. 1.0 means 100%. May not work very well with most blend modes.",
                },
            ),
            "blend_mode": (
                tuple(BLENDING_MODES.keys()),
                {
                    "default": "lerp",
                    "tooltip": "Blending function used when mixing immiscible noise with normal noise. Only slerp seems to work well (requires ComfyUI-bleh).",
                },
            ),
            "custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": "Input for custom noise used during ancestral or SDE sampling.",
                },
            ),
            "reference": ("LATENT",),
        }
        return result

    @classmethod
    def get_item_class(cls):
        def wrapper(
            factor: float, *, reference: dict, blend_mode: str, custom_noise, **kwargs
        ):
            return ImmiscibleReferenceItem(
                factor,
                reference=reference["samples"].clone(),
                blend_function=BLENDING_MODES[blend_mode],
                noise=custom_noise,
                **kwargs,
            )

        return wrapper


class NoiseConditioningNode(metaclass=IntegratedNode):
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": (
                    "CONDITIONING",
                    {"tooltip": "Input conditioning to be noised."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed to use for generated noise.",
                    },
                ),
                "blend": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "tooltip": "Blend strength of noise to be added to conditioning as a percentage where 1.0 would indicate 100%.",
                    },
                ),
                "pooled_blend": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "tooltip": "Blend strength of noise to be added to pooled conditioning as a percentage where 1.0 would indicate 100%.",
                    },
                ),
                "blend_mode": (
                    tuple(BLENDING_MODES.keys()),
                    {
                        "default": "inject",
                        "tooltip": "Blending function used when combining noise with the conditioning. inject just adds it.",
                    },
                ),
                "pooled_blend_mode": (
                    tuple(BLENDING_MODES.keys()),
                    {
                        "default": "inject",
                        "tooltip": "Blending function used when combining noise with the pooled conditioning. inject just adds it.",
                    },
                ),
                "noise_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Strength of the generated noise to be added to conditioning.",
                    },
                ),
                "pooled_noise_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Strength of the generated noise to be added to pooled conditioning.",
                    },
                ),
                "conditioning_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Multiplier applied to conditioning tensors.",
                    },
                ),
                "pooled_conditioning_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Multiplier applied to pooled conditioning tensors.",
                    },
                ),
                "time_mode": (
                    (
                        "relaxed",
                        "strict",
                    ),
                    {
                        "default": "relaxed",
                        "tooltip": "Controls time matching. Strict requires a conditioning item to be fully within the start/end range while relaxed just requires it to have overlap with the range. For example, if the time range is 0.2 through 0.5 and the conditioning item is 0.0 through 0.4 then strict mode would not match.",
                    },
                ),
                "start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Conditioning item start time as a percentage of sampling.",
                    },
                ),
                "end_time": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Conditioning item end time as a percentage of sampling.",
                    },
                ),
                "item_start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Conditioning item start as a percentage of the total number of conditioning items.",
                    },
                ),
                "item_end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Conditioning item end as a percentage of the total number of conditioning items.",
                    },
                ),
                "pooled_item_start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Conditioning pooled output item start as a percentage of the total number of conditioning items.",
                    },
                ),
                "pooled_item_end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Conditioning pooled output item end as a percentage of the total number of conditioning items.",
                    },
                ),
                "slice_start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Noise is generated to match the total size of matched conditioning items. Slices use a percentage of that chunk of noise.",
                    },
                ),
                "slice_end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Noise is generated to match the total size of matched conditioning items. Slices use a percentage of that chunk of noise.",
                    },
                ),
                "pooled_slice_start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Noise is generated to match the total size of matched conditioning items. Slices use a percentage of that chunk of noise.",
                    },
                ),
                "pooled_slice_end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Noise is generated to match the total size of matched conditioning items. Slices use a percentage of that chunk of noise.",
                    },
                ),
                "cpu_noise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether noise will be generated on GPU or CPU. Only affects noise types that support GPU generation.",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether the generated noise is normalized to 1.0 strength before scaling. Generally should be left enabled.",
                    },
                ),
                "fake_channels": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "tooltip": "Noise will be generated with number of channels. Shouldn't make a difference for most noise types.",
                    },
                ),
            },
            "optional": {
                "custom_noise": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": "Custom noise type to use. If not connected, gaussian noise will be used."
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        conditioning,
        seed,
        noise_strength,
        pooled_noise_strength,
        blend,
        pooled_blend,
        conditioning_multiplier,
        pooled_conditioning_multiplier,
        blend_mode,
        pooled_blend_mode,
        start_time,
        end_time,
        item_start,
        item_end,
        pooled_item_start,
        pooled_item_end,
        slice_start,
        slice_end,
        pooled_slice_start,
        pooled_slice_end,
        time_mode,
        cpu_noise,
        normalize,
        fake_channels,
        custom_noise=None,
    ):
        MODULES.initialize()
        blend_function = BLENDING_MODES[blend_mode]
        pblend_function = BLENDING_MODES[pooled_blend_mode]
        noise_spatdim_min = 4 * fake_channels
        size = psize = 0
        count = pcount = 0
        to_noise = []
        for cond, opts, *_ in conditioning:
            stime, etime = opts.get("start_percent", 0.0), opts.get("end_percent", 1.0)
            pooled = opts.get("pooled_output")
            if time_mode == "relaxed":
                time_ok = (start_time <= stime <= end_time) or (
                    start_time <= etime <= end_time
                )
            else:
                time_ok = stime >= start_time and etime <= end_time
            if not time_ok:
                to_noise.append((False, False, False))
                continue
            need_cond = noise_strength != 0 and blend != 0
            if need_cond:
                size += cond.numel()
                count += 1
            need_pooled = (
                pooled_noise_strength != 0 and pooled_blend != 0 and pooled is not None
            )
            if need_pooled:
                psize += pooled.numel()
                pcount += 1
            to_noise.append((True, need_cond, need_pooled))
        conds_size = size + psize
        # print("GOT", size, psize, "-->", conds_size, "::", count, pcount)
        if conds_size != 0:
            noise_spatdim = max(
                noise_spatdim_min, math.ceil((conds_size // fake_channels) ** 0.5)
            )
            empty_ref = torch.zeros(
                1,
                fake_channels,
                noise_spatdim,
                noise_spatdim,
                dtype=torch.float,
                device="cpu" if cpu_noise else get_torch_device(),
            )
            if custom_noise is not None:
                ns = custom_noise.make_noise_sampler(
                    empty_ref,
                    sigma_min=None,
                    sigma_max=None,
                    seed=seed,
                    cpu=cpu_noise,
                    normalized=False,
                )
            else:

                def ns(*_unusedargs, **_unusedkwargs):
                    return torch.randn_like(empty_ref)

            randst = torch.random.get_rng_state()
            try:
                torch.random.manual_seed(seed)
                noise = ns(None, None)
            finally:
                torch.random.set_rng_state(randst)
            noise = noise.reshape(noise.numel())
            noise_conds = noise.new_zeros(size)
            noise_conds[int(size * slice_start) : math.ceil(size * slice_end)] = (
                noise_strength
            )
            noise_conds *= scale_noise(
                noise[:size],
                normalized=normalize,
                normalize_dims=None,
            )

            noise_pooled = noise.new_zeros(psize)
            noise_pooled[
                int(psize * pooled_slice_start) : math.ceil(psize * pooled_slice_end)
            ] = pooled_noise_strength
            noise_pooled *= scale_noise(
                noise[size : size + psize],
                normalized=normalize,
                normalize_dims=None,
            )
            # print("MADE NOISE", noise.shape, noise_conds.shape, noise_pooled.shape)
            del noise

        result = []
        currc = currp = 0
        for (time_matched, need_cond, need_pooled), (cond, opts, *_) in zip(
            to_noise, conditioning
        ):
            # print(">> ITER", currc, currp)
            opts = opts.copy()
            pooled = opts["pooled_output"]
            if time_matched:
                if conditioning_multiplier != 1:
                    cond = cond * conditioning_multiplier
                if pooled is not None and pooled_conditioning_multiplier != 1:
                    pooled = pooled * pooled_conditioning_multiplier
            if need_cond:
                cpct = currc / count
                currc += 1
                if item_start <= cpct <= item_end:
                    # print("COND MATCH", offset, cond.shape)
                    cond = blend_function(
                        cond,
                        noise_conds[: cond.numel()].reshape(cond.shape).to(cond),
                        blend,
                    )
                noise_conds = noise_conds[cond.numel() :]
            if need_pooled:
                ppct = currp / pcount
                currp += 1
                if pooled_item_start <= ppct <= pooled_item_end:
                    # print("POOLED MATCH", offset, pooled.shape)
                    pooled = pblend_function(
                        pooled,
                        noise_pooled[: pooled.numel()].reshape(pooled.shape).to(pooled),
                        pooled_blend,
                    )
                noise_pooled = noise_pooled[pooled.numel() :]
            if pooled is not None:
                opts["pooled_output"] = pooled
            result.append([cond, opts])
        return (result,)


class SamplerNodeConfigOverride(metaclass=IntegratedNode):
    DESCRIPTION = "Allows overriding parameters of a SAMPLER, and specifically lets you use custom and/or Immiscible noise. For use with non-OCS samplers, not recommended to use with the OCS Sampler node as it has internal support for Immiscible noise."

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": (
                    "SAMPLER",
                    {
                        "tooltip": "Sampler to wrap with custom noise handling/parameter overrides."
                    },
                ),
                "immiscible_size": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "tooltip": "Number of batch repeats to use when generating Immiscible noise. Setting this to 0 disables immiscible noise. If the batching type is batch, then Immiscible noise is also disabled unless the size is 2 or higher. Note that this size is in batch repeats regardless of the batching mode. For example, if you are generating a batch of 2 and you set this to 2, then you will generate noise with batch size 4.",
                    },
                ),
                "immiscible_batching": (
                    (
                        "channel",
                        "batch",
                        "row",
                        "column",
                        "frame",
                        "cycle_channel_batch",
                        "cycle_row_column",
                        "cycle_channel_row",
                        "cycle_channel_column",
                    ),
                    {
                        "default": "channel",
                        "tooltip": "Dimension to maximize (or minimize) the noise with. Column mode requires reshaping the input and may require a lot of VRAM. Row mode is also fairly slow, but not as bad as column mode. Row and column modes have a very strong effect.",
                    },
                ),
                "immiscible_reference": (
                    (
                        "cond",
                        "uncond",
                        "denoised",
                        "model_input",
                        "noise_prediction",
                    ),
                    {
                        "default": "cond",
                        "tooltip": "Reference type to use when generating immiscible noise.\ncond - positive prompt.\nuncond - negative prompt.\ndenoised - the model's prediction of a clean image (using both cond and uncond).\nmodel_input - noisy latent image the model was called with (often referred to as x).\nnoise_prediction - model_input with denoised subtracted (leaving just what the model thinks is noise).",
                    },
                ),
                "immiscible_normalize_ref_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "tooltip": "Controls whether the reference gets normalized. If set to 0, no normalization is done.",
                    },
                ),
                "immiscible_normalize_noise_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "tooltip": "Controls whether the noise used as an input for immiscible noise is gets normalized first. If set to 0, no normalization is done.",
                    },
                ),
                "immiscible_maximize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, maximizes the distance between the noise and the reference rather than trying to minimize it.",
                    },
                ),
                "immiscible_distance_scale": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "tooltip": "Multiplier on the input noise for v2 Immiscible noise. Set to 0 to use v1 Immiscible noise.",
                    },
                ),
                "immiscible_distance_scale_ref": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "tooltip": "Multiplier on the refence for v2 Immiscible noise. No effect if immiscible_distance_scale is 0.",
                    },
                ),
                "immiscible_blend": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "tooltip": "Percentage of immiscible noise to use. 1.0 means 100%. May not work very well with most blend modes.",
                    },
                ),
                "immiscible_blend_mode": (
                    tuple(BLENDING_MODES.keys()),
                    {
                        "default": "lerp",
                        "tooltip": "Blending function used when mixing immiscible noise with normal noise. Only slerp seems to work well (requires ComfyUI-bleh).",
                    },
                ),
                "immiscible_start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Start time as a percentage of sampling where immiscible noise will be used.",
                    },
                ),
                "immiscible_end_time": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "End time as a percentage of sampling where immiscible noise will be used.",
                    },
                ),
                "noise_start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Start time as a percentage of sampling where custom noise will be used.",
                    },
                ),
                "noise_end_time": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "End time as a percentage of sampling where custom noise will be used.",
                    },
                ),
                "cpu_noise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether noise is generated on CPU or GPU. Only affects custom noise.",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Controls whether generated noise is normalized to 1.0 strength. This normalization occurs last.",
                    },
                ),
            },
            "optional": {
                "custom_noise_opt": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": "Optional input for custom noise used during ancestral or SDE sampling.",
                    },
                ),
                "yaml_parameters": (
                    "STRING",
                    {
                        "tooltip": "Allows specifying custom parameters via YAML. This input can be converted to a multiline text widget. Note: When specifying parameters this way, there is no error checking.",
                        "placeholder": "# YAML or JSON here",
                        "dynamicPrompts": False,
                        "multiline": True,
                        "defaultInput": True,
                    },
                ),
            },
        }

    def get_sampler(
        self,
        *,
        sampler,
        immiscible_size,
        immiscible_batching,
        immiscible_reference,
        immiscible_normalize_ref_scale,
        immiscible_normalize_noise_scale,
        immiscible_maximize,
        immiscible_distance_scale,
        immiscible_distance_scale_ref,
        immiscible_start_time,
        immiscible_end_time,
        immiscible_blend,
        immiscible_blend_mode,
        noise_start_time,
        noise_end_time,
        cpu_noise=True,
        custom_noise_opt=None,
        normalize=True,
        yaml_parameters="",
    ):
        MODULES.initialize()
        sampler_kwargs = {}
        if yaml_parameters:
            extra_params = yaml.safe_load(yaml_parameters)
            if extra_params is None:
                pass
            elif not isinstance(extra_params, dict):
                raise ValueError(
                    "SamplerConfigOverride: yaml_parameters must either be null or an object",
                )
            else:
                sampler_kwargs |= extra_params
        return (
            comfy.samplers.KSAMPLER(
                self.sampler_function,
                extra_options=sampler.extra_options
                | {
                    "ocs_override_sampler_cfg": {
                        "sampler": sampler,
                        "immiscible": {
                            "size": immiscible_size,
                            "batching": immiscible_batching,
                            "reference": immiscible_reference,
                            "norm_ref_scale": immiscible_normalize_ref_scale,
                            "norm_noise_scale": immiscible_normalize_noise_scale,
                            "maximize": immiscible_maximize,
                            "distance_scale": immiscible_distance_scale,
                            "distance_scale_ref": immiscible_distance_scale_ref,
                            "start_time": immiscible_start_time,
                            "end_time": immiscible_end_time,
                            "blend": immiscible_blend,
                            "blend_mode": immiscible_blend_mode,
                        },
                        "noise_start_time": noise_start_time,
                        "noise_end_time": noise_end_time,
                        "custom_noise": custom_noise_opt,
                        "sampler_kwargs": sampler_kwargs,
                        "cpu_noise": cpu_noise,
                        "normalize": normalize,
                    },
                },
                inpaint_options=sampler.inpaint_options.copy(),
            ),
        )

    @classmethod
    @torch.no_grad()
    def sampler_function(
        cls,
        model,
        x,
        sigmas,
        *args: list,
        ocs_override_sampler_cfg: dict[str] | None = None,
        noise_sampler=None,
        extra_args: dict[str] | None = None,
        **kwargs: dict[str],
    ):
        cfg = ocs_override_sampler_cfg
        if cfg is None:
            raise ValueError("Override sampler config missing!")
        if extra_args is None:
            extra_args = {}
        (
            sampler,
            sampler_kwargs,
            custom_noise,
            cpu,
            normalize,
            noise_start_time,
            noise_end_time,
        ) = (
            cfg[k]
            for k in (
                "sampler",
                "sampler_kwargs",
                "custom_noise",
                "cpu_noise",
                "normalize",
                "noise_start_time",
                "noise_end_time",
            )
        )
        icfg = cfg["immiscible"]
        (
            isize,
            ibatching,
            ireference,
            inorm_ref_scale,
            inorm_noise_scale,
            imaximize,
            idistance_scale,
            idistance_scale_ref,
            istart_time,
            iend_time,
            iblend,
            iblend_mode,
        ) = (
            icfg[k]
            for k in (
                "size",
                "batching",
                "reference",
                "norm_ref_scale",
                "norm_noise_scale",
                "maximize",
                "distance_scale",
                "distance_scale_ref",
                "start_time",
                "end_time",
                "blend",
                "blend_mode",
            )
        )
        sig = inspect.signature(sampler.sampler_function)
        params = sig.parameters
        kwargs |= {k: v for k, v in sampler_kwargs.items() if k in params}
        if "noise_sampler" in params:
            model_sampling = model.inner_model.inner_model.model_sampling
            orig_noise_sampler = kwargs.pop(
                "noise_sampler", lambda *_args, **_kwargs: torch.randn_like(x)
            )
            if custom_noise is not None and noise_start_time < 1 and noise_end_time > 0:
                seed = extra_args.get("seed")
                sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
                custom_noise_sampler = custom_noise.make_noise_sampler(
                    x,
                    sigma_min,
                    sigma_max,
                    seed=seed,
                    cpu=cpu,
                    normalized=False,
                )
            else:
                custom_noise_sampler = None
            if custom_noise_sampler is not None:
                sigma_start = model_sampling.percent_to_sigma(noise_start_time)
                sigma_end = model_sampling.percent_to_sigma(noise_end_time)

                def override_noise_sampler(s, sn, *args, **kwargs):
                    if not sigma_end <= s.max() <= sigma_start:
                        return orig_noise_sampler(s, sn, *args, **kwargs)
                    return custom_noise_sampler(s, sn, *args, **kwargs)

            else:
                override_noise_sampler = orig_noise_sampler
            if (
                istart_time < 1
                and iend_time > 0
                and (isize > 1 if ibatching == "batch" else isize > 0)
                and iblend != 0
            ):
                isigma_start = model_sampling.percent_to_sigma(istart_time)
                isigma_end = model_sampling.percent_to_sigma(iend_time)
                if ibatching.startswith("cycle_"):
                    ibatching = ibatching.split("_")[1:]
                immiscible = ImmiscibleNoise(
                    size=isize,
                    batching=ibatching if isinstance(ibatching, str) else "channel",
                    maximize=imaximize,
                    distance_scale=idistance_scale,
                    distance_scale_ref=idistance_scale_ref,
                )
                blend_function = BLENDING_MODES[iblend_mode]

                ref_latent = None
                ref_handlers = {
                    "cond": "cond_denoised",
                    "uncond": "uncond_denoised",
                    "denoised": "denoised",
                    "model_input": "input",
                    "noise_prediction": lambda args: args["input"] - args["denoised"],
                }
                if (ref_handler := ref_handlers.get(ireference)) is None:
                    raise ValueError("Bad immiscible reference type")

                def postcfg(args):
                    nonlocal ref_latent
                    ref_latent = (
                        args.get(ref_handler)
                        if isinstance(ref_handler, str)
                        else ref_handler(args)
                    )
                    return args["denoised"]

                extra_args = extra_args | {
                    "model_options": set_model_options_post_cfg_function(
                        extra_args.get("model_options", {}).copy(),
                        postcfg,
                        disable_cfg1_optimization=ireference == "uncond",
                    )
                }

                immiscible_counter = 0

                def noise_sampler(s, sn, *args, **kwargs):
                    nonlocal ref_latent, immiscible_counter
                    if not isigma_end <= s.max() <= isigma_start:
                        return override_noise_sampler(s, sn, *args, **kwargs)
                    if ref_latent is None:
                        raise ValueError("Immiscible reference type not available")
                    if not isinstance(ibatching, str):
                        immiscible.batching = ibatching[
                            immiscible_counter % len(ibatching)
                        ]
                    immiscible_counter += 1
                    blending = iblend != 1
                    if inorm_ref_scale != 0:
                        ref_latent = scale_noise(
                            ref_latent, inorm_ref_scale, normalized=True
                        )
                    batch_size = ref_latent.shape[0]
                    noise_batch = torch.cat(
                        tuple(
                            override_noise_sampler(s, sn)
                            for _ in range(max(1, isize) + int(blending))
                        )
                    )
                    immiscible_noise = immiscible.unbatch(
                        immiscible.immiscible(
                            immiscible.batch(
                                scale_noise(
                                    noise_batch[batch_size * int(blending) :],
                                    1.0
                                    if inorm_noise_scale == 0
                                    else inorm_noise_scale,
                                    normalized=inorm_noise_scale != 0,
                                )
                            ),
                            immiscible.batch(ref_latent),
                        ),
                        ref_latent.shape,
                    )
                    immiscible_noise = scale_noise(
                        immiscible_noise, normalized=normalize
                    )
                    if iblend != 1:
                        immiscible_noise = blend_function(
                            noise_batch[: batch_size * int(blending)],
                            immiscible_noise,
                            iblend,
                        )
                    return scale_noise(immiscible_noise, normalized=normalize)

            else:
                if not normalize:
                    noise_sampler = override_noise_sampler
                else:

                    def noise_sampler(s, sn, *args, **kwargs):
                        return scale_noise(
                            override_noise_sampler(s, sn, *args, **kwargs),
                            normalized=True,
                        )

            kwargs["noise_sampler"] = noise_sampler
        return sampler.sampler_function(
            model,
            x,
            sigmas,
            *args,
            extra_args=extra_args,
            **kwargs,
        )
