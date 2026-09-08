from __future__ import annotations

import functools
import inspect
import math
from collections.abc import Callable
from typing import Any, NamedTuple

import comfy.samplers
import torch
import yaml
from comfy.model_management import get_torch_device
from comfy.model_patcher import set_model_options_post_cfg_function
from tqdm import tqdm

from .. import filtering
from ..external import MODULES, IntegratedNode
from ..nodes import NOISE_INPUT_TYPES_HINT, WILDCARD_NOISE
from ..noise import ImmiscibleNoise
from ..utils import scale_noise
from .base import CustomNoiseItemBase, CustomNoiseNodeBase, NormalizeNoiseNodeMixin
from .noise_immiscibleref import ImmiscibleReferenceItem
from .noise_perlin import Perlin, PerlinItem

PERLIN_DEFAULTS = Perlin()


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
        blend_modes = tuple(filtering.BLENDING_MODES.keys())
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
                    "min": -10000.0,
                    "max": 10000.0,
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
                    "min": -10000.0,
                    "max": 10000.0,
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
                    "min": -99999,
                    "max": 99999,
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
                blend_modes,
                {
                    "default": PERLIN_DEFAULTS.blend.name,
                    "tooltip": "Blending function used when generating Perlin noise. When set to values other than LERP may not work at all or may not actually generate Perlin noise.",
                },
            ),
            "pattern_break_blend": (
                blend_modes,
                {
                    "default": PERLIN_DEFAULTS.pattern_break_blend.name,
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
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Controls the amplitude for the first octave.",
                },
            ),
            "initial_frequency_height": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.initial_frequency[0],
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Controls the frequency for the first octave for the this axis.",
                },
            ),
            "initial_frequency_width": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.initial_frequency[1],
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Controls the frequency for the first octave for the this axis.",
                },
            ),
            "initial_frequency_depth": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.initial_frequency[2],
                    "min": -10000.0,
                    "max": 10000.0,
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
        opts = result.get("optional", {})
        opts |= {
            "ridge_weight": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.ridge_weight,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Blend strength for blending in the ridge-adjusted noise.",
                },
            ),
            "ridge_scale": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.ridge_scale,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Controls the amplitude of generated ridges.",
                },
            ),
            "ridge_blend": (
                blend_modes,
                {
                    "default": PERLIN_DEFAULTS.ridge_blend.name,
                    "tooltip": "Blend mode used for blending in the ridge-adjusted noise.",
                },
            ),
            "warp_strength": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.warp_strength,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Strength of domain warping. Shifts coordinates of later octaves using the outputs from previous octaves.",
                },
            ),
            "octave_shift": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.octave_shift,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Shifts gradient dimensions (height, width, depth if enabled) across octaves. Shift is multiplied by the 1-based octave. I.E. shift 0.75 at the first octave is 1*0.75 and this is rounded to the nearest integer value. You can use this to only shift every other octave, etc. The shift can be negative to roll in the other direction.",
                },
            ),
            "curl_strength": (
                "FLOAT",
                {
                    "default": PERLIN_DEFAULTS.curl_strength,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Twists the gradients with additional Gaussian noise.",
                },
            ),
            "curl_dims": (
                "STRING",
                {
                    "default": PERLIN_DEFAULTS.get_commasep("curl_dims"),
                    "tooltip": "Comma-separated axes to apply the curl effect to. Only does something when curl_strength is non-zero. In 3D mode the axes would be 0 (depth), 1 (height), 2 (width). In 2D mode you'd have 0 (height) and 1 (depth).",
                },
            ),
            "base_noise_opt": (
                WILDCARD_NOISE,
                {
                    "tooltip": f"Optional input for noise to use as the Perlin base.\n{NOISE_INPUT_TYPES_HINT}",
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
        orig_opts = result["optional"]
        reqs = {k: v for k, v in orig_reqs.items() if k in cls._COPY_KEYS}
        reqs["lacunarity"] = orig_reqs["lacunarity_height"]
        reqs["res"] = orig_reqs["res_height"]
        result["required"] = reqs
        result["optional"] = (
            {"ocs_noise_opt": orig_opts["ocs_noise_opt"]}
            if "ocs_noise_opt" in orig_opts
            else {}
        )
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
                    "row_plus_column",
                    "channel_plus_row",
                    "channel_plus_column",
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
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Controls whether the reference gets normalized. If set to 0, no normalization is done.",
                },
            ),
            "normalize_noise_scale": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": -10000.0,
                    "max": 10000.0,
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
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Multiplier on the input noise for v2 Immiscible noise. Set to 0 to use v1 Immiscible noise.",
                },
            ),
            "distance_scale_ref": (
                "FLOAT",
                {
                    "default": 0.1,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Multiplier on the refence for v2 Immiscible noise. No effect if distance_scale is 0.",
                },
            ),
            "blend": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "tooltip": "Percentage of immiscible noise to use. 1.0 means 100%. May not work very well with most blend modes.",
                },
            ),
            "blend_mode": (
                tuple(filtering.BLENDING_MODES.keys()),
                {
                    "default": "lerp",
                    "tooltip": "Blending function used when mixing immiscible noise with normal noise. Only slerp seems to work well (requires ComfyUI-bleh).",
                },
            ),
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
            "custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": "Input for custom noise used during ancestral or SDE sampling.",
                },
            ),
        }
        result["optional"] = {
            "reference": (
                "LATENT",
                {
                    "tooltip": "Attach either this or the custom_noise_ref input but not both.",
                },
            ),
            "custom_noise_ref": (
                WILDCARD_NOISE,
                {
                    "tooltip": "Optional input that can be attached instead of the reference latent. When used, noise from this generator will be used as the reference.",
                },
            ),
            "custom_noise_blend": (
                WILDCARD_NOISE,
                {
                    "tooltip": "Optional input for blended noise (only used when blend is not 1.0). Can be used if you want to blend with a different noise type.",
                },
            ),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return ImmiscibleReferenceItem

    def go(
        self,
        *,
        factor: float,
        size: int,
        batching: str,
        normalize_ref_scale: float,
        normalize_noise_scale: float,
        maximize: bool,
        distance_scale: float,
        distance_scale_ref: float,
        blend: float,
        blend_mode: str,
        normalize: bool | None,
        custom_noise: object,
        reference: dict | None = None,
        custom_noise_ref: object | None = None,
        custom_noise_blend: object | None = None,
    ) -> tuple:
        return super().go(
            factor,
            size=size,
            batching=batching,
            normalize_ref_scale=normalize_ref_scale,
            normalize_noise_scale=normalize_noise_scale,
            maximize=maximize,
            distance_scale=distance_scale,
            distance_scale_ref=distance_scale_ref,
            blend=blend,
            blend_function=filtering.BLENDING_MODES[blend_mode],
            normalize=self.get_normalize(normalize),
            noise=custom_noise.clone(),
            reference=reference["samples"].clone() if reference is not None else None,
            custom_noise_ref=custom_noise_ref.clone()
            if custom_noise_ref is not None
            else None,
            custom_noise_blend=custom_noise_blend.clone()
            if custom_noise_blend is not None
            else None,
        )


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
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Blend strength of noise to be added to conditioning as a percentage where 1.0 would indicate 100%.",
                    },
                ),
                "pooled_blend": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Blend strength of noise to be added to pooled conditioning as a percentage where 1.0 would indicate 100%.",
                    },
                ),
                "blend_mode": (
                    tuple(filtering.BLENDING_MODES.keys()),
                    {
                        "default": "inject",
                        "tooltip": "Blending function used when combining noise with the conditioning. inject just adds it.",
                    },
                ),
                "pooled_blend_mode": (
                    tuple(filtering.BLENDING_MODES.keys()),
                    {
                        "default": "inject",
                        "tooltip": "Blending function used when combining noise with the pooled conditioning. inject just adds it.",
                    },
                ),
                "noise_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Strength of the generated noise to be added to conditioning.",
                    },
                ),
                "pooled_noise_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Strength of the generated noise to be added to pooled conditioning.",
                    },
                ),
                "conditioning_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Multiplier applied to conditioning tensors.",
                    },
                ),
                "pooled_conditioning_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10000.0,
                        "max": 10000.0,
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
        blend_function = filtering.BLENDING_MODES[blend_mode]
        pblend_function = filtering.BLENDING_MODES[pooled_blend_mode]
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


class ImmiscibleConfig(NamedTuple):
    size: int = 64
    batching: str = "channel"
    reference: str = "cond"
    norm_ref_scale: float = 0.0
    norm_noise_scale: float = 0.0
    maximize: bool = False
    distance_scale: float = 0.1
    distance_scale_ref: float = 0.1
    use_triton: bool = True
    abs_mode: bool = False
    abs_distance_mode: bool = False
    shuffle_seed: int | None = None
    start_time: float = 0.0
    end_time: float = 1.0
    blend: float = 1.0
    blend_mode: str = "lerp"
    force: bool = False
    operation_reference: Callable | None = None
    operation_noise: Callable | None = None
    operation_result: Callable | None = None
    filter_reference: filtering.Filter | None = None
    filter_noise: filtering.Filter | None = None
    filter_result: filtering.Filter | None = None
    filter_postcfg: filtering.Filter | None = None


DEFAULT_IMMISCIBLE_CONFIG = ImmiscibleConfig()


class OverrideSamplerConfig(NamedTuple):
    verbose: bool = False
    sampler: object | None = None
    sampler_kwargs: dict | None = None
    noise_start_time: float = 0.0
    noise_end_time: float = 1.0
    cpu_noise: bool = False
    normalize: bool = True
    force_params: list | tuple = ()
    custom_noise: object | None = None
    custom_noise_ref: object | None = None
    custom_noise_blend: object | None = None
    latent_ref: torch.Tensor | None = None
    immiscible: ImmiscibleConfig = DEFAULT_IMMISCIBLE_CONFIG


DEFAULT_OVERRIDESAMPLER_CONFIG = OverrideSamplerConfig()


class SamplerNodeConfigOverride(metaclass=IntegratedNode):
    DESCRIPTION = "Allows overriding parameters of a SAMPLER, and specifically lets you use custom and/or Immiscible noise. For use with non-OCS samplers, not recommended to use with the OCS Sampler node as it has internal support for Immiscible noise."

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        DICFG = DEFAULT_IMMISCIBLE_CONFIG
        DOCFG = DEFAULT_OVERRIDESAMPLER_CONFIG
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
                        "default": DICFG.size,
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
                        "default": DICFG.batching,
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
                        "uncond_sub_cond",
                        "cond_sub_uncond",
                        "denoised_sub_uncond",
                        "latent",
                        "noise",
                        "initial_latent",
                        "noise_gen_prev",
                    ),
                    {
                        "default": DICFG.reference,
                        "tooltip": "Reference type to use when generating immiscible noise.\ncond - positive prompt.\nuncond - negative prompt.\ndenoised - the model's prediction of a clean image (using both cond and uncond).\nmodel_input - noisy latent image the model was called with (often referred to as x).\nnoise_prediction - model_input with denoised subtracted (leaving just what the model thinks is noise).",
                    },
                ),
                "immiscible_normalize_ref_scale": (
                    "FLOAT",
                    {
                        "default": DICFG.norm_ref_scale,
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Controls whether the reference gets normalized. If set to 0, no normalization is done.",
                    },
                ),
                "immiscible_normalize_noise_scale": (
                    "FLOAT",
                    {
                        "default": DICFG.norm_noise_scale,
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Controls whether the noise used as an input for immiscible noise is gets normalized first. If set to 0, no normalization is done.",
                    },
                ),
                "immiscible_maximize": (
                    "BOOLEAN",
                    {
                        "default": DICFG.maximize,
                        "tooltip": "When enabled, maximizes the distance between the noise and the reference rather than trying to minimize it.",
                    },
                ),
                "immiscible_distance_scale": (
                    "FLOAT",
                    {
                        "default": DICFG.distance_scale,
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Multiplier on the input noise for v2 Immiscible noise. Set to 0 to use v1 Immiscible noise.",
                    },
                ),
                "immiscible_distance_scale_ref": (
                    "FLOAT",
                    {
                        "default": DICFG.distance_scale_ref,
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Multiplier on the refence for v2 Immiscible noise. No effect if immiscible_distance_scale is 0.",
                    },
                ),
                "immiscible_blend": (
                    "FLOAT",
                    {
                        "default": DICFG.blend,
                        "min": -10000.0,
                        "max": 10000.0,
                        "tooltip": "Percentage of immiscible noise to use. 1.0 means 100%. May not work very well with most blend modes.",
                    },
                ),
                "immiscible_blend_mode": (
                    tuple(filtering.BLENDING_MODES.keys()),
                    {
                        "default": DICFG.blend_mode,
                        "tooltip": "Blending function used when mixing immiscible noise with normal noise. Only slerp seems to work well (requires ComfyUI-bleh).",
                    },
                ),
                "immiscible_start_time": (
                    "FLOAT",
                    {
                        "default": DICFG.start_time,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Start time as a percentage of sampling where immiscible noise will be used.",
                    },
                ),
                "immiscible_end_time": (
                    "FLOAT",
                    {
                        "default": DICFG.end_time,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "End time as a percentage of sampling where immiscible noise will be used.",
                    },
                ),
                "noise_start_time": (
                    "FLOAT",
                    {
                        "default": DOCFG.noise_start_time,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Start time as a percentage of sampling where custom noise will be used.",
                    },
                ),
                "noise_end_time": (
                    "FLOAT",
                    {
                        "default": DOCFG.noise_end_time,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "End time as a percentage of sampling where custom noise will be used.",
                    },
                ),
                "cpu_noise": (
                    "BOOLEAN",
                    {
                        "default": DOCFG.cpu_noise,
                        "tooltip": "Controls whether noise is generated on CPU or GPU. Only affects custom noise.",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": DOCFG.normalize,
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
                "custom_noise_ref": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": "Required when reference mode is reference_noise, otherwise unused.",
                    },
                ),
                "latent_ref": (
                    "LATENT",
                    {
                        "tooltip": "Required when the reference mode is reference_latent, otherwise unused. This doesn't respect stuff like latent from batch that may have been previously applied and must match the shape of the latent being generated.",
                    },
                ),
                "custom_noise_blend": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": "Optional input for blended noise (only used when blend is not 1.0). Can be used if you want to blend with a different noise type.",
                    },
                ),
                "operation_reference": ("LATENT_OPERATION",),
                "operation_noise": ("LATENT_OPERATION",),
                "operation_result": ("LATENT_OPERATION",),
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
        normalize=True,
        yaml_parameters="",
        custom_noise_opt=None,
        custom_noise_ref=None,
        custom_noise_blend=None,
        latent_ref: dict | None = None,
        operation_reference: Callable | None = None,
        operation_noise: Callable | None = None,
        operation_result: Callable | None = None,
    ):
        MODULES.initialize()
        sampler_kwargs = {}
        overridecfg_kwargs = {
            "noise_start_time": noise_start_time,
            "noise_end_time": noise_end_time,
            "cpu_noise": cpu_noise,
            "normalize": normalize,
        }
        immisciblecfg_kwargs = {
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
            "operation_reference": operation_reference,
            "operation_noise": operation_noise,
            "operation_result": operation_result,
        }
        ocs_verbose = False
        ocs_force_params = ()
        if yaml_parameters:
            extra_params = yaml.safe_load(yaml_parameters)
            if extra_params is None:
                pass
            elif not isinstance(extra_params, dict):
                raise ValueError(
                    "SamplerConfigOverride: yaml_parameters must either be null or an object",
                )
            else:
                ocs_extra = extra_params.pop("ocs", {})
                ocs_verbose = bool(ocs_extra.pop("verbose", False))
                ocs_force_params = tuple(ocs_extra.pop("force_params", ()) or ())
                override_extra = ocs_extra.pop("override", {})
                immiscible_extra = ocs_extra.pop("immiscible", {})
                overridecfg_kwargs |= override_extra
                filterdefs = immiscible_extra.pop("filters", {})
                immisciblecfg_kwargs |= {
                    f"filter_{k}": filtering.make_filter(filterdefs[k])
                    for k in ("reference", "noise", "result", "postcfg")
                    if k in filterdefs
                }
                if "filter_reference" in immisciblecfg_kwargs:
                    immisciblecfg_kwargs["reference"] = "expression"
                immisciblecfg_kwargs |= immiscible_extra
                sampler_kwargs |= extra_params
        if latent_ref is not None:
            overridecfg_kwargs["latent_ref"] = latent_ref["samples"].to(
                dtype=torch.float32, device="cpu", copy=True
            )
        if immiscible_reference == "latent":
            if latent_ref is None:
                raise ValueError(
                    "latent_ref input must be connected when reference mode is latent"
                )
        elif immiscible_reference == "noise":
            if custom_noise_ref is None:
                raise ValueError(
                    "custom_noise_ref input must be connected when reference mode is noise"
                )
            overridecfg_kwargs["custom_noise_ref"] = custom_noise_ref.clone()

        sampler_function = functools.partial(
            self.sampler_function,
            ocs_override_sampler_cfg=OverrideSamplerConfig(
                verbose=ocs_verbose,
                force_params=ocs_force_params,
                sampler=sampler,
                sampler_kwargs=sampler_kwargs,
                immiscible=ImmiscibleConfig(**immisciblecfg_kwargs),
                custom_noise=custom_noise_opt.clone()
                if custom_noise_opt is not None
                else None,
                custom_noise_blend=custom_noise_blend.clone()
                if custom_noise_blend is not None
                else None,
                **overridecfg_kwargs,
            ),
        )
        functools.update_wrapper(sampler_function, sampler.sampler_function)
        return (
            comfy.samplers.KSAMPLER(
                sampler_function,
                extra_options=sampler.extra_options.copy(),
                inpaint_options=sampler.inpaint_options.copy(),
            ),
        )

    @staticmethod
    @torch.no_grad()
    def sampler_function(
        model,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        *args: list,
        ocs_override_sampler_cfg: dict[str] | None = None,
        noise_sampler=None,
        extra_args: dict[str] | None = None,
        **kwargs: dict[str],
    ) -> torch.Tensor:
        cfg = ocs_override_sampler_cfg
        if cfg is None:
            raise ValueError("Override sampler config missing!")
        icfg = cfg.immiscible
        if cfg.verbose:
            tqdm.write(f"* OCS: Using immiscible config: {icfg}")
        if extra_args is None:
            extra_args = {}
        sig = inspect.signature(cfg.sampler.sampler_function)
        params = frozenset(cfg.force_params) | sig.parameters.keys()
        kwargs |= {k: v for k, v in cfg.sampler_kwargs.items() if k in params}
        if "noise_sampler" not in params and not icfg.force:
            return cfg.sampler.sampler_function(
                model,
                x,
                sigmas,
                *args,
                extra_args=extra_args,
                **kwargs,
            )
        requires_uncond_modes = {
            "uncond",
            "cond_sub_uncond",
            "denoised_sub_uncond",
        }
        requires_prev_modes = {
            "model_input_prev",
            "model_input_sub_model_input_prev",
            "noise_gen_prev",
            "noise_sub_noise_prev",
            "cond_prev",
            "uncond_prev",
            "denoised_prev",
            "denoised_sub_denoised_prev",
        }
        args_prev: dict | None = None
        noise_prev = None
        x_orig = x.clone()
        seed = extra_args.get("seed")
        seed_gen = torch.Generator(device="cpu" if cfg.cpu_noise else x.device)
        seed_gen.manual_seed(seed if seed is not None else 0)
        if icfg.use_triton and icfg.shuffle_seed is not None:
            shuffle_gen = torch.Generator(device="cpu" if cfg.cpu_noise else x.device)
            shuffle_gen.manual_seed(icfg.shuffle_seed)
        else:
            shuffle_gen = None
        model_sampling = model.inner_model.inner_model.model_sampling
        orig_noise_sampler = kwargs.pop(
            "noise_sampler", lambda *_args, **_kwargs: torch.randn_like(x)
        )
        if (
            cfg.custom_noise is not None
            and cfg.noise_start_time < 1
            and cfg.noise_end_time > 0
        ):
            sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
            custom_noise_sampler = cfg.custom_noise.make_noise_sampler(
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cfg.cpu_noise,
                normalized=False,
            )
        else:
            custom_noise_sampler = None
        sigma_start = model_sampling.percent_to_sigma(cfg.noise_start_time)
        sigma_end = model_sampling.percent_to_sigma(cfg.noise_end_time)

        def override_noise_sampler(s, sn, *args, **kwargs):
            nonlocal noise_prev
            if custom_noise_sampler is None or not sigma_end <= s.max() <= sigma_start:
                noise = orig_noise_sampler(s, sn, *args, **kwargs)
            else:
                noise = custom_noise_sampler(s, sn, *args, **kwargs)
            noise_prev = noise.clone()
            return noise

        def fallback_noise_sampler(
            s: torch.Tensor, sn: torch.Tensor, *args: list, **kwargs: dict
        ) -> torch.Tensor:
            nonlocal noise_prev
            noise = scale_noise(
                override_noise_sampler(s, sn, *args, **kwargs),
                normalized=cfg.normalize,
            )
            noise_prev = noise.clone()
            return noise

        if icfg.force or (
            icfg.start_time < 1
            and icfg.end_time > 0
            and (icfg.size > 1 if icfg.batching == "batch" else icfg.size > 0)
            and (icfg.blend_mode != "lerp" or icfg.blend != 0)
        ):
            isigma_start = model_sampling.percent_to_sigma(icfg.start_time)
            isigma_end = model_sampling.percent_to_sigma(icfg.end_time)
            if icfg.batching.startswith("cycle_"):
                ibatching = icfg.batching.split("_")[1:]
            else:
                ibatching = icfg.batching
            immiscible = ImmiscibleNoise(
                size=icfg.size,
                batching=ibatching if isinstance(ibatching, str) else "channel",
                maximize=icfg.maximize,
                distance_scale=icfg.distance_scale,
                distance_scale_ref=icfg.distance_scale_ref,
                abs_mode=icfg.abs_mode,
                abs_distance_mode=icfg.abs_distance_mode,
                use_triton=icfg.use_triton,
                generator=shuffle_gen,
            )
            blend_function = filtering.BLENDING_MODES[icfg.blend_mode]

            ref_latent = None

            requires_patch = icfg.filter_reference or icfg.reference not in {
                "latent",
                "noise",
                "noise_gen_prev",
                "initial_latent",
            }

            filter_refs = None
            using_filters = (
                icfg.filter_reference
                or icfg.filter_noise
                or icfg.filter_result
                or icfg.filter_postcfg
            )

            immiscible_counter = 0

            if requires_patch:
                requires_uncond = (
                    using_filters
                    or icfg.filter_noise
                    or icfg.filter_result
                    or icfg.reference in requires_uncond_modes
                )

                def ng_prev_handler(*args, **kwargs):
                    nonlocal noise_prev
                    return noise_prev

                ref_handlers = {
                    "initial_latent": x_orig,
                    "noise_gen_prev": ng_prev_handler,
                    "cond": "cond_denoised",
                    "uncond": "uncond_denoised",
                    "denoised": "denoised",
                    "model_input": "input",
                    "noise_prediction": lambda args: args["input"] - args["denoised"],
                    "cond_sub_uncond": lambda args: (
                        args["cond_denoised"] - args["uncond_denoised"]
                    ),
                    "uncond_sub_cond": lambda args: (
                        args["uncond_denoised"] - args["cond_denoised"]
                    ),
                    "denoised_sub_uncond": lambda args: (
                        args["denoised"] - args["uncond_denoised"]
                    ),
                }
                if (
                    icfg.filter_reference is None
                    and (ref_handler := ref_handlers.get(icfg.reference)) is None
                ):
                    raise ValueError("Bad immiscible reference type")

                def postcfg(args: dict) -> torch.Tensor:
                    nonlocal ref_latent, filter_refs, immiscible_counter

                    denoised = args["denoised"]
                    if using_filters:
                        uncond = args.get("uncond_denoised")
                        filter_refs = filtering.FilterRefs(
                            kvs={
                                "immiscible_counter": immiscible_counter,
                                "sigmas": sigmas.clone(),
                                "model_sigma": args["sigma"].clone(),
                                "model_sigma_float": args["sigma"].max().item(),
                                "x": args["input"].clone(),
                                "denoised": denoised.clone(),
                                "cond": args["cond_denoised"].clone(),
                                "uncond": uncond.clone()
                                if uncond is not None
                                else None,
                                "latent_ref": None
                                if cfg.latent_ref is None
                                else cfg.latent_ref.to(
                                    device=denoised.device,
                                    dtype=denoised.dtype,
                                    copy=True,
                                ),
                            }
                        )
                    if icfg.filter_reference:
                        ref_latent = icfg.filter_reference.apply(
                            denoised, refs=filter_refs
                        )
                    else:
                        ref_latent = (
                            args.get(ref_handler)
                            if isinstance(ref_handler, str)
                            else ref_handler(args)
                        )
                    if icfg.filter_postcfg:
                        return icfg.filter_postcfg.apply(denoised, refs=filter_refs)
                    return denoised

                extra_args = extra_args | {
                    "model_options": set_model_options_post_cfg_function(
                        extra_args.get("model_options", {}).copy(),
                        postcfg,
                        disable_cfg1_optimization=requires_uncond,
                    )
                }

            if icfg.reference == "noise":
                ns_ref = cfg.custom_noise_ref.make_noise_sampler(
                    x,
                    sigma_min,
                    sigma_max,
                    seed=torch.randint(
                        0,
                        1 << 32,
                        (1,),
                        device="cpu" if cfg.cpu_noise else x.device,
                        dtype=torch.int64,
                        generator=seed_gen,
                    )
                    .detach()
                    .cpu()
                    .item(),
                    cpu=cfg.cpu_noise,
                    normalized=False,
                )
            elif icfg.reference == "latent":
                ref_latent = cfg.latent_ref.to(
                    dtype=x.dtype, device=x.device, copy=True
                )
                if icfg.norm_ref_scale != 0:
                    ref_latent = scale_noise(
                        ref_latent, icfg.norm_ref_scale, normalized=True
                    )
                if ref_latent.shape[1:] != x.shape[1:]:
                    raise ValueError(
                        "Reference latent shape must match shape of generation with exception that batch size may be 1",
                    )
                if ref_latent.shape[0] != x.shape[0]:
                    if ref_latent.shape[0] == 1:
                        ref_latent = ref_latent.expand(x.shape)
                    else:
                        raise ValueError(
                            "Reference latent batch size must be either 1 or equal to generation batch size"
                        )

            if icfg.blend != 1 and cfg.custom_noise_blend is not None:
                ns_blend = cfg.custom_noise_blend.make_noise_sampler(
                    x,
                    sigma_min,
                    sigma_max,
                    seed=torch.randint(
                        0,
                        1 << 32,
                        (1,),
                        device="cpu" if cfg.cpu_noise else x.device,
                        dtype=torch.int64,
                        generator=seed_gen,
                    )
                    .detach()
                    .cpu()
                    .item(),
                    cpu=cfg.cpu_noise,
                    normalized=False,
                )
            else:
                ns_blend = None

            def noise_sampler(
                s: torch.Tensor,
                sn: torch.Tensor,
                *args: list,
                **kwargs: dict,
            ) -> torch.Tensor:
                nonlocal ref_latent, filter_refs, immiscible_counter, noise_prev
                if not isigma_end <= s.max() <= isigma_start or icfg.size == 0:
                    return override_noise_sampler(s, sn, *args, **kwargs)
                if icfg.filter_noise or icfg.filter_result:
                    curr_refs = filtering.FilterRefs(
                        kvs={
                            "sigma": s.clone(),
                            "sigma_next": sn.clone(),
                            "immiscible_counter": immiscible_counter,
                        }
                    )
                    if filter_refs is not None:
                        curr_refs = curr_refs | filter_refs
                if icfg.reference == "noise":
                    # FIXME: This doesn't honor immiscible ref scale
                    ref_latent = ns_ref(s, sn)
                elif icfg.reference == "initial_latent":
                    ref_latent = x_orig
                elif icfg.reference == "noise_gen_prev":
                    if noise_prev is None:
                        return fallback_noise_sampler(s, sn, *args, **kwargs)
                    ref_latent = noise_prev
                elif ref_latent is None:
                    raise ValueError("Immiscible reference type not available")
                if not isinstance(ibatching, str):
                    immiscible.batching = ibatching[immiscible_counter % len(ibatching)]
                immiscible_counter += 1
                blend_in_batch = icfg.blend != 1 and ns_blend is None
                # blending = icfg.blend != 1
                if icfg.norm_ref_scale != 0 and icfg.reference != "latent":
                    ref_latent = scale_noise(
                        ref_latent, icfg.norm_ref_scale, normalized=True
                    )
                batch_size = ref_latent.shape[0]
                noise_batch = torch.cat(
                    tuple(
                        override_noise_sampler(s, sn)
                        for _ in range(max(1, icfg.size) + int(blend_in_batch))
                    )
                )
                if icfg.filter_noise:
                    noise_batch = icfg.filter_noise.apply(noise_batch, refs=curr_refs)
                immiscible_noise = immiscible.unbatch(
                    immiscible.immiscible(
                        immiscible.batch(
                            scale_noise(
                                noise_batch[batch_size * int(blend_in_batch) :],
                                1.0
                                if icfg.norm_noise_scale == 0
                                else icfg.norm_noise_scale,
                                normalized=icfg.norm_noise_scale != 0,
                            )
                        ),
                        immiscible.batch(ref_latent),
                    ),
                    ref_latent.shape,
                )
                immiscible_noise = scale_noise(
                    immiscible_noise, normalized=cfg.normalize
                )
                if icfg.blend != 1:
                    immiscible_noise = blend_function(
                        noise_batch[:batch_size] if blend_in_batch else ns_blend(s, sn),
                        immiscible_noise,
                        icfg.blend,
                    )
                if icfg.filter_result:
                    immiscible_noise = icfg.filter_result.apply(
                        immiscible_noise, refs=curr_refs
                    )
                noise = scale_noise(immiscible_noise, normalized=cfg.normalize)
                noise_prev = noise.clone()
                return noise

        else:
            noise_sampler = fallback_noise_sampler

        kwargs["noise_sampler"] = noise_sampler
        return cfg.sampler.sampler_function(
            model,
            x,
            sigmas,
            *args,
            extra_args=extra_args,
            **kwargs,
        )


class ExpressionFilteredNoiseItem(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        normalize,
        noise: object,
        noise_filter: filtering.Filter,
        ref_filter: filtering.Filter | None,
        latent_refs: dict,
        call_sampler: bool = True,
        list_mode: bool = False,
    ):
        super().__init__(
            factor,
            noise=noise,
            noise_filter=noise_filter,
            ref_filter=ref_filter,
            normalize=normalize,
            latent_refs={k: v.clone() for k, v in latent_refs.items()},
            call_sampler=call_sampler,
            list_mode=list_mode,
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        if k == "latent_refs":
            return {k: v.clone() for k, v in self.latent_refs.items()}
        return super().clone_key(k)

    def make_noise_sampler(
        self,
        x: torch.Tensor,
        *args: Any,
        normalized=True,
        **kwargs: Any,
    ) -> Callable:
        list_mode = self.list_mode
        noise_filter = self.noise_filter
        initial_shape = x.shape
        latent_refs = self.latent_refs
        if self.ref_filter is not None:
            x = self.ref_filter.apply(
                x,
                refs=filtering.FilterRefs({k: v.to(x) for k, v in latent_refs.items()}),
            )
        if list_mode:
            list_items = getattr(self.noise, "items", None)
            if list_items is not None and not isinstance(list_items, (list, tuple)):
                raise TypeError("Bad type for noise.items - expected a list or tuple")
            ns_items = (self.noise,) if list_items is None else tuple(list_items)
        else:
            ns_items = (self.noise,)
        ns_items = tuple(
            n.make_noise_sampler(x, *args, normalized=False, **kwargs) for n in ns_items
        )
        if not ns_items:
            raise ValueError("Noise sampler list is empty!")
        # ns = self.noise.make_noise_sampler(x, *args, normalized=False, **kwargs)
        normalize_noise = self.normalize != False and normalized
        factor = self.factor
        initial_x = x.clone()
        call_sampler = self.call_sampler
        kvs = {
            "initial_x": initial_x,
            "initial_shape": initial_shape,
        } | {k: v.to(initial_x) for k, v in latent_refs.items()}
        sample_counter = 0
        last_noise = None

        def noise_sampler(s, sn, *args: Any, **kwargs: Any) -> torch.Tensor:
            nonlocal sample_counter, last_noise
            s_orig, sn_orig = (
                t.clone() if isinstance(t, torch.Tensor) else t for t in (s, sn)
            )
            curr_samplers = tuple(
                functools.partial(ns, s_orig, sn_orig, *args, **kwargs)
                for ns in ns_items
            )
            noise = curr_samplers[0]() if call_sampler else initial_x
            if (
                isinstance(s, torch.Tensor)
                and isinstance(sn, torch.Tensor)
                and s.numel() > 1
                and s.ndim < initial_x.ndim
            ):
                padded_shape = tuple(-1 if d == 0 else 1 for d in range(initial_x.ndim))
                s, sn = s.reshape(padded_shape), sn.reshape(padded_shape)

            refs = filtering.FilterRefs(
                kvs
                | {
                    "sigma": s,
                    "sigma_next": sn,
                    "sigma_orig": s_orig,
                    "sigma_next_orig": sn_orig,
                    "sample_counter": sample_counter,
                    "last_noise": last_noise,
                    "noise_samplers": tuple(
                        functools.partial(ns, s_orig, sn_orig, *args, **kwargs)
                        for ns in ns_items
                    ),
                }
            )
            sample_counter += 1
            noise = noise_filter.apply(noise, refs=refs)
            if isinstance(noise, dict):
                last_noise = noise["last"]
                if isinstance(last_noise, torch.Tensor):
                    last_noise = last_noise.clone()
                noise = scale_noise(noise["result"], factor, normalized=normalize_noise)
            else:
                noise = scale_noise(noise, factor, normalized=normalize_noise)
                last_noise = noise.clone()
            return noise

        return noise_sampler


class ExpressionFilteredNoiseNode(CustomNoiseNodeBase, NormalizeNoiseNodeMixin):
    DESCRIPTION = "Allows applying an OCS filter to custom noise. The following keys are supported:\nfilter: defines a filter for the generated noise. The filter will have these globals in scope: sigma, sigma_next, sample_counter, initial_x, initial_shape, last_noise (none on the first call), latent_ref_1 (to 3), noise_samplers (tuple of callables, must pass sigma and sigma_next). The filter should return either a tensor or a dict with result (noise to use) and last (tensor to use for last_noise) keys.\nUse the ref_filter key to define a filter for the initial latent reference. This will only be passed the latent (as default) and latent references.\ncall_sampler: boolean that defaults to true. When disabled, default will be initial_x and items from noise_samplers must be called manually.\nlist_mode: boolean that defaults to false."

    @classmethod
    def INPUT_TYPES(cls, *args: Any, **kwargs: Any) -> dict:
        MODULES.initialize()
        result = super().INPUT_TYPES(*args, **kwargs)
        result["required"] |= {
            "normalize": (
                ("default", "forced", "disabled"),
                {
                    "tooltip": "Controls whether the generated noise is normalized to 1.0 strength.",
                },
            ),
            "custom_noise": (
                WILDCARD_NOISE,
                {
                    "tooltip": "Input for custom noise used during ancestral or SDE sampling.",
                },
            ),
            "yaml_config": (
                "STRING",
                {
                    "default": "",
                    "placeholder": """\
# YAML or JSON filter definition
""",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": "Enter your filter definition here. There is essentially no error handling.",
                },
            ),
        }
        if "optional" not in result:
            result["optional"] = {}
        result["optional"] |= {
            "latent_ref_1_opt": ("LATENT",),
            "latent_ref_2_opt": ("LATENT",),
            "latent_ref_3_opt": ("LATENT",),
        }
        return result

    @classmethod
    def get_item_class(cls):
        return ExpressionFilteredNoiseItem

    def go(
        self,
        *,
        factor: float,
        rescale: float,
        normalize: str,
        custom_noise: object,
        yaml_config: str,
        latent_ref_1_opt: dict | None = None,
        latent_ref_2_opt: dict | None = None,
        latent_ref_3_opt: dict | None = None,
    ) -> tuple:
        config = yaml.safe_load(yaml_config)
        if isinstance(config, str):
            config = {"filter": {"final": config}}
        elif not isinstance(config, dict) or "filter" not in config:
            raise ValueError(
                "Bad YAML config type (must be object) or missing filter key in config",
            )
        noise_filter_def = config.get("filter")
        if not isinstance(noise_filter_def, dict):
            raise TypeError("Bad type for filter definition, must be object")
        ref_filter_def = config.get("ref_filter")
        if ref_filter_def is not None and not isinstance(ref_filter_def, dict):
            raise TypeError("ref_filter key must be an object if present")
        latent_refs = {
            k: v["samples"].to(device="cpu", dtype=torch.float32, copy=True)
            for k, v in (
                ("latent_ref_1", latent_ref_1_opt),
                ("latent_ref_2", latent_ref_2_opt),
                ("latent_ref_3", latent_ref_3_opt),
            )
            if v is not None
        }
        noise_filter = filtering.make_filter(noise_filter_def)
        ref_filter = (
            None if ref_filter_def is None else filtering.make_filter(ref_filter_def)
        )
        return super().go(
            factor,
            rescale=rescale,
            normalize=self.get_normalize(normalize),
            noise=custom_noise.clone(),
            noise_filter=noise_filter,
            ref_filter=ref_filter,
            latent_refs=latent_refs,
        )
