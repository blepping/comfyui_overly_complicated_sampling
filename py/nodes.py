from .sampling import composable_sampler
from .substep_sampling import StepSamplerChain, StepSamplerGroups, ParamGroup
from .substep_samplers import STEP_SAMPLERS
from .substep_merging import MERGE_SUBSTEPS_CLASSES

import comfy
import yaml

DEFAULT_YAML_PARAMS = """\
# Enter parameters here in JSON or YAML format
s_noise: 1.0
eta: 1.0
"""


class SamplerNode:
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/OCS"

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "groups": ("OCS_GROUPS",),
            },
            "optional": {
                "params_opt": ("OCS_PARAMS",),
                "parameters": (
                    "STRING",
                    {
                        "default": DEFAULT_YAML_PARAMS,
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
            },
        }

    def go(
        self,
        *,
        groups,
        params_opt=None,
        parameters="",
    ):
        options = {}
        parameters = parameters.strip()
        if parameters:
            extra_params = yaml.safe_load(parameters)
            if extra_params is not None:
                if not isinstance(extra_params, dict):
                    raise ValueError("Parameters must be a JSON or YAML object")
                options |= extra_params
        if params_opt is not None:
            options |= params_opt.items
        options["_groups"] = groups.clone()
        return (
            comfy.samplers.KSAMPLER(
                composable_sampler, {"overly_complicated_options": options}
            ),
        )


class GroupNode:
    RETURN_TYPES = ("OCS_GROUPS",)
    CATEGORY = "sampling/custom_sampling/OCS"

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merge_method": (tuple(MERGE_SUBSTEPS_CLASSES.keys()),),
                "time_mode": (("step", "step_pct", "sigma"),),
                "time_start": (
                    "FLOAT",
                    {"default": 0, "min": 0.0, "step": 0.1, "round'": False},
                ),
                "time_end": (
                    "FLOAT",
                    {"default": 999, "min": 0.0, "step": 0.1, "round'": False},
                ),
                "substeps": ("OCS_SUBSTEPS",),
            },
            "optional": {
                "groups_opt": ("OCS_GROUPS",),
                "params_opt": ("OCS_PARAMS",),
                "parameters": (
                    "STRING",
                    {
                        "default": DEFAULT_YAML_PARAMS,
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
            },
        }

    def go(
        self,
        *,
        merge_method,
        time_mode,
        time_start,
        time_end,
        substeps,
        groups_opt=None,
        params_opt=None,
        parameters="",
    ):
        group = StepSamplerGroups() if groups_opt is None else groups_opt.clone()
        chain = substeps.clone()
        chain.merge_method = merge_method
        chain.time_mode = time_mode
        chain.time_start, chain.time_end = time_start, time_end
        options = {}
        parameters = parameters.strip()
        if parameters:
            extra_params = yaml.safe_load(parameters)
            if extra_params is not None:
                if not isinstance(extra_params, dict):
                    raise ValueError("Parameters must be a JSON or YAML object")
                options |= extra_params
        if params_opt is not None:
            options |= params_opt.items
        chain.options |= options
        group.append(chain)
        return (group,)


class SubstepsNode:
    RETURN_TYPES = ("OCS_SUBSTEPS",)
    CATEGORY = "sampling/custom_sampling/OCS"

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "substeps": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "step_method": (tuple(STEP_SAMPLERS.keys()),),
            },
            "optional": {
                "substeps_opt": ("OCS_SUBSTEPS",),
                "params_opt": ("OCS_PARAMS",),
                "parameters": (
                    "STRING",
                    {
                        "default": DEFAULT_YAML_PARAMS,
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
            },
        }

    def go(
        self,
        *,
        parameters="",
        substeps_opt=None,
        params_opt=None,
        **kwargs,
    ):
        if substeps_opt is not None:
            chain = substeps_opt.clone()
        else:
            chain = StepSamplerChain()
        parameters = parameters.strip()
        if parameters:
            extra_params = yaml.safe_load(parameters)
            if extra_params is not None:
                if not isinstance(extra_params, dict):
                    raise ValueError("Parameters must be a JSON or YAML object")
                kwargs |= extra_params
        if params_opt is not None:
            kwargs |= params_opt.items
        chain.append(kwargs)
        return (chain,)


class Wildcard(str):
    __slots__ = ()

    def __ne__(self, _unused):
        return False


class ParamNode:
    RETURN_TYPES = ("OCS_PARAMS",)
    CATEGORY = "sampling/custom_sampling/OCS"
    FUNCTION = "go"

    WC = Wildcard("*")

    OCS_PARAM_TYPES = {
        "custom_noise": lambda v: hasattr(v, "make_noise_sampler"),
        "merge_sampler": lambda v: isinstance(v, StepSamplerChain),
        "restart_custom_noise": lambda v: hasattr(v, "make_noise_sampler"),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": (tuple(cls.OCS_PARAM_TYPES.keys()),),
                "value": (cls.WC,),
            },
            "optional": {"params_opt": ("OCS_PARAMS",)},
        }

    def go(self, *, key, value, params_opt=None):
        if not self.OCS_PARAM_TYPES[key](value):
            raise ValueError(f"CSamplerParam: Bad value type for key {key}")
        params = ParamGroup(items={}) if params_opt is None else params_opt.clone()
        params[key] = value
        return (params,)


class MultiParamNode:
    RETURN_TYPES = ("OCS_PARAMS",)
    CATEGORY = "sampling/custom_sampling/OCS"
    FUNCTION = "go"

    PARAM_COUNT = 5

    @classmethod
    def INPUT_TYPES(cls):
        param_keys = (("", *ParamNode.OCS_PARAM_TYPES.keys()),)
        return {
            "required": {
                f"key_{idx}": param_keys for idx in range(1, cls.PARAM_COUNT + 1)
            },
            "optional": {"params_opt": ("OCS_PARAMS",)}
            | {
                f"value_opt_{idx}": (ParamNode.WC,)
                for idx in range(1, cls.PARAM_COUNT + 1)
            },
        }

    def go(self, *, params_opt=None, **kwargs):
        params = ParamGroup(items={}) if params_opt is None else params_opt.clone()
        for idx in range(1, self.PARAM_COUNT + 1):
            key, value = kwargs.get(f"key_{idx}"), kwargs.get(f"value_opt_{idx}")
            if not key or value is None:
                continue
            if not ParamNode.OCS_PARAM_TYPES[key](value):
                raise ValueError(f"CSamplerParamGroup: Bad value type for key {key}")
            params[key] = value
        return (params,)


class SimpleRestartSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"sigmas": ("SIGMAS",)},
            "optional": {"segments": ("STRING", {"default": "10+4x2"})},
        }


__all__ = (
    "SamplerNode",
    "GroupNode",
    "SubstepsNode",
    "ParamNode",
    "MultiParamNode",
)
