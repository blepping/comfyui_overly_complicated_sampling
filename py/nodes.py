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


class ComposableSampler:
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "step_sampler_groups": ("STEP_SAMPLER_GROUPS",),
            },
            "optional": {
                "csampler_params_opt": ("CSAMPLER_PARAMS",),
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
        step_sampler_groups,
        csampler_params_opt=None,
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
        if csampler_params_opt is not None:
            options |= csampler_params_opt.items
        options["_groups"] = step_sampler_groups.clone()
        return (
            comfy.samplers.KSAMPLER(
                composable_sampler,
                {"composable_sampler_options": options},
            ),
        )


class SubstepsGroup:
    RETURN_TYPES = ("STEP_SAMPLER_GROUPS",)
    CATEGORY = "sampling/custom_sampling/samplers"

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
                "step_sampler_chain": ("STEP_SAMPLER_CHAIN",),
            },
            "optional": {
                "step_sampler_groups_opt": ("STEP_SAMPLER_GROUPS",),
                "csampler_params_opt": ("CSAMPLER_PARAMS",),
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
        step_sampler_chain,
        step_sampler_group_opt=None,
        csampler_params_opt=None,
        parameters="",
    ):
        group = (
            StepSamplerGroups()
            if step_sampler_group_opt is None
            else step_sampler_group_opt
        )
        chain = step_sampler_chain.clone()
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
        if csampler_params_opt is not None:
            options |= csampler_params_opt.items
        chain.options |= options
        group.append(chain)
        return (group,)


class ComposableStepSampler:
    RETURN_TYPES = ("STEP_SAMPLER_CHAIN",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "substeps": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "step_method": (tuple(STEP_SAMPLERS.keys()),),
            },
            "optional": {
                "step_sampler_opt": ("STEP_SAMPLER_CHAIN",),
                "csampler_params_opt": ("CSAMPLER_PARAMS",),
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
        step_sampler_opt=None,
        csampler_params_opt=None,
        **kwargs,
    ):
        if step_sampler_opt is not None:
            chain = step_sampler_opt.clone()
        else:
            chain = StepSamplerChain()
        parameters = parameters.strip()
        if parameters:
            extra_params = yaml.safe_load(parameters)
            if extra_params is not None:
                if not isinstance(extra_params, dict):
                    raise ValueError("Parameters must be a JSON or YAML object")
                kwargs |= extra_params
        if csampler_params_opt is not None:
            kwargs |= csampler_params_opt.items
        chain.append(kwargs)
        return (chain,)


class Wildcard(str):
    __slots__ = ()

    def __ne__(self, _unused):
        return False


class CSamplerParam:
    RETURN_TYPES = ("CSAMPLER_PARAMS",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "go"

    WC = Wildcard("*")

    CPARAM_TYPES = {
        "custom_noise": lambda v: hasattr(v, "make_noise_sampler"),
        "merge_sampler": lambda v: isinstance(v, StepSamplerChain),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"key": (tuple(cls.CPARAM_TYPES.keys()),), "value": (cls.WC,)},
            "optional": {"csampler_params_opt": ("CSAMPLER_PARAMS",)},
        }

    def go(self, *, key, value, csampler_params_opt=None):
        if not self.CPARAM_TYPES[key](value):
            raise ValueError(f"CSamplerParam: Bad value type for key {key}")
        params = (
            ParamGroup(items={})
            if csampler_params_opt is None
            else csampler_params_opt.clone()
        )
        params[key] = value
        return (params,)


class CSamplerParamMulti:
    RETURN_TYPES = ("CSAMPLER_PARAMS",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "go"

    PARAM_COUNT = 5

    @classmethod
    def INPUT_TYPES(cls):
        param_keys = (("", *CSamplerParam.CPARAM_TYPES.keys()),)
        return {
            "required": {
                f"key_{idx}": param_keys for idx in range(1, cls.PARAM_COUNT + 1)
            },
            "optional": {"csampler_params_opt": ("CSAMPLER_PARAMS",)}
            | {
                f"value_opt_{idx}": (CSamplerParam.WC,)
                for idx in range(1, cls.PARAM_COUNT + 1)
            },
        }

    def go(self, *, csampler_params_opt=None, **kwargs):
        params = (
            ParamGroup(items={})
            if csampler_params_opt is None
            else csampler_params_opt.clone()
        )
        for idx in range(1, self.PARAM_COUNT + 1):
            key, value = kwargs.get(f"key_{idx}"), kwargs.get(f"value_opt_{idx}")
            if not key or value is None:
                continue
            if not CSamplerParam.CPARAM_TYPES[key](value):
                raise ValueError(f"CSamplerParamGroup: Bad value type for key {key}")
            params[key] = value
        return (params,)


__all__ = (
    "ComposableStepSampler",
    "ComposableSampler",
    "CSamplerParam",
    "CSamplerParamMulti",
)
