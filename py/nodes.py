import yaml

import comfy

from .sampling import composable_sampler
from .substep_sampling import StepSamplerChain, StepSamplerGroups, ParamGroup
from .step_samplers import STEP_SAMPLERS
from .substep_merging import MERGE_SUBSTEPS_CLASSES
from .utils import Restart

DEFAULT_YAML_PARAMS = """\
# JSON or YAML parameters
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
        "SAMPLER": lambda _v: True,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": (tuple(cls.OCS_PARAM_TYPES.keys()),),
                "value": (cls.WC,),
            },
            "optional": {
                "params_opt": ("OCS_PARAMS",),
                "parameters": (
                    "STRING",
                    {
                        "default": "# Additional YAML or JSON parameters\n",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
            },
        }

    def go(self, *, key, value, params_opt=None, parameters=""):
        if not self.OCS_PARAM_TYPES[key](value):
            raise ValueError(f"CSamplerParam: Bad value type for key {key}")
        if parameters:
            extra_params = yaml.safe_load(parameters)
            if extra_params is not None:
                if not isinstance(extra_params, dict):
                    raise ValueError("Parameters must be a JSON or YAML object")
        else:
            extra_params = None
        params = ParamGroup(items={}) if params_opt is None else params_opt.clone()
        params[key] = value
        if extra_params is not None:
            params[f"{key}.params"] = extra_params
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
            "optional": {
                "params_opt": ("OCS_PARAMS",),
                "parameters": (
                    "STRING",
                    {
                        "default": """\
# Additional YAML or JSON parameters
# Should be an object with key corresponding to the index of the input
""",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
            }
            | {
                f"value_opt_{idx}": (ParamNode.WC,)
                for idx in range(1, cls.PARAM_COUNT + 1)
            },
        }

    def go(self, *, params_opt=None, parameters="", **kwargs):
        params = ParamGroup(items={}) if params_opt is None else params_opt.clone()
        if parameters:
            extra_params = yaml.safe_load(parameters)
            if extra_params is not None:
                if not isinstance(extra_params, dict):
                    raise ValueError("Parameters must be a JSON or YAML object")
            else:
                extra_params = {}
        else:
            extra_params = {}
        for idx in range(1, self.PARAM_COUNT + 1):
            key, value = kwargs.get(f"key_{idx}"), kwargs.get(f"value_opt_{idx}")
            if not key or value is None:
                continue
            if not ParamNode.OCS_PARAM_TYPES[key](value):
                raise ValueError(f"CSamplerParamGroup: Bad value type for key {key}")
            params[key] = value
            extra = extra_params.get(str(idx))
            if extra is not None:
                params[f"{key}.params"] = extra

        return (params,)


class SimpleRestartSchedule:
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/OCS"
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "start_step": ("INT", {"min": 0, "default": 0}),
            },
            "optional": {
                "schedule": (
                    "STRING",
                    {
                        "default": """\
# YAML or JSON restart schedule
# Every 5 steps, jump back 3 steps
- [5, -3]
# Jump to schedule item 0
- 0
""",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
            },
        }

    def go(self, *, sigmas, start_step=0, schedule="[]"):
        if schedule:
            parsed_schedule = yaml.safe_load(schedule)
            if parsed_schedule is not None:
                if not isinstance(parsed_schedule, (list, tuple)):
                    raise ValueError("Schedule must be a JSON or YAML list")
            else:
                parsed_schedule = []
        else:
            parsed_schedule = []
        return (Restart.simple_schedule(sigmas, start_step, parsed_schedule),)


class ModelSetMaxSigmaNode:
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "hacks"
    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (("recalculate", "simple_multiply"),),
                "sigma_max": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -10000.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round'": False,
                    },
                ),
                "fake_sigma_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "round'": False,
                    },
                ),
            }
        }

    def go(self, model, mode="recalculate", sigma_max=-1.0, fake_sigma_min=0.0):
        if sigma_max == 0:
            raise ValueError("ModelSetMaxSigma: Invalid sigma_max value")
        if mode not in ("recalculate", "simple_multiply"):
            raise ValueError("ModelSetMaxSigma: Invalid mode value")
        orig_ms = model.get_model_object("model_sampling")
        model = model.clone()
        orig_max_sigma, orig_min_sigma = (
            orig_ms.sigma_max.item(),
            orig_ms.sigma_min.item(),
        )
        max_multiplier = abs(sigma_max) if sigma_max < 0 else sigma_max / orig_max_sigma
        if max_multiplier == 1:
            return (model,)
        mcfg = model.get_model_object("model_config")
        orig_sigmas = orig_ms.sigmas
        fake_sigma_min = orig_sigmas.new_full((1,), fake_sigma_min)

        class NewModelSampling(orig_ms.__class__):
            if fake_sigma_min != 0:

                @property
                def sigma_min(self):
                    return fake_sigma_min

        ms = NewModelSampling(mcfg)
        if mode == "simple_multiply":
            ms.set_sigmas(orig_sigmas * max_multiplier)
        else:
            ss = getattr(mcfg, "sampling_setting", None) or {}
            if ss.get("beta_schedule", "linear") != "linear":
                raise NotImplementedError(
                    "ModelSetMaxSigma: Can only handle linear beta schedules in reschedule mode"
                )
            ms.set_sigmas((orig_sigmas**2 * max_multiplier**2) ** 0.5)
        new_max_sigma, new_min_sigma = ms.sigma_max.item(), ms.sigma_min.item()
        if new_min_sigma >= new_max_sigma:
            raise ValueError(
                "ModelSetMaxSigma: Invalid fake_min_sigma value, result max <= min"
            )
        model.add_object_patch("model_sampling", ms)
        print(
            f"ModelSetMaxSigma: Set model sigmas({mode}): old_max={orig_max_sigma:.04}, old_min={orig_min_sigma:.03}, new_max={new_max_sigma:.04}, new_min={new_min_sigma:.03}"
        )
        return (model,)


__all__ = (
    "SamplerNode",
    "GroupNode",
    "SubstepsNode",
    "ParamNode",
    "MultiParamNode",
    "ModelSetMaxSigmaNode",
)
