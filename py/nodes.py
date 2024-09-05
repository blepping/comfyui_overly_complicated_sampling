import yaml

import comfy

from .sampling import composable_sampler
from .substep_sampling import StepSamplerChain, StepSamplerGroups, ParamGroup
from .step_samplers import STEP_SAMPLERS
from .substep_merging import MERGE_SUBSTEPS_CLASSES
from .restart import Restart

DEFAULT_YAML_PARAMS = """\
# JSON or YAML parameters
s_noise: 1.0
eta: 1.0
"""


class SamplerNode:
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/OCS"
    DESCRIPTION = "Overly Complicated Sampling main sampler node. Can be connected to a SamplerCustom or other sampler node that supports a SAMPLER input."
    OUTPUT_TOOLTIPS = (
        "SAMPLER that can be connected to a SamplerCustom or other sampler node that supports a SAMPLER input.",
    )

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "groups": (
                    "OCS_GROUPS",
                    {
                        "tooltip": "Connect OCS substep groups here which are output from the OCS Group node."
                    },
                ),
            },
            "optional": {
                "params_opt": (
                    "OCS_PARAMS",
                    {
                        "tooltip": "Optionally connect parameters like custom noise here. Output from the OCS Param or OCS MultiParam nodes.",
                    },
                ),
                "parameters": (
                    "STRING",
                    {
                        "default": DEFAULT_YAML_PARAMS,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "The text parameter block allows setting custom parameters using YAML (recommended) or JSON. Optional, may be left blank.",
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
    DESCRIPTION = "Over Complicated Sampling group definition node."
    OUTPUT_TOOLTIPS = (
        "This output can be connect to another OCS Group node or an OCS Sampler node.",
    )

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merge_method": (
                    tuple(MERGE_SUBSTEPS_CLASSES.keys()),
                    {
                        "tooltip": "The merge method determines how multiple substeps are combined together during sampling.",
                    },
                ),
                "time_mode": (
                    ("step", "step_pct", "sigma"),
                    {
                        "tooltip": "The time mode controls how the time_start and time_end parameters are interpreted. The default of step is generally easiest to use.",
                    },
                ),
                "time_start": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0.0,
                        "step": 0.1,
                        "round'": False,
                        "tooltip": "The start time this group will be active (inclusive).",
                    },
                ),
                "time_end": (
                    "FLOAT",
                    {
                        "default": 999,
                        "min": 0.0,
                        "step": 0.1,
                        "round'": False,
                        "tooltip": "The group will become inactive when the current time is GREATER than the specified end time.",
                    },
                ),
                "substeps": (
                    "OCS_SUBSTEPS",
                    {
                        "tooltip": "Connect output from an OCS Substeps node here.",
                    },
                ),
            },
            "optional": {
                "groups_opt": (
                    "OCS_GROUPS",
                    {
                        "tooltip": "You may optionally connect the output from another OCS Group node here. Only one group per step is used, matching (based on time or other constraints) starts with the OCS Group node furthest from the OCS Sampler.",
                    },
                ),
                "params_opt": (
                    "OCS_PARAMS",
                    {
                        "tooltip": "Optionally connect parameters like custom noise here. Output from the OCS Param or OCS MultiParam nodes.",
                    },
                ),
                "parameters": (
                    "STRING",
                    {
                        "default": DEFAULT_YAML_PARAMS,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "The text parameter block allows setting custom parameters using YAML (recommended) or JSON. Optional, may be left blank.",
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
    DESCRIPTION = "Overly Complicated Sampling substeps definition node. Used to define a sampler type and other sampler-specific parameters."
    OUTPUT_TOOLTIPS = (
        "This output can be connected to another OCS Substeps node or an OCS Group node.",
    )

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "substeps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Number of substeps to use for each step, in other words (depending on the OCS Group merge strategy) it may split a step into multiple smaller steps.",
                    },
                ),
                "step_method": (
                    tuple(STEP_SAMPLERS.keys()),
                    {
                        "tooltip": "In other words, the sampler.",
                    },
                ),
            },
            "optional": {
                "substeps_opt": (
                    "OCS_SUBSTEPS",
                    {
                        "tooltip": "Optionally connect another OCS Substeps node here. Substeps will run in order, starting from the OCS Substeps node FURTHEST from the OCS Group node.",
                    },
                ),
                "params_opt": (
                    "OCS_PARAMS",
                    {
                        "tooltip": "Optionally connect parameters like custom noise here. Output from the OCS Param or OCS MultiParam nodes.",
                    },
                ),
                "parameters": (
                    "STRING",
                    {
                        "default": DEFAULT_YAML_PARAMS,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "The text parameter block allows setting custom parameters using YAML (recommended) or JSON. Optional, may be left blank.",
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
    DESCRIPTION = "Overly Complicated Sampling parameter definition node. Used to set parameters like custom noise types that require an input."
    OUTPUT_TYPES = (
        "Can be connected to another OCS Param or OCS MultiParam node or any other OCS node that takes OCS_PARAMS as an input.",
    )

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
                "key": (
                    tuple(cls.OCS_PARAM_TYPES.keys()),
                    {
                        "tooltip": "Used to set the type of custom parameter.",
                    },
                ),
                "value": (
                    cls.WC,
                    {
                        "tooltip": "Connect the type of value expected by the key. Allows connecting output from any type of node HOWEVER if it is the wrong type expected by the key you will get an error when you run the workflow.",
                    },
                ),
            },
            "optional": {
                "params_opt": (
                    "OCS_PARAMS",
                    {
                        "tooltip": "You may optionally connect the output from other OCS Param or OCS MultiParam nodes here to set multiple parameters.",
                    },
                ),
                "parameters": (
                    "STRING",
                    {
                        "default": "# Additional YAML or JSON parameters\n",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "The text parameter block allows setting custom parameters using YAML (recommended) or JSON. Optional, may be left blank.",
                    },
                ),
            },
        }

    @classmethod
    def get_renamed_key(cls, key, params):
        rename = params.get("rename")
        if rename is None:
            return key
        if not isinstance(rename, str):
            raise ValueError("Param rename key must be a string if set")
        rename = rename.strip()
        if not rename or not all(c == "_" or c.isalnum() for c in rename):
            raise ValueError(
                "Param rename keys must consist of one or more alphanumeric or underscore characters"
            )
        return f"{key}_{rename}"

    def go(self, *, key, value, params_opt=None, parameters=""):
        if not self.OCS_PARAM_TYPES[key](value):
            raise ValueError(f"CSamplerParam: Bad value type for key {key}")
        if parameters:
            extra_params = yaml.safe_load(parameters)
            if extra_params is not None:
                if not isinstance(extra_params, dict):
                    raise ValueError("Parameters must be a JSON or YAML object")
                key = self.get_renamed_key(key, extra_params)
        else:
            extra_params = None
        params = ParamGroup(items={}) if params_opt is None else params_opt.clone()
        params[key] = value
        if extra_params is not None:
            params[f"{key}.params"] = extra_params
        return (params,)


class MultiParamNode(ParamNode):
    RETURN_TYPES = ("OCS_PARAMS",)
    CATEGORY = "sampling/custom_sampling/OCS"
    DESCRIPTION = "Overly Complicated Sampling parameter definition node. Used to set parameters like custom noise types that require an input. Like the OCS Param node but allows setting multiple parameters at the same time."
    OUTPUT_TYPES = (
        "Can be connected to another OCS Param or OCS MultiParam node or any other OCS node that takes OCS_PARAMS as an input.",
    )

    FUNCTION = "go"

    PARAM_COUNT = 5

    @classmethod
    def INPUT_TYPES(cls):
        param_keys = (
            ("", *ParamNode.OCS_PARAM_TYPES.keys()),
            {
                "tooltip": "Used to set the type of custom parameter.",
            },
        )
        return {
            "required": {
                f"key_{idx}": param_keys for idx in range(1, cls.PARAM_COUNT + 1)
            },
            "optional": {
                "params_opt": (
                    "OCS_PARAMS",
                    {
                        "tooltip": "You may optionally connect the output from other OCS MultiParam or OCS Param nodes here to set multiple parameters.",
                    },
                ),
                "parameters": (
                    "STRING",
                    {
                        "default": """\
# Additional YAML or JSON parameters
# Should be an object with key corresponding to the index of the input
""",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "The text parameter block allows setting custom parameters using YAML (recommended) or JSON. Optional, may be left blank.",
                    },
                ),
            }
            | {
                f"value_opt_{idx}": (
                    ParamNode.WC,
                    {
                        "tooltip": "Connect the type of value expected by the corresponding key. Allows connecting output from any type of node HOWEVER if it is the wrong type expected by the corresponding key you will get an error when you run the workflow.",
                    },
                )
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
            if not self.OCS_PARAM_TYPES[key](value):
                raise ValueError(f"CSamplerParamGroup: Bad value type for key {key}")
            extra = extra_params.get(str(idx))
            key = self.get_renamed_key(key, extra)
            params[key] = value
            if extra is not None:
                params[f"{key}.params"] = extra

        return (params,)


class SimpleRestartSchedule:
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/OCS"
    DESCRIPTION = "Overly Complicated Sampling simple Restart schedule node. Allows generating a Restart sampling schedule based on a text definition."
    OUTPUT_TYPES = (
        "Can be connected to an OCS Sampler or RestartSampler node. Do not connect directly to a sampler that doesn't have built-in support for Restart schedules.",
    )

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": "Connect the output from another scheduler node (i.e. BasicScheduler) here.",
                    },
                ),
                "start_step": (
                    "INT",
                    {
                        "min": 0,
                        "default": 0,
                        "tooltip": "Step the restart schedule definition starts applying. Zero-based.",
                    },
                ),
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
                        "tooltip": "Define a schedule here using YAML (recommended) or JSON.",
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
    DESCRIPTION = "Allows forcing a model's maximum and minumum sigmas to a specified value. You generally do NOT want to connect this to a sampler node. Connect it to a scheduler node (i.e. BasicScheduler) instead."
    OUTPUT_TOOLTIPS = (
        "Patched model. Can be connected to a scheduler node (i.e. BasicScheduler). Generally NOT recommended to connect to an actual sampler.",
    )

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Model to patch with the min/max sigmas.",
                    },
                ),
                "mode": (
                    ("recalculate", "simple_multiply"),
                    {
                        "tooltip": "Mode use for setting sigmas in the patched model. Recalculate should generally be more accurate.",
                    },
                ),
                "sigma_max": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -10000.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "You can set the maximum sigma here. If you use a negative value, it will be interpreted as the absolute value for the max sigma. If you use a positive value it will be interpreted as a percentage (where 1.0 signified 100%). Schedules generated with the patched model should start from sigma_max (or close to it).",
                    },
                ),
                "fake_sigma_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "You can set the minimum sigma here. Disabled if set to 0. If you use a negative value, it will be interpreted as the absolute value for the max sigma. If you use a positive value it will be interpreted as a percentage (where 1.0 signified 100%). Schedules generated with the patched model should end with [sigma_min, 0]. NOTE: May not work with some schedulers. I recommend leaving this at 0 unless you know you need it (and even then it may not work).",
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
