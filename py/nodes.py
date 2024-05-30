from .sampling import composable_sampler, STEP_SAMPLERS
from .substep_sampling import StepSamplerChain
from .substep_merging import MERGE_SUBSTEPS_CLASSES

import comfy
import yaml


class ComposableSampler:
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "merge_method": (tuple(MERGE_SUBSTEPS_CLASSES.keys()),),
                "step_sampler_chain": ("STEP_SAMPLER_CHAIN",),
            },
            "optional": {
                "merge_sampler_opt": ("STEP_SAMPLER_CHAIN",),
                "parameters": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
            },
        }

    def go(
        self,
        *,
        s_noise,
        eta,
        merge_method,
        step_sampler_chain,
        merge_sampler_opt=None,
        parameters="",
    ):
        if merge_sampler_opt is not None:
            merge_sampler = merge_sampler_opt.items[0]
        else:
            merge_sampler = ComposableStepSampler().go(step_method="euler")[0].items[0]
        options = {
            "s_noise": s_noise,
            "eta": eta,
            "merge_method": merge_method,
            "merge_sampler": merge_sampler,
        }
        parameters = parameters.strip()
        if parameters:
            extra_params = yaml.safe_load(parameters)
            if not isinstance(extra_params, dict):
                raise ValueError("Parameters must be a JSON or YAML object")
            options |= extra_params
        options["chain"] = step_sampler_chain.clone()
        return (
            comfy.samplers.KSAMPLER(
                composable_sampler,
                {"composable_sampler_options": options},
            ),
        )


class ComposableStepSampler:
    RETURN_TYPES = ("STEP_SAMPLER_CHAIN",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "go"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "substeps": ("INT", {"default": 1, "min": 1, "max": 100}),
                "step_method": (tuple(STEP_SAMPLERS.keys()),),
            },
            "optional": {
                "step_sampler_opt": ("STEP_SAMPLER_CHAIN",),
                "custom_noise_opt": ("SONAR_CUSTOM_NOISE",),
                "parameters": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
            },
        }

    def go(self, *, parameters="", step_sampler_opt=None, **kwargs):
        if step_sampler_opt is not None:
            chain = step_sampler_opt.clone()
        else:
            chain = StepSamplerChain()
        parameters = parameters.strip()
        if parameters:
            extra_params = yaml.safe_load(parameters)
            if not isinstance(extra_params, dict):
                raise ValueError("Parameters must be a JSON or YAML object")
            kwargs |= extra_params
        chain.items.append(kwargs)
        return (chain,)


__all__ = ("ComposableStepSampler", "ComposableSampler")
