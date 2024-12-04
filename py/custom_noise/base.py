import abc
import torch

from typing import Callable, Any

from ..noise import scale_noise
from ..nodes import WILDCARD_NOISE


class CustomNoiseItemBase(abc.ABC):
    def __init__(self, factor, **kwargs):
        self.factor = factor
        self.keys = set(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def clone_key(self, k):
        return getattr(self, k)

    def clone(self):
        return self.__class__(self.factor, **{k: self.clone_key(k) for k in self.keys})

    def set_factor(self, factor):
        self.factor = factor
        return self

    def get_normalize(self, k, default=None):
        val = getattr(self, k, None)
        return default if val is None else val

    @abc.abstractmethod
    def make_noise_sampler(
        self,
        x: torch.Tensor,
        sigma_min=None,
        sigma_max=None,
        seed=None,
        cpu=True,
        normalized=True,
    ):
        raise NotImplementedError


class CustomNoiseChain:
    def __init__(self, items=None):
        self.items = items if items is not None else []

    def clone(self):
        return CustomNoiseChain(
            [i.clone() for i in self.items],
        )

    def add(self, item):
        if item is None:
            raise ValueError("Attempt to add nil item")
        self.items.append(item)

    @property
    def factor(self):
        return sum(abs(i.factor) for i in self.items)

    def rescaled(self, scale=1.0):
        divisor = self.factor / scale
        divisor = divisor if divisor != 0 else 1.0
        result = self.clone()
        if divisor != 1:
            for i in result.items:
                i.set_factor(i.factor / divisor)
        return result

    @torch.no_grad()
    def make_noise_sampler(
        self,
        x: torch.Tensor,
        sigma_min=None,
        sigma_max=None,
        seed=None,
        cpu=True,
        normalized=True,
    ) -> Callable:
        noise_samplers = tuple(
            i.make_noise_sampler(
                x,
                sigma_min,
                sigma_max,
                seed=seed,
                cpu=cpu,
                normalized=False,
            )
            for i in self.items
        )
        if not noise_samplers or not all(noise_samplers):
            raise ValueError("Failed to get noise sampler")
        factor = self.factor

        def noise_sampler(sigma, sigma_next):
            result = None
            for ns in noise_samplers:
                noise = ns(sigma, sigma_next)
                if result is None:
                    result = noise
                else:
                    result += noise
            return scale_noise(result, factor, normalized=normalized)

        return noise_sampler


class CustomNoiseNodeBase(abc.ABC):
    DESCRIPTION = "An Overly Complicated Sampling custom noise item."
    RETURN_TYPES = ("OCS_NOISE",)
    OUTPUT_TOOLTIPS = ("A custom noise chain.",)
    CATEGORY = "OveryComplicatedSampling/noise"
    FUNCTION = "go"

    @abc.abstractmethod
    def get_item_class(self):
        raise NotImplementedError

    @classmethod
    def INPUT_TYPES(cls, *, include_rescale=True, include_chain=True):
        result = {
            "required": {
                "factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.001,
                        "round": False,
                        "tooltip": "Scaling factor for the generated noise of this type.",
                    },
                ),
            },
            "optional": {},
        }
        if include_rescale:
            result["required"] |= {
                "rescale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.001,
                        "round": False,
                        "tooltip": "When non-zero, this custom noise item and other custom noise items items connected to it will have their factor scaled to add up to the specified rescale value.",
                    },
                ),
            }
        if include_chain:
            result["optional"] |= {
                "ocs_noise_opt": (
                    WILDCARD_NOISE,
                    {
                        "tooltip": "Optional input for more custom noise items.",
                    },
                ),
            }
        return result

    def go(
        self,
        factor=1.0,
        rescale=0.0,
        ocs_noise_opt=None,
        **kwargs: dict[str, Any],
    ):
        nis = ocs_noise_opt.clone() if ocs_noise_opt else CustomNoiseChain()
        if factor != 0:
            nis.add(self.get_item_class()(factor, **kwargs))
        return (nis if rescale == 0 else nis.rescaled(rescale),)


class NormalizeNoiseNodeMixin:
    @staticmethod
    def get_normalize(val: str) -> None | bool:
        return None if val == "default" else val == "forced"
