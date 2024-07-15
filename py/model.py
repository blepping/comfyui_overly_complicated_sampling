import torch

import comfy
from comfy.k_diffusion.sampling import to_d

from .utils import fallback


class History:
    def __init__(self, size):
        self.history = []
        self.size = size

    def __len__(self):
        return len(self.history)

    def __getitem__(self, k):
        return self.history[k]

    def push(self, val):
        if len(self.history) >= self.size:
            self.history = self.history[-(self.size - 1) :]
        self.history.append(val)

    def reset(self):
        self.history = []

    def clone(self):
        obj = self.__new__(self.__class__)
        obj.__init__(self.size)
        obj.history = self.history.copy()
        return obj


class ModelResult:
    def __init__(
        self,
        call_idx,
        sigma,
        x,
        denoised,
        **kwargs,
    ):
        self.call_idx = call_idx
        self.sigma = sigma
        self.x = x
        self.denoised = denoised
        for k in ("denoised_uncond", "denoised_cond", "tangents", "jdenoised"):
            setattr(self, k, kwargs.pop(k, None))
        if len(kwargs) != 0:
            raise ValueError(f"Unexpected keyword arguments: {tuple(kwargs.keys())}")

    def to_d(
        self,
        /,
        x=None,
        sigma=None,
        denoised=None,
        denoised_uncond=None,
        alt_cfgpp_scale=0,
        cfgpp=False,
    ):
        x = fallback(x, self.x)
        sigma = fallback(sigma, self.sigma)
        denoised = fallback(denoised, self.denoised)
        denoised_uncond = fallback(denoised_uncond, self.denoised_uncond)
        if alt_cfgpp_scale != 0:
            x = x - denoised * alt_cfgpp_scale + denoised_uncond * alt_cfgpp_scale
        return to_d(x, sigma, denoised if not cfgpp else denoised_uncond)

    @property
    def d(self):
        return self.to_d()

    def clone(self, deep=False):
        obj = self.__new__(self.__class__)
        for k in (
            "denoised",
            "call_idx",
            "sigma",
            "x",
            "denoised_uncond",
            "denoised_cond",
            "tangents",
            "jdenoised",
        ):
            val = getattr(self, k)
            if deep and isinstance(val, torch.Tensor):
                val = val.copy()
            setattr(obj, k, val)
        return obj


class ModelCallCache:
    def __init__(
        self, model, x, s_in, extra_args, *, size=0, max_use=1000000, threshold=1
    ):
        self.size = size
        self.model = model
        self.threshold = threshold
        self.s_in = s_in
        self.max_use = max_use
        self.extra_args = extra_args
        if self.size < 1:
            return
        self.reset_cache()

    def reset_cache(self):
        size = self.size
        self.slot = [None] * size
        self.slot_use = [self.max_use] * size

    def get(self, idx, *, jvp=False):
        idx -= self.threshold
        if (
            idx >= self.size
            or idx < 0
            or self.slot[idx] is None
            or self.slot_use[idx] < 1
        ):
            return None
        result = self.slot[idx]
        if jvp and result.jdenoised is None:
            return None
        self.slot_use[idx] -= 1
        return result

    def set(self, idx, mr):
        idx -= self.threshold
        if idx < 0 or idx >= self.size:
            return
        self.slot_use[idx] = self.max_use
        self.slot[idx] = mr

    def call_model(self, x, sigma, **kwargs):
        return self.model(x, sigma * self.s_in, **self.extra_args | kwargs)

    @property
    def model_sampling(self):
        return self.model.inner_model.inner_model.model_sampling

    def __call__(
        self,
        x,
        sigma,
        *,
        model_call_idx=0,
        s_in=None,
        tangents=None,
        return_cached=False,
        **kwargs,
    ):
        result = self.get(model_call_idx, jvp=tangents is not None)
        # print(
        #     f"MODEL: idx={model_call_idx}, size={self.size}, threshold={self.threshold}, cached={result is not None}"
        # )
        if result is not None:
            return (result, True) if return_cached else result

        comfy.model_management.throw_exception_if_processing_interrupted()

        model_options = self.extra_args.get("model_options", {}).copy()
        denoised_cond = denoised_uncond = None

        def postcfg(args):
            nonlocal denoised_cond, denoised_uncond
            denoised_uncond = args["uncond_denoised"]
            denoised_cond = args["cond_denoised"]
            return args["denoised"]

        extra_args = self.extra_args | {
            "model_options": comfy.model_patcher.set_model_options_post_cfg_function(
                model_options, postcfg, disable_cfg1_optimization=True
            )
        }
        s_in = fallback(s_in, self.s_in)

        def call_model(x, sigma, **kwargs):
            return self.model(x, sigma * s_in, **extra_args | kwargs)

        if tangents is None:
            denoised = call_model(x, sigma, **kwargs)
            mr = ModelResult(
                model_call_idx,
                sigma,
                x,
                denoised,
                denoised_uncond=denoised_uncond,
                denoised_cond=denoised_cond,
            )
            self.set(model_call_idx, mr)
            return (mr, False) if return_cached else mr
        denoised, denoised_prime = torch.func.jvp(call_model, (x, sigma), tangents)
        mr = ModelResult(
            model_call_idx,
            sigma,
            x,
            denoised,
            jdenoised=denoised_prime,
            denoised_uncond=denoised_uncond,
            denoised_cond=denoised_cond,
        )
        self.set(model_call_idx, mr)
        return (mr, False) if return_cached else mr

    def ___call__(
        self,
        x,
        sigma,
        *,
        model_call_idx=0,
        tangents=None,
        return_cached=False,
        **kwargs,
    ):
        result = self.get(model_call_idx, jvp=tangents is not None)
        # print(
        #     f"MODEL: idx={model_call_idx}, size={self.size}, threshold={self.threshold}, cached={result is not None}"
        # )
        if result is not None:
            return (result, True) if return_cached else result
        if tangents is None:
            denoised = self.call_model(x, sigma, **kwargs)
            self.set(model_call_idx, denoised)
            return (denoised, False) if return_cached else denoised
        denoised, denoised_prime = torch.func.jvp(self.call_model, (x, sigma), tangents)
        self.set(model_call_idx, denoised, jdenoised=denoised_prime)
        return (
            (denoised, denoised_prime, False)
            if return_cached
            else (denoised, denoised_prime)
        )
