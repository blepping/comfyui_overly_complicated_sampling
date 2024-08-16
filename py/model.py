from collections import namedtuple

import torch

import comfy
from comfy.k_diffusion.sampling import to_d

from . import filtering

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


ModelCallCacheConfig = namedtuple(
    "ModelCallCacheConfig", ("size", "max_use", "threshold"), defaults=(0, 1000000, 1)
)


class ModelCallCache:
    def __init__(
        self,
        model,
        x,
        s_in,
        extra_args,
        *,
        cache=None,
        filter=None,
    ):
        self.cache = ModelCallCacheConfig(**fallback(cache, {}))
        filtargs = fallback(filter, {}).copy()
        self.filters = {}
        for key in ("input", "denoised", "jdenoised", "cond", "uncond", "x"):
            filt = filtargs.pop(key, None)
            if filt is None:
                continue
            self.filters[key] = filtering.make_filter(filt)
        self.model = model
        self.s_in = s_in
        self.extra_args = extra_args
        if self.cache.size < 1:
            return
        self.reset_cache()

    def maybe_filter(self, name, latent, *args, **kwargs):
        filt = self.filters.get(name)
        if filt is None:
            return latent
        return filt.apply(latent, *args, **kwargs)

    def filter_result(self, result, *args, **kwargs):
        if not self.filters:
            return result
        result = result.clone()
        for key in ("denoised", "cond", "uncond", "jdenoised", "x"):
            filt = self.filters.get(key)
            if filt is None:
                continue
            attk = f"denoised_{key}" if key in ("cond", "uncond") else key
            inpval = getattr(result, attk, None)
            if inpval is None:
                continue
            setattr(result, attk, filt.apply(inpval, *args, **kwargs))
        return result

    @staticmethod
    def _fr_add_mr(fr, mr):
        frmr = filtering.FilterRefs.from_mr(mr)
        fr.kvs |= {f"{k}_curr": v for k, v in frmr.kvs.items()}
        return fr

    def reset_cache(self):
        size = self.cache.size
        self.slot = [None] * size
        self.slot_use = [self.cache.max_use] * size

    def get(self, idx, *, jvp=False):
        idx -= self.cache.threshold
        if (
            idx >= self.cache.size
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
        idx -= self.cache.threshold
        if idx < 0 or idx >= self.cache.size:
            return
        self.slot_use[idx] = self.cache.max_use
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
        call_index=0,
        ss,
        s_in=None,
        tangents=None,
        return_cached=False,
        **kwargs,
    ):
        filter_refs = ss.refs | filtering.FilterRefs({
            "model_call": call_index,
            "orig_x": x,
        })
        result = self.get(call_index, jvp=tangents is not None)
        # print(
        #     f"MODEL: idx={call_index}, size={self.size}, threshold={self.threshold}, cached={result is not None}"
        # )
        if result is not None:
            self._fr_add_mr(filter_refs, result)
            result = self.filter_result(result, default_ref=x, refs=filter_refs)
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
        x = self.maybe_filter("input", x, refs=filter_refs)

        def call_model(x, sigma, **kwargs):
            return self.model(x, sigma * s_in, **extra_args | kwargs)

        if tangents is None:
            denoised = call_model(x, sigma, **kwargs)
            mr = ModelResult(
                call_index,
                sigma,
                x,
                denoised,
                denoised_uncond=denoised_uncond,
                denoised_cond=denoised_cond,
            )
            self.set(call_index, mr)
            self._fr_add_mr(filter_refs, mr)
            mr = self.filter_result(mr, default_ref=x, refs=filter_refs)
            return (mr, False) if return_cached else mr
        denoised, denoised_prime = torch.func.jvp(call_model, (x, sigma), tangents)
        mr = ModelResult(
            call_index,
            sigma,
            x,
            denoised,
            jdenoised=denoised_prime,
            denoised_uncond=denoised_uncond,
            denoised_cond=denoised_cond,
        )
        self.set(call_index, mr)
        self._fr_add_mr(filter_refs, mr)
        mr = self.filter_result(mr, default_ref=x, refs=filter_refs)
        return (mr, False) if return_cached else mr
