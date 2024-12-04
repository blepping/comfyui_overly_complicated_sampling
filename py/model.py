from collections import namedtuple

import torch

import comfy
from comfy.k_diffusion.sampling import to_d

from . import filtering

from .utils import fallback
from .latent import OCSLatentFormat


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

    def get_split_prediction(
        self,
        *,
        x=None,
        d=None,
        sigma=None,
        denoised=None,
        denoised_uncond=None,
        alt_cfgpp_scale=0,
        cfgpp=False,
    ):
        denoised = fallback(denoised, self.denoised)
        denoised_uncond = fallback(denoised_uncond, self.denoised_uncond)
        x = fallback(x, self.x)
        sigma = fallback(sigma, self.sigma)
        if d is None:
            d = self.to_d(
                x=x,
                sigma=sigma,
                denoised=denoised,
                denoised_uncond=denoised_uncond,
                alt_cfgpp_scale=alt_cfgpp_scale,
                cfgpp=cfgpp,
            )
        denoised_pred = denoised if alt_cfgpp_scale == 0 else x - d * sigma
        return (denoised_pred, d)

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

    def get_error(self, other, *, override=None, alt_cfgpp_scale=0, cfgpp=False):
        slf = fallback(override, self)
        first, second = (other, slf) if other.sigma > slf.sigma else (slf, other)
        if first.sigma == second.sigma:
            return 0.0
        d = first.to_d(alt_cfgpp_scale=alt_cfgpp_scale, cfgpp=cfgpp)
        d_pred = second.to_d(
            x=first.x + d * (second.sigma - first.sigma),
            alt_cfgpp_scale=alt_cfgpp_scale,
            cfgpp=cfgpp,
        )
        return torch.linalg.norm(d_pred.sub_(d)).div_(torch.linalg.norm(d)).item()


ModelCallCacheConfig = namedtuple(
    "ModelCallCacheConfig", ("size", "max_use", "threshold"), defaults=(0, 1000000, 1)
)


class ModelCallCache:
    def __init__(
        self,
        model,
        x: torch.Tensor,
        s_in: torch.Tensor,
        extra_args: dict,
        *,
        cache: None | dict = None,
        filter: None | dict = None,
        cfg1_uncond_optimization: bool = False,
        cfg_scale_override: None | int | float = None,
    ) -> None:
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
        self.cfg1_uncond_optimization = cfg1_uncond_optimization
        self.cfg_scale_override = cfg_scale_override
        self.is_rectified_flow = x.shape[1] == 16 and isinstance(
            model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST
        )
        self.latent_format = OCSLatentFormat(
            x.device, model.inner_model.inner_model.latent_format
        )
        if self.cache.size < 1:
            return
        self.reset_cache()

    def maybe_filter(
        self, name: str, latent: torch.Tensor, *args: list, **kwargs: dict
    ) -> torch.Tensor:
        filt = self.filters.get(name)
        if filt is None:
            return latent
        return filt.apply(latent, *args, **kwargs)

    def filter_result(
        self, result: ModelResult, *args: list, **kwargs: dict
    ) -> ModelResult:
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
    def _fr_add_mr(fr: filtering.FilterRefs, mr: ModelResult) -> filtering.FilterRefs:
        frmr = filtering.FilterRefs.from_mr(mr)
        fr.kvs |= {f"{k}_curr": v for k, v in frmr.kvs.items()}
        return fr

    def reset_cache(self) -> None:
        size = self.cache.size
        self.slot = [None] * size
        self.slot_use = [self.cache.max_use] * size

    def get(self, idx: int, *, jvp: bool = False) -> None | ModelResult:
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

    def set(self, idx: int, mr: ModelResult) -> None:
        idx -= self.cache.threshold
        if idx < 0 or idx >= self.cache.size:
            return
        self.slot_use[idx] = self.cache.max_use
        self.slot[idx] = mr

    def call_model(
        self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: dict
    ) -> torch.Tensor:
        return self.model(x, sigma * self.s_in, **self.extra_args | kwargs)

    @property
    def model_sampling(self):
        return self.model.inner_model.inner_model.model_sampling

    @property
    def inner_cfg_scale(self) -> None | int | float:
        maybe_cfg_scale = getattr(self.model.inner_model, "cfg", None)
        return maybe_cfg_scale if isinstance(maybe_cfg_scale, (int, float)) else None

    def set_inner_cfg_scale(self, scale: None | int | float) -> None | int | float:
        eff_scale = self.cfg_scale_override
        if scale is not None:
            eff_scale = None if scale < 0 else scale
        if eff_scale is None or eff_scale < 0:
            return None
        curr_cfg_scale = self.inner_cfg_scale
        if curr_cfg_scale is None:
            return None
        self.model.inner_model.cfg = eff_scale
        return curr_cfg_scale

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        *,
        call_index: int = 0,
        ss,
        s_in=None,
        tangents=None,
        require_uncond: bool = False,
        cfg_scale_override: None | int = None,
        **kwargs,
    ) -> ModelResult:
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
            return result

        comfy.model_management.throw_exception_if_processing_interrupted()

        model_options = self.extra_args.get("model_options", {}).copy()
        denoised_cond = denoised_uncond = None

        def postcfg(args):
            nonlocal denoised_cond, denoised_uncond
            denoised_uncond = args["uncond_denoised"]
            denoised_cond = args["cond_denoised"]
            if denoised_uncond is None:
                denoised_uncond = denoised_cond
            return args["denoised"]

        orig_cfg_scale = self.set_inner_cfg_scale(cfg_scale_override)

        model_options = comfy.model_patcher.set_model_options_post_cfg_function(
            model_options,
            postcfg,
            disable_cfg1_optimization=require_uncond
            or not self.cfg1_uncond_optimization,
        )

        extra_args = self.extra_args | {"model_options": model_options}
        s_in = fallback(s_in, self.s_in)
        x = self.maybe_filter("input", x, refs=filter_refs)

        def call_model(x, sigma, **kwargs):
            return self.model(x, sigma * s_in, **extra_args | kwargs)

        if tangents is None:
            denoised = call_model(x, sigma, **kwargs)
            self.set_inner_cfg_scale(orig_cfg_scale)
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
            return mr
        denoised, denoised_prime = torch.func.jvp(call_model, (x, sigma), tangents)
        self.set_inner_cfg_scale(orig_cfg_scale)
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
        return mr
