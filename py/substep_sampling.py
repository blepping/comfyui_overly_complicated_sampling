import torch

import comfy
from comfy.k_diffusion.sampling import get_ancestral_step, to_d

from .utils import fallback


class Items:
    def __init__(self, items=None):
        self.items = [] if items is None else items

    def clone(self):
        return self.__class__(items=self.items.copy())

    def append(self, item):
        self.items.append(item)
        return item

    def __getitem__(self, key):
        return self.items[key]

    def __setitem__(self, key, value):
        self.items[key] = value

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return self.items.__iter__()


class CommonOptionsItems(Items):
    def __init__(self, *, s_noise=1.0, eta=1.0, items=None, **kwargs):
        super().__init__(items=items)
        self.options = kwargs
        self.s_noise = s_noise
        self.eta = eta

    def clone(self):
        obj = super().clone()
        obj.options = self.options.copy()
        obj.s_noise = self.s_noise
        obj.eta = self.eta
        return obj


class StepSamplerChain(CommonOptionsItems):
    def __init__(
        self,
        *,
        merge_method="divide",
        time_mode="step",
        time_start=0,
        time_end=999,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # step, step_pct, sigma
        self.merge_method = merge_method
        self.time_mode = time_mode
        self.time_start, self.time_end = time_start, time_end

    def check_time(self, sigma, step, steps):
        step_pct = step / steps if steps != 0 else 0.0
        if self.time_mode == "step":
            return self.time_start <= step <= self.time_end
        if self.time_mode == "step_pct":
            return self.time_start <= step_pct <= self.time_end
        if self.time_mode == "sigma":
            return self.time_start >= sigma >= self.time_end
        raise ValueError("Bad time mode")

    def clone(self):
        obj = super().clone()
        obj.merge_method = self.merge_method
        obj.time_mode = self.time_mode
        obj.time_start, obj.time_end = self.time_start, self.time_end
        obj.options = self.options.copy()
        return obj


class ParamGroup(Items):
    pass


class StepSamplerGroups(CommonOptionsItems):
    def find_match(self, sigma, step, steps):
        for idx, item in enumerate(self.items):
            if item.check_time(sigma, step, steps):
                return idx
        return None


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
        return self.model(x, sigma * self.s_in, **self.extra_args, **kwargs)

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
            return self.model(x, sigma * s_in, **extra_args, **kwargs)

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


class SamplerState:
    def __init__(
        self,
        model,
        sigmas,
        idx,
        extra_args,
        *,
        step=0,
        noise_sampler,
        callback=None,
        denoised=None,
        noise=None,
        eta=1.0,
        reta=1.0,
        s_noise=1.0,
        disable_status=False,
    ):
        self.model = model
        self.hist = History(4)
        self.extra_args = extra_args
        self.eta = eta
        self.reta = reta
        self.s_noise = s_noise
        self.sigmas = sigmas
        self.callback_ = callback
        self.noise_sampler = noise_sampler
        self.noise = noise
        self.disable_status = disable_status
        self.update(idx)

    @property
    def hcur(self):
        return self.hist[-1]

    @property
    def hprev(self):
        return self.hist[-2]

    @property
    def denoised(self):
        return self.hcur.denoised

    @property
    def dt(self):
        return self.sigma_next - self.sigma

    @property
    def d(self):
        return self.hcur.d

    def update(self, idx=None, step=None):
        idx = self.idx if idx is None else idx
        self.idx = idx
        self.sigma_prev = None if idx < 1 else self.sigmas[idx - 1]
        self.sigma, self.sigma_next = self.sigmas[idx], self.sigmas[idx + 1]
        self.sigma_down, self.sigma_up = get_ancestral_step(
            self.sigma, self.sigma_next, eta=self.eta
        )
        if step is not None:
            self.step = step

    def get_ancestral_step(self, eta=1.0, sigma=None, sigma_next=None):
        sigma = self.sigma if sigma is None else sigma
        sigma_next = self.sigma_next if sigma_next is None else sigma_next
        sd, su = (
            v if isinstance(v, torch.Tensor) else sigma.new_full((1,), v)
            for v in get_ancestral_step(
                sigma, sigma_next, eta=eta if sigma_next != 0 else 0
            )
        )
        return sd, su

    def clone_edit(self, **kwargs):
        obj = self.__class__.__new__(self.__class__)
        for k in (
            "model",
            "hist",
            "extra_args",
            "disable_status",
            "eta",
            "reta",
            "s_noise",
            "sigmas",
            "callback_",
            "noise_sampler",
            "noise",
            "idx",
            "step",
            "sigma",
            "sigma_next",
            "sigma_prev",
            "sigma_down",
            "sigma_up",
        ):
            setattr(obj, k, kwargs[k] if k in kwargs else getattr(self, k))
        obj.update()
        return obj

    def callback(self, hi=None):
        if not self.callback_:
            return None
        hi = self.hcur if hi is None else hi
        return self.callback_({
            "x": hi.x,
            "i": self.step,
            "sigma": hi.sigma,
            "sigma_hat": hi.sigma,
            "denoised": hi.denoised,
        })

    def reset(self):
        self.hist.reset()
        self.denoised = None
