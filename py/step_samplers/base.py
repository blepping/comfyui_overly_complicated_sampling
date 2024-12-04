import contextlib
import typing

import torch

from . import registry  # noqa: F401

from .. import filtering, noise, utils
from ..utils import fallback


class SamplerResult:
    CLONE_KEYS = (
        "denoised_cond",
        "denoised_uncond",
        "denoised",
        "final",
        "is_rectified_flow",
        "noise_pred",
        "noise_sampler",
        "s_noise",
        "sampler",
        "sigma_down",
        "sigma_next",
        "sigma_up",
        "sigma",
        "step",
        "substep",
        "x_",
    )

    def __init__(
        self,
        ss,
        sampler,
        x,
        sigma_up=None,
        *,
        split_result=None,
        sigma=None,
        sigma_next=None,
        sigma_down=None,
        s_noise=None,
        noise_sampler=None,
        final=True,
    ):
        self.is_rectified_flow = ss.model.is_rectified_flow
        self.sampler = sampler
        self.sigma_up = fallback(sigma_up, ss.sigma.new_zeros(1))
        self.s_noise = fallback(s_noise, sampler.s_noise)
        self.sigma = fallback(sigma, ss.sigma)
        self.sigma_next = fallback(sigma_next, ss.sigma_next)
        self.sigma_down = fallback(sigma_down, self.sigma_next)
        self.noise_sampler = fallback(noise_sampler, sampler.noise_sampler)
        self.final = final
        self.step = ss.step
        self.substep = ss.substep
        self.x_ = x
        if split_result is not None:
            self.denoised, self.noise_pred = split_result
        elif x is None:
            raise ValueError("SamplerResult requires at least one of x, split_result")
        else:
            self.denoised = self.noise_pred = None
            _ = self.extract_pred(ss)
        self.denoised_uncond = ss.hcur.denoised_uncond
        self.denoised_cond = ss.hcur.denoised_cond

    def get_noise(self, *, scaled=True, ss=None):
        if self.sigma_next == 0 or self.noise_scale == 0:
            return torch.zeros_like(self.x_)
        return self.noise_sampler(
            self.sigma,
            self.sigma_next,
            out_hw=self.x.shape[-2:],
            x_ref=self.x,
            refs=filtering.FilterRefs.from_sr(self) if ss is None else ss.refs,
        ).mul_(self.noise_scale if scaled else 1.0)

    def extract_pred(self, ss):
        if self.denoised is None or self.noise_pred is None:
            self.denoised, self.noise_pred = utils.extract_pred(
                ss.hcur.x, self.x_, ss.sigma, self.sigma_down
            )
        return self.denoised, self.noise_pred

    @property
    def x(self):
        if self.x_ is None:
            self.x_ = self.denoised + self.sigma_down * self.noise_pred
        return self.x_

    @property
    def noise_scale(self):
        return self.sigma_up * self.s_noise

    def noise_x(self, x=None, scale=1.0, *, ss=None):
        x = fallback(x, self.x)
        if self.sigma_next == 0 or self.noise_scale == 0:
            return x
        noise = self.get_noise(ss=ss) * scale
        if not self.is_rectified_flow:
            return x + noise
        x_coeff = (1 - self.sigma_next) / (1 - self.sigma_down)
        # print(f"\nRF noise: {x_coeff}")
        return x_coeff * x + noise

    def clone(self):
        obj = self.__new__(self.__class__)
        for k in self.CLONE_KEYS:
            if hasattr(self, k):
                setattr(obj, k, getattr(self, k))
        return obj


class StepSamplerContext:
    def __init__(self, sampler, *args, **kwargs):
        self.sampler = sampler
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        if self.sampler.ss is not None:
            raise RuntimeError("Cannot reenter prepared sampler in context manager!")
        self.sampler.prepare(*self.args, **self.kwargs)
        return self.sampler

    def __exit__(self, *_unused):
        self.sampler.reset()


class SingleStepSampler:
    name = None
    self_noise = 0
    model_calls = 0
    ancestralize = False
    sample_sigma_zero = False
    immiscible = None
    allow_cfgpp = False
    allow_alt_cfgpp = False
    afs_end_step = -1

    default_eta = 1.0

    def __init__(
        self,
        *,
        noise_sampler=None,
        substeps=1,
        s_noise=1.0,
        eta=None,
        eta_retry_increment=0,
        dyn_eta_start=None,
        dyn_eta_end=None,
        weight=1.0,
        pre_filter=None,
        post_filter=None,
        immiscible=None,
        **kwargs,
    ):
        self.ss = None
        self.options = kwargs
        self.cfgpp = self.allow_cfgpp and self.options.pop("cfgpp", False) is True
        alt_cfgpp_scale = self.options.pop("alt_cfgpp_scale", 0.0)
        self.alt_cfgpp_scale = 0.0 if not self.allow_alt_cfgpp else alt_cfgpp_scale
        self.s_noise = s_noise
        self.eta = fallback(eta, self.default_eta)
        self.eta_retry_increment = eta_retry_increment
        self.dyn_eta_start = dyn_eta_start
        self.dyn_eta_end = dyn_eta_end
        self.noise_sampler = noise_sampler
        self.immiscible = (
            noise.ImmiscibleNoise(**immiscible)
            if immiscible not in (False, None)
            else immiscible
        )
        self.weight = weight
        self.afs_end_step = self.options.pop("afs_end_step", -1)
        self.substeps = substeps
        self.pre_filter = (
            None if pre_filter is None else filtering.make_filter(pre_filter)
        )
        self.post_filter = (
            None if post_filter is None else filtering.make_filter(post_filter)
        )
        self.custom_noise = self.options.get("custom_noise")
        if isinstance(self.custom_noise, str):
            self.custom_noise = self.options.get(f"custom_noise_{self.custom_noise}")

    def __call__(self, x):
        ss = self.ss
        orig_x = x
        if not self.sample_sigma_zero and ss.sigma_next == 0:
            return (yield from self.denoised_result())
        if ss.step <= self.afs_end_step:
            return (yield from self.afs_step(x))
        if self.pre_filter or self.post_filter:
            filter_refs = ss.refs | filtering.FilterRefs({"orig_x": orig_x})
        if self.pre_filter:
            x = self.pre_filter.apply(x, refs=filter_refs)
        next_x = None
        sg = self.step(x)
        with contextlib.suppress(StopIteration):
            while True:
                sr = sg.send(next_x)
                if sr.final:
                    if self.ancestralize:
                        sr = self.ancestralize_result(sr)
                    curr_x = sr.x
                    if self.post_filter:
                        curr_x = self.post_filter.apply(curr_x, refs=filter_refs)
                    sr.x_ = curr_x
                    return (yield sr)
                next_x = sr.noise_x(ss=ss)

    def step(self, x):
        raise NotImplementedError

    def prepare(self, ss):
        self.ss = ss
        self.noise_sampler = ss.noise.make_caching_noise_sampler(
            self.custom_noise,
            self.max_noise_samples,
            ss.sigma,
            ss.sigma_next,
            immiscible=fallback(self.immiscible, ss.noise.immiscible),
        )

    def reset(self):
        self.ss = None
        self.noise_sampler = None

    def afs_step(self, x):
        sigma, sigma_next = self.ss.sigma, self.ss.sigma_next
        afs_d = x / ((1 + sigma**2).sqrt())
        dt = sigma_next - sigma
        return (yield from self.result(x + afs_d * dt))

    # Euler - based on original ComfyUI implementation
    def euler_step(
        self,
        x,
        *,
        sigma_down=None,
        sigma_up=None,
        eta=None,
        sigma=None,
        sigma_next=None,
    ):
        eta = fallback(eta, self.get_dyn_eta())
        if sigma_down is None or sigma_up is None:
            if not (sigma_down is None and sigma_up is None):
                raise ValueError("Must pass both sigma_down and sigma_up or neither")
            sigma_down, sigma_up = self.get_ancestral_step(
                eta=eta, sigma=sigma, sigma_next=sigma_next
            )
        return (
            yield from self.split_result(
                *self.get_split_prediction(), sigma_down=sigma_down, sigma_up=sigma_up
            )
        )

    def denoised_result(self, **kwargs):
        ss = self.ss
        return (
            yield SamplerResult(ss, self, ss.denoised, ss.sigma.new_zeros(1), **kwargs)
        )

    def result(self, x, noise_scale=None, **kwargs):
        return (yield SamplerResult(self.ss, self, x, noise_scale, **kwargs))

    def split_result(
        self, denoised=None, noise_pred=None, sigma_up=None, sigma_down=None, **kwargs
    ):
        return (
            yield SamplerResult(
                ss=self.ss,
                sampler=self,
                x=None,
                sigma_up=sigma_up,
                sigma_down=sigma_down,
                split_result=(denoised, noise_pred),
                **kwargs,
            )
        )

    def get_ancestral_step(
        self, *args, dyn_eta=False, as_dict=False, retry_increment=None, **kwargs
    ):
        if dyn_eta:
            args = (self.get_dyn_eta(), *args)
        retry_increment = fallback(retry_increment, self.eta_retry_increment)
        sigma_down, sigma_up = self.ss.get_ancestral_step(
            *args, retry_increment=retry_increment, **kwargs
        )
        if not as_dict:
            return sigma_down, sigma_up
        return {"sigma_down": sigma_down, "sigma_up": sigma_up}

    def ancestralize_result(self, sr):
        ss = self.ss
        new_sr = sr.clone()
        if new_sr.sigma_down is not None and new_sr.sigma_down != new_sr.sigma_next:
            return sr
        eta = self.get_dyn_eta()
        if sr.sigma_next == 0 or eta == 0:
            return sr
        sd, su = self.get_ancestral_step(eta, sigma=sr.sigma, sigma_next=sr.sigma_next)
        _ = new_sr.extract_pred(ss)
        new_sr.x_ = None
        new_sr.sigma_up = su
        new_sr.sigma_down = sd
        return new_sr

    def __str__(self):
        return f"<SS({self.name}): s_noise={self.s_noise}, eta={self.eta}>"

    def get_dyn_value(self, start, end):
        if None in (start, end):
            return 1.0
        if start == end:
            return start
        ss = self.ss
        main_idx = getattr(ss, "main_idx", ss.idx)
        main_sigmas = getattr(ss, "main_sigmas", ss.sigmas)
        step_pct = main_idx / (len(main_sigmas) - 1)
        dd_diff = end - start
        return start + dd_diff * step_pct

    def get_dyn_eta(self):
        return self.eta * self.get_dyn_value(self.dyn_eta_start, self.dyn_eta_end)

    @property
    def max_noise_samples(self):
        return (1 + self.self_noise) * self.substeps

    @property
    def require_uncond(self):
        return self.cfgpp or self.alt_cfgpp_scale != 0

    def to_d(self, mr, *, use_cfgpp=True, **kwargs):
        if not use_cfgpp:
            return mr.to_d(**kwargs)
        return mr.to_d(alt_cfgpp_scale=self.alt_cfgpp_scale, cfgpp=self.cfgpp, **kwargs)

    def get_split_prediction(self, *, mr=None, sigma=None, **kwargs):
        mr = fallback(mr, self.ss.hcur)
        sigma = fallback(sigma, mr.sigma)
        return mr.get_split_prediction(
            sigma=sigma,
            alt_cfgpp_scale=self.alt_cfgpp_scale,
            cfgpp=self.cfgpp,
            **kwargs,
        )

    def call_model(self, *args, **kwargs):
        ss = self.ss
        kwargs["require_uncond"] = self.require_uncond or kwargs.get(
            "require_uncond", False
        )
        kwargs["cfg_scale_override"] = kwargs.get(
            "cfg_scale_override",
            self.options.get("cfg_scale_override", ss.cfg_scale_override),
        )
        return ss.call_model(*args, ss=ss, **kwargs)

    def step_mix(self, x, denoised, uncond, ratio, *, blend=torch.lerp):
        if self.cfgpp:
            return denoised + (x - uncond).mul_(ratio)
        pp = self.alt_cfgpp_scale
        if pp == 0:
            return blend(denoised, x, ratio)
        return blend(denoised * (1 + pp) - uncond * pp, x, ratio)


class HistorySingleStepSampler(SingleStepSampler):
    default_history_limit, max_history = 0, 0

    def __init__(self, *args, history_limit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_limit = min(
            self.max_history,
            max(
                0,
                self.default_history_limit if history_limit is None else history_limit,
            ),
        )

    def available_history(self):
        ss = self.ss
        return max(
            0, min(ss.idx, self.history_limit, self.max_history, len(ss.hist) - 1)
        )


class ReversibleConfig(typing.NamedTuple):
    scale: float
    eta: float
    dyn_eta_start: float | None = None
    dyn_eta_end: float | None = None
    eta_retry_increment: float = 0.0
    start_step: int = 0
    end_step: int = 9999
    use_cfgpp: bool = False

    @classmethod
    def build(cls, *, default_eta, default_scale, eta=None, scale=None, **kwargs):
        return cls.__new__(
            cls,
            eta=fallback(eta, default_eta),
            scale=fallback(scale, default_scale),
            **kwargs,
        )

    def check(self, step):
        return self.scale != 0 and self.start_step <= step <= self.end_step


class ReversibleSingleStepSampler(HistorySingleStepSampler):
    default_reversible_scale = 1.0
    default_reta = 1.0

    def __init__(
        self,
        *,
        reversible_scale=None,
        reta=None,
        dyn_reta_start=None,
        dyn_reta_end=None,
        reversible_start_step=0,
        reversible=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if reversible is None:
            # For backward compatibility.
            self.reversible = ReversibleConfig.build(
                default_eta=self.default_reta,
                default_scale=self.default_reversible_scale,
                scale=reversible_scale,
                eta=reta,
                dyn_eta_start=dyn_reta_start,
                dyn_eta_end=dyn_reta_end,
                start_step=reversible_start_step,
            )
            return
        self.reversible = ReversibleConfig.build(
            default_eta=self.default_reta,
            default_scale=self.default_reversible_scale,
            **reversible,
        )

    def reversible_correction(self):
        raise NotImplementedError

    def get_dyn_reta(self, *, r=None):
        r = fallback(r, self.reversible)
        ss = self.ss
        if not r.check(ss.step):
            return 0.0
        return r.eta * self.get_dyn_value(r.dyn_eta_start, r.dyn_eta_end)

    dyn_reta = property(get_dyn_reta)

    def get_reversible_cfg(self, *, reversible=None):
        reversible = fallback(reversible, self.reversible)
        ss = self.ss
        if not reversible.check(ss.step):
            return 0.0, 0.0
        return self.get_dyn_reta(r=reversible), reversible.scale


class DPMPPStepMixin:
    @staticmethod
    def sigma_fn(t):
        return t.neg().exp()

    @staticmethod
    def t_fn(t):
        return t.log().neg()


class MinSigmaStepMixin:
    @staticmethod
    def adjust_step(sigma, min_sigma, threshold=5e-04):
        if min_sigma - sigma > threshold:
            return sigma.clamp(min=min_sigma)
        return sigma

    def adjusted_step(self, sn, result, mcc, sigma_up):
        ss = self.ss
        if sn == ss.sigma_next:
            return sigma_up, result
        # FIXME: Make sure we're noising from the right sigma.
        result = yield from self.result(
            result, sigma_up, sigma=ss.sigma, sigma_next=sn, final=False
        )
        mr = self.call_model(result, sn, call_index=mcc)
        dt = ss.sigma_next - sn
        result = result + self.to_d(mr) * dt
        return sigma_up.new_zeros(1), result
