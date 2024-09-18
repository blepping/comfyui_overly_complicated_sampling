import contextlib
import inspect
import math
import os
import typing
import warnings

import torch
import tqdm
import torchsde
import numpy

import comfy
from comfy.k_diffusion.sampling import (
    get_ancestral_step,
    to_d,
)

from . import filtering, noise, res_support, utils
from . import expression as expr
from .utils import fallback

HAVE_DIFFRAX = HAVE_TDE = HAVE_TODE = False

with contextlib.suppress(ImportError):
    import torchdiffeq as tde

    HAVE_TDE = True

with contextlib.suppress(ImportError, RuntimeError):
    import torchode as tode

    HAVE_TODE = True

if not os.environ.get("COMFYUI_OCS_NO_DIFFRAX_SOLVER"):
    with contextlib.suppress(ImportError):
        import diffrax
        import jax

        if not os.environ.get("COMFYUI_OCS_NO_DISABLE_JAX_PREALLOCATE"):
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # jax.config.update("jax_enable_x64", True)

        HAVE_DIFFRAX = True


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

    def get_ancestral_step(self, *args, dyn_eta=False, as_dict=False, **kwargs):
        if dyn_eta:
            args = (self.get_dyn_eta(), *args)
        sigma_down, sigma_up = self.ss.get_ancestral_step(
            *args, retry_increment=self.eta_retry_increment, **kwargs
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

    def to_d(self, mr, **kwargs):
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
            "cfg_scale_override", ss.cfg_scale_override
        )
        return ss.call_model(*args, ss=ss, **kwargs)


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reversible_scale = fallback(
            reversible_scale, self.default_reversible_scale
        )
        self.reta = fallback(reta, self.default_reta)
        self.reversible_start_step = reversible_start_step
        self.dyn_reta_start = dyn_reta_start
        self.dyn_reta_end = dyn_reta_end

    def reversible_correction(self):
        raise NotImplementedError

    def get_dyn_reta(self):
        ss = self.ss
        if ss.step < self.reversible_start_step:
            return 0.0
        return self.reta * self.get_dyn_value(self.dyn_reta_start, self.dyn_reta_end)

    def get_reversible_cfg(self):
        ss = self.ss
        if ss.step < self.reversible_start_step:
            return 0.0, 0.0
        return self.get_dyn_reta(), self.reversible_scale


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


class EulerStep(SingleStepSampler):
    name = "euler"
    allow_cfgpp = True
    allow_alt_cfgpp = True
    step = SingleStepSampler.euler_step


class CycleSingleStepSampler(SingleStepSampler):
    default_eta = 0.0

    def __init__(self, *, cycle_pct=0.25, **kwargs):
        super().__init__(**kwargs)
        if cycle_pct < 0:
            raise ValueError("cycle_pct must be positive")
        self.cycle_pct = cycle_pct

    def get_cycle_scales(self, sigma_next):
        keep_scale = 1 - self.cycle_pct
        add_scale = ((sigma_next**2.0 - (keep_scale * sigma_next) ** 2.0) ** 0.5) * (
            0.95 + 0.25 * self.cycle_pct
        )
        # print(f">> keep={keep_scale}, add={add_scale}")
        return keep_scale, add_scale


class EulerCycleStep(CycleSingleStepSampler):
    name = "euler_cycle"
    allow_alt_cfgpp = True
    allow_cfgpp = True

    def step(self, x):
        sigma_next = self.ss.sigma_next
        denoised_pred, d = self.get_split_prediction()
        keep_scale, add_scale = self.get_cycle_scales(sigma_next)
        return (
            yield from self.split_result(
                denoised_pred, d * keep_scale, sigma_up=add_scale, sigma_down=sigma_next
            )
        )


class DPMPP2MStep(HistorySingleStepSampler, DPMPPStepMixin):
    name = "dpmpp_2m"
    default_history_limit, max_history = 1, 1
    ancestralize = True
    default_eta = 0.0

    def step(self, x):
        ss = self.ss
        s, sn = ss.sigma, ss.sigma_next
        t, t_next = self.t_fn(s), self.t_fn(sn)
        h = t_next - t
        st, st_next = self.sigma_fn(t), self.sigma_fn(t_next)
        if self.available_history() > 0:
            h_last = t - self.t_fn(ss.sigma_prev)
            r = h_last / h
            denoised, old_denoised = ss.denoised, ss.hprev.denoised
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
        else:
            denoised_d = ss.denoised
        yield from self.result((st_next / st) * x - (-h).expm1() * denoised_d)


class DPMPP2MSDEStep(HistorySingleStepSampler):
    name = "dpmpp_2m_sde"
    default_history_limit, max_history = 1, 1

    def __init__(self, *, solver_type="midpoint", **kwargs):
        super().__init__(**kwargs)
        self.solver_type = solver_type

    def step(self, x):
        ss = self.ss
        denoised = ss.denoised
        # DPM-Solver++(2M) SDE
        t, s = -ss.sigma.log(), -ss.sigma_next.log()
        h = s - t
        eta_h = self.get_dyn_eta() * h

        x = (
            ss.sigma_next / ss.sigma * (-eta_h).exp() * x
            + (-h - eta_h).expm1().neg() * denoised
        )
        noise_strength = ss.sigma_next * (-2 * eta_h).expm1().neg().sqrt()
        if self.available_history() == 0:
            return (yield from self.result(x, noise_strength))
        h_last = (-ss.sigma.log()) - (-ss.sigma_prev.log())
        r = h_last / h
        old_denoised = ss.hprev.denoised
        if self.solver_type == "heun":
            x = x + (
                ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1)
                * (1 / r)
                * (denoised - old_denoised)
            )
        elif self.solver_type == "midpoint":
            x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (
                denoised - old_denoised
            )
        yield from self.result(x, noise_strength)


class DPMPP3MSDEStep(HistorySingleStepSampler):
    name = "dpmpp_3m_sde"
    default_history_limit, max_history = 2, 2

    def step(self, x):
        ss = self.ss
        denoised = ss.denoised
        t, s = -ss.sigma.log(), -ss.sigma_next.log()
        h = s - t
        eta = self.get_dyn_eta()
        h_eta = h * (eta + 1)

        x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised
        noise_strength = ss.sigma_next * (-2 * h * eta).expm1().neg().sqrt()
        ah = self.available_history()
        if ah == 0:
            return (yield from self.result(x, noise_strength))
        hist = ss.hist
        h_1 = (-ss.sigma.log()) - (-ss.sigma_prev.log())
        denoised_1 = hist[-2].denoised
        if ah == 1:
            r = h_1 / h
            d = (denoised - denoised_1) / r
            phi_2 = h_eta.neg().expm1() / h_eta + 1
            x = x + phi_2 * d
        else:  # 2+ history items available
            h_2 = (-ss.sigma_prev.log()) - (-ss.sigmas[ss.idx - 2].log())
            denoised_2 = hist[-3].denoised
            r0 = h_1 / h
            r1 = h_2 / h
            d1_0 = (denoised - denoised_1) / r0
            d1_1 = (denoised_1 - denoised_2) / r1
            d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
            d2 = (d1_0 - d1_1) / (r0 + r1)
            phi_2 = h_eta.neg().expm1() / h_eta + 1
            phi_3 = phi_2 / h_eta - 0.5
            x = x + phi_2 * d1 - phi_3 * d2
        yield from self.result(x, noise_strength)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
# Apparently the only difference between Heun and Trapezoidal is the first step using ETA or not.
class ReversibleHeunStep(ReversibleSingleStepSampler):
    name = "reversible_heun"
    model_calls = 1
    allow_alt_cfgpp = True
    allow_cfgpp = True
    trapezoidal_mode = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blend = filtering.BLENDING_MODES[self.options.get("blend_mode", "lerp")]

    def step_altcfgpp(self, x):
        ss, pp = self.ss, self.alt_cfgpp_scale

        def alt_cfgpp_mix(denoised, uncond):
            return denoised * (1 + pp) - uncond * pp

        sigma, _sigma_next = ss.sigma, ss.sigma_next
        denoised, uncond = ss.denoised, ss.denoised_uncond
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        # reta, reversible_scale = self.get_reversible_cfg()
        # sigma_down_reversible, _sigma_up_reversible = self.get_ancestral_step(reta)
        # dt_reversible = sigma_down_reversible - ss.sigma
        ratio = sigma_down / sigma
        dratio = 1 - (sigma / sigma_down) * 0.5
        x_next = self.blend(alt_cfgpp_mix(denoised, uncond), x, ratio)
        mr_next = self.call_model(x_next, sigma_down, call_index=1)
        denoised_prime = self.blend(mr_next.denoised, denoised, dratio)
        uncond_prime = self.blend(mr_next.denoised_uncond, uncond, dratio)
        x = self.blend(alt_cfgpp_mix(denoised_prime, uncond_prime), x, ratio)
        yield from self.result(x, sigma_up)

    def step_cfgpp(self, x):
        ss = self.ss
        sigma, _sigma_next = ss.sigma, ss.sigma_next
        denoised, uncond = ss.denoised, ss.denoised_uncond
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        # reta, reversible_scale = self.get_reversible_cfg()
        # sigma_down_reversible, _sigma_up_reversible = self.get_ancestral_step(reta)
        # dt_reversible = sigma_down_reversible - ss.sigma
        ratio = sigma_down / sigma
        dratio = 1 - (sigma / sigma_down) * 0.5
        x_next = denoised + (x - uncond) * ratio
        mr_next = self.call_model(x_next, sigma_down, call_index=1)
        denoised_prime = self.blend(mr_next.denoised, denoised, dratio)
        uncond_adj = uncond + mr_next.denoised_uncond * (1 - dratio)
        x = denoised_prime + denoised * 0.5 + (x - uncond_adj) * ratio
        yield from self.result(x, sigma_up)

    def step(self, x):
        if self.cfgpp:
            return (yield from self.step_cfgpp(x))
        if self.alt_cfgpp_scale != 0:
            return (yield from self.step_altcfgpp(x))
        ss = self.ss
        sigma, _sigma_next, denoised = ss.sigma, ss.sigma_next, ss.denoised
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        # reta, reversible_scale = self.get_reversible_cfg()
        # sigma_down_reversible, _sigma_up_reversible = self.get_ancestral_step(reta)
        # dt_reversible = sigma_down_reversible - ss.sigma
        ratio = sigma_down / sigma
        dratio = 1 - (sigma / sigma_down) * 0.5
        x_next = self.blend(denoised, x, ratio)
        mr_next = self.call_model(x_next, sigma_down, call_index=1)
        denoised_prime = self.blend(mr_next.denoised, denoised, dratio)
        x = self.blend(denoised_prime, x, ratio)
        yield from self.result(x, sigma_up)

    # def step(self, x):
    #     if self.cfgpp:
    #         return (yield from self.step_cfgpp(x))
    #     ss = self.ss
    #     sigma, sigma_next, denoised = ss.sigma, ss.sigma_next, ss.denoised
    #     sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
    #     reta, reversible_scale = self.get_reversible_cfg()
    #     sigma_down_reversible, _sigma_up_reversible = self.get_ancestral_step(reta)
    #     dt_reversible = sigma_down_reversible - ss.sigma
    #     ratio = sigma_down / sigma
    #     iratio = 1 - ratio
    #     dratio = 1 - (sigma / sigma_down) * 0.5
    #     x_next = denoised + (x - denoised) * (sigma_next / sigma)
    #     # x_next = denoised * iratio + x * ratio
    #     # if sigma_down < sigma_next:
    #     #     x_next = yield from self.result(
    #     #         x_next,
    #     #         sigma_up,
    #     #         sigma=sigma,
    #     #         sigma_down=sigma_down,
    #     #         final=False,
    #     #     )
    #     # yield from self.result(x_next, sigma_up * 0)
    #     # return
    #     mr_next = self.call_model(x_next, sigma_next, call_index=1)
    #     # yield from self.result(mr_next.denoised * iratio + x_next * ratio, sigma_up)
    #     # return
    #     denoised_prime = (denoised * dratio) + (mr_next.denoised * (1 - dratio))
    #     x = denoised_prime * iratio + x * ratio
    #     yield from self.result(x, sigma_up)

    def stepz(self, x):
        ss = self.ss
        sigma, sigma_next, denoised = ss.sigma, ss.sigma_next, ss.denoised
        ratio = sigma_next / sigma
        iratio = 1 - ratio
        x_next = denoised * iratio + x * ratio
        mr_next = self.call_model(x_next, sigma_next, call_index=1)
        x = (denoised + mr_next.denoised) * (iratio / 2) + x * ratio
        # x = (denoised * ratio + mr_next.denoised * iratio) * iratio + x * ratio
        # denoised_next = mr_next.denoised * (sigma / sigma_next)
        # x = (denoised * ratio + denoised_next * iratio) * iratio + x * ratio
        yield from self.result(x, sigma * 0.0)

    # def step(self, x):
    #     ss = self.ss
    #     sigma, sigma_next, denoised = ss.sigma, ss.sigma_next, ss.denoised
    #     ratio = sigma_next / sigma
    #     iratio = 1 - ratio
    #     x_next = denoised * iratio + x * ratio
    #     mr_next = self.call_model(x_next, sigma_next, call_index=1)
    #     w = 2 * ss.sigmas[0]
    #     w2 = sigma_next / w
    #     w1 = 1 - w2
    #     print("\n>>", w, w1, w2, ratio, iratio)
    #     x = (denoised * w1 + mr_next.denoised * w2) * iratio + x * ratio
    #     yield from self.result(x, sigma * 0.0)

    # def step(self, x):
    #     ss = self.ss
    #     sigma, sigma_next, denoised = ss.sigma, ss.sigma_next, ss.denoised
    #     ratio = sigma_next / sigma
    #     iratio = 1 - ratio
    #     x_next = denoised * iratio + x * ratio
    #     mr_next = self.call_model(x_next, sigma_next, call_index=1)
    #     x = (denoised + mr_next.denoised) * (iratio / 2) + x * ratio
    #     yield from self.result(x, sigma * 0.0)

    def step__(self, x):
        ss = self.ss
        sigma, sigma_next, denoised = ss.sigma, ss.sigma_next, ss.denoised
        uncond = ss.denoised_uncond
        ratio = sigma_next / sigma
        iratio = 1 - ratio
        print(f">>> ratio={ratio:.4}, iratio={iratio:.4}")
        if self.cfgpp:
            x = denoised + (x - uncond) * ratio
        elif self.alt_cfgpp_scale != 0:
            alt_scale = self.alt_cfgpp_scale
            x = (denoised * (1 + alt_scale) - uncond * alt_scale) * iratio + x * ratio
        else:
            x = denoised * iratio + x * ratio
        yield from self.result(x, sigma * 0.0)

    def step_(self, x):
        ss = self.ss
        sigma_next = ss.sigma_next
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        reta, reversible_scale = self.get_reversible_cfg()
        sigma_down_reversible, _sigma_up_reversible = self.get_ancestral_step(reta)
        dt_reversible = sigma_down_reversible - ss.sigma
        print("\n>>>", sigma_up, sigma_next, sigma_down)

        # Predict the sample at the next sigma using Euler step
        euler_sr = next(
            self.euler_step(
                x,
                sigma_down=sigma_next if self.trapezoidal_mode else sigma_down,
                sigma_up=0,
            )
        )
        # yield from self.result(euler_sr.x)
        # return
        d = euler_sr.noise_pred
        # d = to_d(x, ss.sigma, euler_sr.denoised)
        # yield from self.split_result(
        #     euler_sr.denoised,
        #     d,
        #     sigma_up=sigma_up,
        #     sigma_down=sigma_down,
        # )
        # return

        # Denoised sample at the next sigma
        mr_next = self.call_model(euler_sr.x, euler_sr.sigma_down, call_index=1)

        denoised_pred_next, d_next = self.get_split_prediction(mr=mr_next)
        derp = euler_sr.x - euler_sr.noise_pred * euler_sr.sigma_down
        # yield from self.result(x + ((d + d_next) / 2) * (sigma_down - ss.sigma))
        print("\n::", torch.allclose(ss.denoised, derp))
        yield from self.result(
            ((mr_next.denoised + derp) / 2) + ((d + d_next) / 2) * sigma_down
        )
        return
        yield from self.split_result(
            euler_sr.denoised,
            d_next,
            sigma_up=sigma_up,
            sigma_down=sigma_down,
        )
        return
        if reversible_scale != 0:
            denoised_pred_next -= (dt_reversible**2 * (d_next - d) / 4).mul_(
                reversible_scale
            )
        yield from self.split_result(
            denoised_pred_next,
            d.add_(d_next).mul_(0.5),
            sigma_up=sigma_up,
            sigma_down=sigma_down,
        )


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class ReversibleHeun1SStep(ReversibleSingleStepSampler):
    name = "reversible_heun_1s"
    model_calls = 1
    default_history_limit, max_history = 1, 1
    allow_alt_cfgpp = True
    allow_cfgpp = True

    def step(self, x):
        if self.available_history() < 1:
            return (yield from ReversibleHeunStep.step(self, x))
        ss = self.ss
        sigma_next = ss.sigma_next
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        reta, reversible_scale = self.get_reversible_cfg()
        sigma_down_reversible, _sigma_up_reversible = self.get_ancestral_step(reta)
        dt_reversible = sigma_down_reversible - ss.sigma

        # Predict the sample at the next sigma using Euler step
        denoised_pred_prev, d_prev = ss.hprev.get_split_prediction(
            x=x, cfgpp=self.cfgpp, alt_cfgpp_scale=self.alt_cfgpp_scale
        )
        x_pred = denoised_pred_prev + d_prev * ss.sigma
        # euler_sr = next(self.euler_step(x, sigma_down=sigma_next, sigma_up=0))
        # d = euler_sr.noise_pred

        # Denoised sample at the next sigma
        # mr_next = self.call_model(x_pred, sigma_next, call_index=1)

        # denoised_pred_next, d_next = self.get_split_prediction(mr=mr_next)
        denoised_pred_next, d_next = ss.hcur.get_split_prediction(
            x=x_pred, cfgpp=self.cfgpp, alt_cfgpp_scale=self.alt_cfgpp_scale
        )
        correction = dt_reversible**2 * (d_next - d_prev) / 4
        yield from self.split_result(
            denoised_pred_next - correction * reversible_scale,
            (d_prev + d_next) * 0.5,
            sigma_up=sigma_up,
            sigma_down=sigma_down,
        )


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class RESStep(SingleStepSampler):
    name = "res"
    model_calls = 1
    allow_alt_cfgpp = True  # May not be implemented correctly.

    def __init__(self, *, res_simple_phi=False, res_c2=0.5, **kwargs):
        super().__init__(**kwargs)
        self.simple_phi = res_simple_phi
        self.c2 = res_c2

    def step(self, x):
        ss = self.ss
        eta = self.get_dyn_eta()
        sigma_down, sigma_up = self.get_ancestral_step(eta)
        denoised = ss.denoised
        lam_next = sigma_down.log().neg() if eta != 0 else ss.sigma_next.log().neg()
        lam = ss.sigma.log().neg()

        h = lam_next - lam
        a2_1, b1, b2 = res_support._de_second_order(
            h=h, c2=self.c2, simple_phi_calc=self.simple_phi
        )

        c2_h = 0.5 * h

        eff_x = (
            x
            if self.alt_cfgpp_scale == 0 or ss.hcur.denoised_uncond is None
            else x + (ss.denoised - ss.hcur.denoised_uncond) * self.alt_cfgpp_scale
        )
        x_2 = math.exp(-c2_h) * eff_x + a2_1 * h * denoised
        lam_2 = lam + c2_h
        sigma_2 = lam_2.neg().exp()

        denoised2 = self.call_model(x_2, sigma_2, call_index=1).denoised

        x = math.exp(-h) * eff_x + h * (b1 * denoised + b2 * denoised2)
        yield from self.result(x, sigma_up, sigma_down=sigma_down)


class TrapezoidalStep(ReversibleHeunStep):
    reversible = False
    trapezoidal_mode = True


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class TrapezoidalStep_(SingleStepSampler):
    name = "trapezoidal"
    model_calls = 1
    allow_alt_cfgpp = True

    def step(self, x):
        ss = self.ss
        sigma_next = ss.sigma_next
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())

        # Predict the sample at the next sigma using Euler step
        euler_sr = next(self.euler_step(x, sigma_down=sigma_next, sigma_up=0))
        d = euler_sr.noise_pred

        # Denoised sample at the next sigma
        mr_next = self.call_model(euler_sr.x, euler_sr.sigma_down, call_index=1)

        denoised_pred_next, d_next = self.get_split_prediction(mr=mr_next)
        yield from self.split_result(
            denoised_pred_next,
            (d + d_next) * 0.5,
            sigma_up=sigma_up,
            sigma_down=sigma_down,
        )

    def step_(self, x):
        ss = self.ss
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())

        # Calculate the derivative using the model
        d_i = self.to_d(ss.hcur)

        # Predict the sample at the next sigma using Euler step
        x_pred = x + d_i * ss.dt

        # Denoised sample at the next sigma
        mr_next = self.call_model(x_pred, ss.sigma_next, call_index=1)

        # Calculate the derivative at the next sigma
        d_next = self.to_d(mr_next)
        dt_2 = sigma_down - ss.sigma

        # Update the sample using the Trapezoidal rule
        x = x + dt_2 * (d_i + d_next) / 2
        yield from self.result(x, sigma_up, sigma_down=sigma_down)


class TrapezoidalCycleStep(CycleSingleStepSampler):
    name = "trapezoidal_cycle"
    model_calls = 1
    allow_alt_cfgpp = True

    def step(self, x):
        ss = self.ss
        # Calculate the derivative using the model
        d_i = self.to_d(ss.hcur)

        # Predict the sample at the next sigma using Euler step
        x_pred = x + d_i * ss.dt

        # Denoised sample at the next sigma
        mr_next = self.call_model(x_pred, ss.sigma_next, call_index=1)

        # Calculate the derivative at the next sigma
        d_next = self.to_d(mr_next)

        # Update the sample using the Trapezoidal rule
        keep_scale, add_scale = self.get_cycle_scales(ss.sigma_next)
        noise_pred = (d_i + d_next) * 0.5  # Combined noise prediction
        denoised_pred = x - noise_pred * ss.sigma  # Denoised prediction
        yield from self.result(denoised_pred + noise_pred * keep_scale, add_scale)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class BogackiStep(ReversibleSingleStepSampler):
    name = "bogacki"
    reversible = False
    model_calls = 2
    allow_alt_cfgpp = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.reversible:
            self.reversible_scale = 0

    def step(self, x):
        ss = self.ss
        s = ss.sigma
        sd, su = self.get_ancestral_step(self.get_dyn_eta())
        reta, reversible_scale = self.get_reversible_cfg()
        sdr, _sur = self.get_ancestral_step(reta)
        dt, dtr = sd - s, sdr - s

        # Calculate the derivative using the model
        d = self.to_d(ss.hcur)

        # Bogacki-Shampine steps
        k1 = d * dt
        k2 = self.to_d(self.call_model(x + k1 / 2, s + dt / 2, call_index=1)) * dt
        k3 = (
            self.to_d(
                self.call_model(x + 3 * k1 / 4 + k2 / 4, s + 3 * dt / 4, call_index=2)
            )
            * dt
        )

        # Reversible correction term (inspired by Reversible Heun)
        correction = dtr**2 * (k3 - k2) / 6

        # Update the sample
        x = (x + 2 * k1 / 9 + k2 / 3 + 4 * k3 / 9) - correction * reversible_scale
        yield from self.result(x, su, sigma_down=sd)


class ReversibleBogackiStep(BogackiStep):
    name = "reversible_bogacki"
    reversible = True


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class RK4Step(SingleStepSampler):
    name = "rk4"
    model_calls = 3
    allow_alt_cfgpp = True

    def step(self, x):
        ss = self.ss
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        sigma = ss.sigma
        d = self.to_d(ss.hcur)
        dt = sigma_down - sigma

        # Runge-Kutta steps
        k1 = d * dt
        k2 = self.to_d(self.call_model(x + k1 / 2, sigma + dt / 2, call_index=1)) * dt
        k3 = self.to_d(self.call_model(x + k2 / 2, sigma + dt / 2, call_index=2)) * dt
        k4 = self.to_d(self.call_model(x + k3, sigma + dt, call_index=3)) * dt

        # Update the sample
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        yield from self.result(x, sigma_up, sigma_down=sigma_down)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class RKF45Step(SingleStepSampler):
    name = "rkf45"
    model_calls = 5
    allow_alt_cfgpp = True

    def step(self, x):
        ss = self.ss
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        sigma = ss.sigma
        d = self.to_d(ss.hcur)
        dt = sigma_down - sigma

        # Runge-Kutta steps
        sigma_progression = (
            sigma + dt / 4,
            sigma + 3 * dt / 8,
            sigma + 12 * dt / 13,
            sigma + dt,
        )

        call_progression = (
            lambda k1: x + k1 / 4,
            lambda k1, k2: x + 3 * k1 / 32 + 9 * k2 / 32,
            lambda k1, k2, k3: x
            + 1932 * k1 / 2197
            - 7200 * k2 / 2197
            + 7296 * k3 / 2197,
            lambda k1, k2, k3, k4: x
            + 439 * k1 / 216
            - 8 * k2
            + 3680 * k3 / 513
            - 845 * k4 / 4104,
        )

        k = [d * dt]
        for idx, (ksigma, kfun) in enumerate(zip(sigma_progression, call_progression)):
            curr_x = kfun(*k)
            k.append(self.to_d(self.call_model(curr_x, ksigma)) * dt)
        del curr_x
        x = x + 25 * k[0] / 216 + 1408 * k[2] / 2565 + 2197 * k[3] / 4104 - k[4] / 5
        yield from self.result(x, sigma_up, sigma_down=sigma_down)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class RKDynamicStep(SingleStepSampler):
    name = "rk_dynamic"
    model_calls = 3
    allow_alt_cfgpp = True

    rk_weights = (
        (1,),
        (0.5, 0.5),
        (1 / 6, 2 / 3, 1 / 6),
        (1 / 8, 3 / 8, 3 / 8, 1 / 8),
    )

    rk_error_orders = ((0.0375, 4), (0.075, 3), (0.15, 2))

    def __init__(self, *args, max_order=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_order = max(0, min(max_order, 4))

    def get_rk_error_order(self, error):
        for threshold, order in self.rk_error_orders:
            if error < threshold:
                return order
        return 1

    def step(self, x):
        ss = self.ss
        order = self.max_order

        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        sigma = ss.sigma
        d = self.to_d(ss.hcur)
        dt = sigma_down - sigma

        error = ss.hcur.get_error(ss.hprev) if len(ss.hist) > 1 else 0.0
        if order < 1:
            order = self.get_rk_error_order(error)

        k = [d * dt]
        curr_weight = self.rk_weights[order - 1]

        # print(
        #     f"\nRK: weight={curr_weight!r}, histlen={len(ss.hist)}, order={order} ({self.max_order}), err={error:.6}\n"
        # )
        for j in range(1, order):
            # Calculate intermediate k values based on the current order
            k_sum = sum(curr_weight[i] * k[i] for i in range(j))
            mr = self.call_model(x + k_sum, sigma + dt * sum(curr_weight[:j]))
            k.append(self.to_d(mr) * dt)
            del mr

        # Update the sample using the weighted sum of k values
        x = x + sum(curr_weight[j] * k[j] for j in range(order))

        yield from self.result(x, sigma_up, sigma_down=sigma_down)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class EulerDancingStep(SingleStepSampler):
    name = "euler_dancing"
    self_noise = 1

    def __init__(
        self,
        *,
        deta=1.0,
        ds_noise=None,
        leap=2,
        dyn_deta_start=None,
        dyn_deta_end=None,
        dyn_deta_mode="lerp",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.deta = deta
        self.ds_noise = ds_noise if ds_noise is not None else self.s_noise
        self.leap = leap
        self.dyn_deta_start = dyn_deta_start
        self.dyn_deta_end = dyn_deta_end
        if dyn_deta_mode not in ("lerp", "lerp_alt", "deta"):
            raise ValueError("Bad dyn_deta_mode")
        self.dyn_deta_mode = dyn_deta_mode

    def step(self, x):
        ss = self.ss
        eta = self.eta
        deta = self.deta
        leap_sigmas = ss.sigmas[ss.idx :]
        leap_sigmas = leap_sigmas[: utils.find_first_unsorted(leap_sigmas)]
        zero_idx = (leap_sigmas <= 0).nonzero().flatten()[:1]
        max_leap = (zero_idx.item() if len(zero_idx) else len(leap_sigmas)) - 1
        is_danceable = max_leap > 1 and ss.sigma_next != 0
        curr_leap = max(1, min(self.leap, max_leap))
        sigma_leap = leap_sigmas[curr_leap] if is_danceable else ss.sigma_next
        del leap_sigmas
        sigma_down, sigma_up = get_ancestral_step(ss.sigma, sigma_leap, eta)
        print("???", sigma_down, sigma_up)
        d = to_d(x, ss.sigma, ss.denoised)
        # Euler method
        dt = sigma_down - ss.sigma
        x = x + d * dt
        if curr_leap == 1:
            return (yield from self.result(x, sigma_up))
        noise_strength = self.ds_noise * sigma_up
        if noise_strength != 0:
            x = yield from self.result(x, sigma_up, sigma_next=sigma_leap, final=False)

            # x = x + self.noise_sampler(ss.sigma, sigma_leap).mul_(
            #     self.ds_noise * sigma_up
            # )
        # sigma_down2, sigma_up2 = get_ancestral_step(sigma_leap, ss.sigma, eta=deta)
        # _sigma_down2, sigma_up2 = get_ancestral_step(sigma_leap, ss.sigma, eta=deta)
        # sigma_up2 = ss.sigma_next + (ss.sigma - ss.sigma_next) * 0.5
        sigma_up2 = get_ancestral_step(ss.sigma_next, sigma_leap, eta=deta)[1] + (
            ss.sigma_next * 0.5
        )
        sigma_down2, _sigma_up2 = get_ancestral_step(
            ss.sigma_next, sigma_leap, eta=deta
        )
        print(">>>", sigma_down2, sigma_up2, "--", ss.sigma, "->", sigma_leap)
        # sigma_down2, sigma_up2 = get_ancestral_step(ss.sigma_next, sigma_leap, eta=deta)
        d_2 = to_d(x, sigma_leap, ss.denoised)
        dt_2 = sigma_down2 - sigma_leap
        x = x + d_2 * dt_2
        yield from self.result(x, sigma_up2, sigma_down=sigma_down2)

    # def _step(self, x, ss):
    #     eta = self.get_dyn_eta(ss)
    #     leap_sigmas = ss.sigmas[ss.idx :]
    #     leap_sigmas = leap_sigmas[: utils.find_first_unsorted(leap_sigmas)]
    #     zero_idx = (leap_sigmas <= 0).nonzero().flatten()[:1]
    #     max_leap = (zero_idx.item() if len(zero_idx) else len(leap_sigmas)) - 1
    #     is_danceable = max_leap > 1 and ss.sigma_next != 0
    #     curr_leap = max(1, min(self.leap, max_leap))
    #     sigma_leap = leap_sigmas[curr_leap] if is_danceable else ss.sigma_next
    #     # DANCE 35 6 tensor(10.0947, device='cuda:0') -- tensor([21.9220,
    #     # print("DANCE", max_leap, curr_leap, sigma_leap, "--", leap_sigmas)
    #     del leap_sigmas
    #     sigma_down, sigma_up = get_ancestral_step(ss.sigma, sigma_leap, eta)
    #     d = to_d(x, ss.sigma, ss.denoised)
    #     # Euler method
    #     dt = sigma_down - ss.sigma
    #     x = x + d * dt
    #     if curr_leap == 1:
    #         return x, sigma_up
    #     dance_scale = self.get_dyn_value(ss, self.dyn_deta_start, self.dyn_deta_end)
    #     if curr_leap == 1 or not is_danceable or abs(dance_scale) < 1e-04:
    #         print("NODANCE", dance_scale, self.deta, is_danceable, ss.sigma_next)
    #         yield SamplerResult(ss, self, x, sigma_up)
    #     print(
    #         "DANCE", dance_scale, self.deta, self.dyn_deta_mode, self.ds_noise, sigma_up
    #     )
    #     sigma_down_normal, sigma_up_normal = get_ancestral_step(
    #         ss.sigma, ss.sigma_next, eta
    #     )
    #     if self.dyn_deta_mode == "lerp":
    #         dt_normal = sigma_down_normal - ss.sigma
    #         x_normal = x + d * dt_normal
    #     else:
    #         x_normal = x
    #     sigma_down2, sigma_up2 = get_ancestral_step(
    #         sigma_leap,
    #         ss.sigma_next,
    #         eta=self.deta * (1.0 if self.dyn_deta_mode != "deta" else dance_scale),
    #     )
    #     print(
    #         "-->",
    #         sigma_down2,
    #         sigma_up2,
    #         "--",
    #         self.deta * (1.0 if self.dyn_deta_mode != "deta" else dance_scale),
    #     )
    #     x = x + self.noise_sampler(ss.sigma, sigma_leap).mul_(self.ds_noise * sigma_up)
    #     d_2 = to_d(x, sigma_leap, ss.denoised)
    #     dt_2 = sigma_down2 - sigma_leap
    #     result = x + d_2 * dt_2
    #     # SIGMA: norm_up=9.062416076660156, up=10.703859329223633, up2=19.376544952392578, str=21.955078125
    #     noise_strength = sigma_up2 + ((sigma_up - sigma_up_normal) ** 5.0)
    #     noise_strength = sigma_up2 + ((sigma_up2 - sigma_up) * 0.5)
    #     # noise_strength = sigma_up2 + (
    #     #     (sigma_up2 - sigma_up) ** (1.0 - (sigma_up_normal / sigma_up2))
    #     # )
    #     noise_diff = (
    #         sigma_up - sigma_up_normal
    #         if sigma_up > sigma_up_normal
    #         else sigma_up_normal - sigma_up
    #     )
    #     noise_div = (
    #         sigma_up / sigma_up_normal
    #         if sigma_up > sigma_up_normal
    #         else sigma_up_normal / sigma_up
    #     )
    #     noise_diff = sigma_up2 - sigma_up_normal
    #     noise_div = sigma_up2 / sigma_up_normal
    #     noise_div = ss.sigma / sigma_leap

    #     # noise_strength = sigma_up2 + (noise_diff * noise_div)
    #     # noise_strength = sigma_up2 + ((noise_diff * 0.5) ** 2.0)
    #     # noise_strength = sigma_up2 + ((1.0 - noise_diff) ** 0.5)
    #     # noise_strength = sigma_up2 + (((sigma_up2 - sigma_up) * 0.5) ** 2.0)
    #     # noise_strength = sigma_up2 + (((sigma_up2 - sigma_up_normal) * 0.5) ** 1.5)
    #     # noise_strength = sigma_up2 + (
    #     #     (noise_diff * 0.1875) ** (1.0 / (noise_div - 0.0))
    #     # )
    #     # noise_strength = sigma_up2 + (
    #     #     (noise_diff * 0.125) ** (1.0 / (noise_div * 1.25))
    #     # )
    #     # noise_strength = sigma_up2 + ((noise_diff * 0.2) ** (1.0 / (noise_div * 1.0)))
    #     noise_strength = sigma_up2 + (noise_diff * 0.9 * max(0.0, noise_div - 0.8))
    #     noise_strength = sigma_up2 + (
    #         (noise_diff / (curr_leap * 0.4))
    #         * ((noise_div - (curr_leap / 2.0)).clamp(min=0, max=1.5) * 1.0)
    #     )
    #     # (1.0 / (noise_div * 1.25)))
    #     # noise_strength = sigma_up2 + ((noise_diff * 0.5) ** noise_div)
    #     print(
    #         f"SIGMA: norm_up={sigma_up_normal}, up={sigma_up}, up2={sigma_up2}, str={noise_strength}",
    #         # noise_diff,
    #         noise_div,
    #     )
    #     return result, noise_strength

    #     noise_diff = sigma_up2 - sigma_up * dance_scale
    #     noise_scale = sigma_up2 + noise_diff * (0.025 * curr_leap)
    #     # noise_scale = sigma_up2 * self.ds_noise
    #     if self.dyn_deta_mode == "deta" or dance_scale == 1.0:
    #         return result, noise_scale
    #     result = torch.lerp(x_normal, result, dance_scale)
    #     # FIXME: Broken for noise samplers that care about s/sn
    #     return result, noise_scale

    # def step(self, x, ss):
    #     eta = self.get_dyn_eta(ss)
    #     leap_sigmas = ss.sigmas[ss.idx :]
    #     leap_sigmas = leap_sigmas[: find_first_unsorted(leap_sigmas)]
    #     zero_idx = (leap_sigmas <= 0).nonzero().flatten()[:1]
    #     max_leap = (zero_idx.item() if len(zero_idx) else len(leap_sigmas)) - 1
    #     is_danceable = max_leap > 1 and ss.sigma_next != 0
    #     curr_leap = max(1, min(self.leap, max_leap))
    #     sigma_leap = leap_sigmas[curr_leap] if is_danceable else ss.sigma_next
    #     # print("DANCE", max_leap, curr_leap, sigma_leap, "--", leap_sigmas)
    #     del leap_sigmas
    #     sigma_down, sigma_up = get_ancestral_step(ss.sigma, sigma_leap, eta)
    #     d = to_d(x, ss.sigma, ss.denoised)
    #     # Euler method
    #     dt = sigma_down - ss.sigma
    #     x = x + d * dt
    #     if curr_leap == 1:
    #         return x, sigma_up
    #     dance_scale = self.get_dyn_value(ss, self.dyn_deta_start, self.dyn_deta_end)
    #     if not is_danceable or abs(dance_scale) < 1e-04:
    #         print("NODANCE", dance_scale, self.deta)
    #         return x, sigma_up
    #     print("NODANCE", dance_scale, self.deta)
    #     sigma_down_normal, _sigma_up_normal = get_ancestral_step(
    #         ss.sigma, ss.sigma_next, eta
    #     )
    #     if self.dyn_deta_mode == "lerp":
    #         dt_normal = sigma_down_normal - ss.sigma
    #         x_normal = x + d * dt_normal
    #     else:
    #         x_normal = x
    #     x = x + self.noise_sampler(ss.sigma, sigma_leap).mul_(self.s_noise * sigma_up)
    #     sigma_down2, sigma_up2 = get_ancestral_step(
    #         sigma_leap,
    #         ss.sigma_next,
    #         eta=self.deta * (1.0 if self.dyn_deta_mode != "deta" else dance_scale),
    #     )
    #     d_2 = to_d(x, sigma_leap, ss.denoised)
    #     dt_2 = sigma_down2 - sigma_leap
    #     result = x + d_2 * dt_2
    #     noise_diff = sigma_up2 - sigma_up * dance_scale
    #     noise_scale = sigma_up2 + noise_diff * (0.025 * curr_leap)
    #     if self.dyn_deta_mode == "deta" or dance_scale == 1.0:
    #         return result, noise_scale
    #     result = torch.lerp(x_normal, result, dance_scale)
    #     # FIXME: Broken for noise samplers that care about s/sn
    #     return result, noise_scale


# Alt CFG++ approach referenced from https://github.com/comfyanonymous/ComfyUI/pull/3871 - thanks!
class DPMPP2SStep(SingleStepSampler, DPMPPStepMixin):
    name = "dpmpp_2s"
    model_calls = 1
    allow_alt_cfgpp = True

    def step(self, x):
        ss = self.ss
        t_fn, sigma_fn = self.t_fn, self.sigma_fn
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        # DPM-Solver++(2S)
        t, t_next = t_fn(ss.sigma), t_fn(sigma_down)
        r = 1 / 2
        h = t_next - t
        s = t + r * h
        eff_x = (
            x
            if self.alt_cfgpp_scale == 0 or ss.hcur.denoised_uncond is None
            else x + (ss.denoised - ss.hcur.denoised_uncond) * self.alt_cfgpp_scale
        )
        x_2 = (sigma_fn(s) / sigma_fn(t)) * eff_x - (-h * r).expm1() * ss.denoised
        denoised_2 = self.call_model(x_2, sigma_fn(s), call_index=1).denoised
        x = (sigma_fn(t_next) / sigma_fn(t)) * eff_x - (-h).expm1() * denoised_2
        yield from self.result(x, sigma_up, sigma_down=sigma_down)


class DPMPPSDEStep(SingleStepSampler, DPMPPStepMixin):
    name = "dpmpp_sde"
    self_noise = 1
    model_calls = 1
    allow_alt_cfgpp = True  # Implementation may not be correct.

    def __init__(self, *args, r=1 / 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r

    def step(self, x):
        ss = self.ss
        t_fn, sigma_fn = self.t_fn, self.sigma_fn
        r, eta = self.r, self.get_dyn_eta()
        # DPM-Solver++
        t, t_next = t_fn(ss.sigma), t_fn(ss.sigma_next)
        h = t_next - t
        s = t + h * r
        fac = 1 / (2 * r)

        # Step 1
        sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
        s_ = t_fn(sd)
        eff_x = (
            x
            if self.alt_cfgpp_scale == 0 or ss.hcur.denoised_uncond is None
            else x + (ss.denoised - ss.hcur.denoised_uncond) * self.alt_cfgpp_scale
        )
        x_2 = (sigma_fn(s_) / sigma_fn(t)) * eff_x - (t - s_).expm1() * ss.denoised
        x_2 = yield from self.result(
            x_2, su, sigma=sigma_fn(t), sigma_next=sigma_fn(s), final=False
        )
        denoised_2 = self.call_model(x_2, sigma_fn(s), call_index=1).denoised

        # Step 2
        sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
        t_next_ = t_fn(sd)
        denoised_d = (1 - fac) * ss.denoised + fac * denoised_2
        x = (sigma_fn(t_next_) / sigma_fn(t)) * eff_x - (
            t - t_next_
        ).expm1() * denoised_d
        yield from self.result(x, su, sigma_down=sd)


# Based on implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
# Which was originally written by Katherine Crowson
class TTMJVPStep(SingleStepSampler):
    name = "ttm_jvp"
    model_calls = 1

    def __init__(self, *args, alternate_phi_2_calc=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.alternate_phi_2_calc = alternate_phi_2_calc

    def step(self, x):
        ss = self.ss
        eta = self.get_dyn_eta()
        sigma, sigma_next = ss.sigma, ss.sigma_next
        # 2nd order truncated Taylor method
        t, s = -sigma.log(), -sigma_next.log()
        h = s - t
        h_eta = h * (eta + 1)

        eps = to_d(x, sigma, ss.denoised)
        denoised_prime = self.call_model(
            x, sigma, tangents=(eps * -sigma, -sigma), call_index=1
        ).jdenoised

        phi_1 = -torch.expm1(-h_eta)
        if self.alternate_phi_2_calc:
            phi_2 = torch.expm1(-h) + h  # seems to work better with eta > 0
        else:
            phi_2 = torch.expm1(-h_eta) + h_eta
        x = torch.exp(-h_eta) * x + phi_1 * ss.denoised + phi_2 * denoised_prime

        noise_scale = (
            sigma_next * torch.sqrt(-torch.expm1(-2 * h * eta))
            if eta
            else ss.sigma.new_zeros(1)
        )
        yield from self.result(x, noise_scale)


# Adapted from https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
# under Apache 2 license
class IPNDMStep(HistorySingleStepSampler):
    name = "ipndm"
    ancestralize = True
    default_history_limit, max_history = 1, 3
    allow_alt_cfgpp = True
    default_eta = 0.0

    IPNDM_MULTIPLIERS = (
        ((1,), 1),
        ((3, -1), 2),
        ((23, -16, 5), 12),
        ((55, -59, 37, -9), 24),
    )

    def step(self, x):
        ss = self.ss
        order = self.available_history() + 1
        if order > 1:
            hd = tuple(self.to_d(ss.hist[-hidx]) for hidx in range(order, 1, -1))
        (dm, *hms), divisor = self.IPNDM_MULTIPLIERS[order - 1]
        noise = dm * self.to_d(ss.hcur)
        for hidx, hm in enumerate(hms, start=1):
            noise += hm * hd[-hidx]
        noise /= divisor
        yield from self.result(x + ss.dt * noise)


# Adapted from https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
# under Apache 2 license
class IPNDMVStep(HistorySingleStepSampler):
    name = "ipndm_v"
    ancestralize = True
    default_history_limit, max_history = 1, 3
    allow_alt_cfgpp = True
    default_eta = 0.0

    def step(self, x):
        ss = self.ss
        dt = ss.dt
        d = self.to_d(ss.hcur)
        order = self.available_history() + 1
        if order > 1:
            hd = tuple(self.to_d(ss.hist[-hidx]) for hidx in range(order, 1, -1))
            hns = (
                ss.sigmas[ss.idx - (order - 2) : ss.idx + 1]
                - ss.sigmas[ss.idx - (order - 1) : ss.idx]
            )
        if order == 1:
            noise = d
        elif order == 2:
            coeff1 = (2 + (dt / hns[-1])) / 2
            coeff2 = -(dt / hns[-1]) / 2
            noise = coeff1 * d + coeff2 * hd[-1]
        elif order == 3:
            temp = (
                1
                - dt
                / (3 * (dt + hns[-1]))
                * (dt * (dt + hns[-1]))
                / (hns[-1] * (hns[-1] + hns[-2]))
            ) / 2
            coeff1 = (2 + (dt / hns[-1])) / 2 + temp
            coeff2 = -(dt / hns[-1]) / 2 - (1 + hns[-1] / hns[-2]) * temp
            coeff3 = temp * hns[-1] / hns[-2]
            noise = coeff1 * d + coeff2 * hd[-1] + coeff3 * hd[-2]
        else:
            temp1 = (
                1
                - dt
                / (3 * (dt + hns[-1]))
                * (dt * (dt + hns[-1]))
                / (hns[-1] * (hns[-1] + hns[-2]))
            ) / 2
            temp2 = (
                (
                    (1 - dt / (3 * (dt + hns[-1]))) / 2
                    + (1 - dt / (2 * (dt + hns[-1])))
                    * dt
                    / (6 * (dt + hns[-1] + hns[-2]))
                )
                * (dt * (dt + hns[-1]) * (dt + hns[-1] + hns[-2]))
                / (hns[-1] * (hns[-1] + hns[-2]) * (hns[-1] + hns[-2] + hns[-3]))
            )
            coeff1 = (2 + (dt / hns[-1])) / 2 + temp1 + temp2
            coeff2 = (
                -(dt / hns[-1]) / 2
                - (1 + hns[-1] / hns[-2]) * temp1
                - (
                    1
                    + (hns[-1] / hns[-2])
                    + (hns[-1] * (hns[-1] + hns[-2]) / (hns[-2] * (hns[-2] + hns[-3])))
                )
                * temp2
            )
            coeff3 = (
                temp1 * hns[-1] / hns[-2]
                + (
                    (hns[-1] / hns[-2])
                    + (hns[-1] * (hns[-1] + hns[-2]) / (hns[-2] * (hns[-2] + hns[-3])))
                    * (1 + hns[-2] / hns[-3])
                )
                * temp2
            )
            coeff4 = (
                -temp2
                * (hns[-1] * (hns[-1] + hns[-2]) / (hns[-2] * (hns[-2] + hns[-3])))
                * hns[-1]
                / hns[-2]
            )
            noise = coeff1 * d + coeff2 * hd[-1] + coeff3 * hd[-2] + coeff4 * hd[-3]
        yield from self.result(x + ss.dt * noise)


class DEISStep(HistorySingleStepSampler):
    name = "deis"
    ancestralize = True
    default_history_limit, max_history = 1, 3
    allow_alt_cfgpp = True
    default_eta = 0.0

    def __init__(self, *args, deis_mode="tab", **kwargs):
        super().__init__(*args, **kwargs)
        self.deis_mode = deis_mode
        self.deis_coeffs_key = None
        self.deis_coeffs = None

    def get_deis_coeffs(self):
        ss = self.ss
        key = (
            self.history_limit,
            len(ss.sigmas),
            ss.sigmas[0].item(),
            ss.sigmas[-1].item(),
        )
        if self.deis_coeffs_key == key:
            return self.deis_coeffs
        self.deis_coeffs_key = key
        self.deis_coeffs = comfy.k_diffusion.deis.get_deis_coeff_list(
            ss.sigmas, self.history_limit + 1, deis_mode=self.deis_mode
        )
        return self.deis_coeffs

    def step(self, x):
        ss = self.ss
        dt = ss.dt
        d = self.to_d(ss.hcur)
        order = self.available_history() + 1
        if order < 2:
            noise = dt * d  # Euler
        else:
            c = self.get_deis_coeffs()[ss.idx]
            hd = tuple(self.to_d(ss.hist[-hidx]) for hidx in range(order, 1, -1))
            noise = c[0] * d
            for i in range(1, order):
                noise += c[i] * hd[-i]
        yield from self.result(x + noise)


class HeunPP2Step(SingleStepSampler):
    name = "heunpp2"
    ancestralize = True
    model_calls = 2
    allow_alt_cfgpp = True

    def __init__(self, *args, max_order=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_order = max(1, min(self.model_calls + 1, max_order))

    def step(self, x):
        ss = self.ss
        steps_remain = max(0, len(ss.sigmas) - (ss.idx + 2))
        order = min(self.max_order, steps_remain + 1)
        sn = ss.sigma_next
        if order == 1:
            return (yield from self.euler_step(x))
        d = self.to_d(ss.hcur)
        dt = ss.dt
        w = order * ss.sigma
        w2 = sn / w
        x_2 = x + d * dt
        d_2 = self.to_d(self.call_model(x_2, sn, call_index=1))
        if order == 2:
            # Heun's method (ish)
            w1 = 1 - w2
            d_prime = d * w1 + d_2 * w2
        else:
            # Heun++ (ish)
            snn = ss.sigmas[ss.idx + 2]
            dt_2 = snn - sn
            x_3 = x_2 + d_2 * dt_2
            d_3 = self.to_d(self.call_model(x_3, snn, call_index=2))
            w3 = snn / w
            w1 = 1 - w2 - w3
            d_prime = w1 * d + w2 * d_2 + w3 * d_3
        yield from self.result(x + d_prime * dt)


class DESolverStep(SingleStepSampler, MinSigmaStepMixin):
    de_default_solver = None
    sample_sigma_zero = True
    default_eta = 0.0

    def __init__(
        self,
        *args,
        de_solver=None,
        de_max_nfe=100,
        de_rtol=-2.5,
        de_atol=-3.5,
        de_fixup_hack=0.025,
        de_split=1,
        de_min_sigma=0.0292,
        **kwargs,
    ):
        self.check_solver_support()
        if not HAVE_TDE:
            raise RuntimeError(
                "TDE sampler requires torchdiffeq installed in venv. Example: pip install torchdiffeq"
            )
        super().__init__(*args, **kwargs)
        de_solver = self.de_default_solver if de_solver is None else de_solver
        self.de_solver_name = de_solver
        self.de_max_nfe = de_max_nfe
        self.de_rtol = 10**de_rtol
        self.de_atol = 10**de_atol
        self.de_fixup_hack = de_fixup_hack
        self.de_split = de_split
        self.de_min_sigma = de_min_sigma if de_min_sigma is not None else 0.0

    def check_solver_support(self):
        raise NotImplementedError

    def de_get_step(self, x):
        eta = self.get_dyn_eta()
        ss = self.ss
        s, sn = ss.sigma, ss.sigma_next
        sn = self.adjust_step(sn, self.de_min_sigma)
        sigma_down, sigma_up = self.get_ancestral_step(eta, sigma_next=sn)
        if self.de_fixup_hack != 0:
            sigma_down = (sigma_down - (s - sigma_down) * self.de_fixup_hack).clamp(
                min=0
            )
        return s, sn, sigma_down, sigma_up

    @staticmethod
    def reverse_time(t, t0, t1):
        return t1 + (t0 - t)


class TDEStep(DESolverStep):
    name = "tde"
    model_calls = 2
    allow_alt_cfgpp = True
    allow_cfgpp = False
    de_default_solver = "rk4"
    default_eta = 0.0

    def __init__(
        self,
        *args,
        de_split=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.de_split = de_split

    def check_solver_support(self):
        if not HAVE_TDE:
            raise RuntimeError(
                "TDE sampler requires torchdiffeq installed in venv. Example: pip install torchdiffeq"
            )

    def step(self, x):
        s, sn, sigma_down, sigma_up = self.de_get_step(x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x))
        ss = self.ss
        delta = (s - sigma_down).item()
        mcc = 0
        bidx = 0
        pbar = None

        def odefn(t, y):
            nonlocal mcc
            if t < 1e-05:
                return torch.zeros_like(y)
            if mcc >= self.de_max_nfe:
                raise RuntimeError("TDEStep: Model call limit exceeded")

            pct = (s - t) / delta
            pbar.n = round(min(999, pct.item() * 999))
            pbar.update(0)
            pbar.set_description(
                f"{self.de_solver_name}({mcc}/{self.de_max_nfe})", refresh=True
            )

            if t == ss.sigma and torch.equal(x[bidx], y):
                mr_cached = True
                mr = ss.hcur
                mcc = 1
            else:
                mr_cached = False
                mr = self.call_model(
                    y.unsqueeze(0), t, call_index=mcc, s_in=t.new_ones(1)
                )
                mcc += 1
            return self.to_d(mr)[bidx if mr_cached else 0]

        result = torch.zeros_like(x)
        t = sigma_down.new_zeros(self.de_split + 1)
        torch.linspace(ss.sigma, sigma_down, t.shape[0], out=t)

        for batch in tqdm.trange(
            1,
            x.shape[0] + 1,
            desc="batch",
            leave=False,
            disable=x.shape[0] == 1 or ss.disable_status,
        ):
            bidx = batch - 1
            mcc = 0
            if pbar is not None:
                pbar.close()
            pbar = tqdm.tqdm(
                total=1000,
                desc=self.de_solver_name,
                leave=True,
                disable=ss.disable_status,
            )
            solution = tde.odeint(
                odefn,
                x[bidx],
                t,
                rtol=self.de_rtol,
                atol=self.de_atol,
                method=self.de_solver_name,
                options={
                    "min_step": 1e-05,
                    "dtype": torch.float64,
                },
            )[-1]
            result[bidx] = solution

        sigma_up, result = yield from self.adjusted_step(sn, result, mcc, sigma_up)
        if pbar is not None:
            pbar.n = pbar.total
            pbar.update(0)
            pbar.close()
        yield from self.result(result, sigma_up, sigma_down=sigma_down)


class TODEStep(DESolverStep):
    name = "tode"
    model_calls = 2
    allow_alt_cfgpp = True
    de_default_solver = "dopri5"
    default_eta = 0.0

    def __init__(
        self,
        *args,
        de_initial_step=0.25,
        tode_compile=False,
        de_ctl_pcoeff=0.3,
        de_ctl_icoeff=0.9,
        de_ctl_dcoeff=0.2,
        **kwargs,
    ):
        if not HAVE_TODE:
            raise RuntimeError(
                "TODE sampler requires torchode installed in venv. Example: pip install torchode"
            )
        super().__init__(*args, **kwargs)
        self.de_solver_method = tode.interface.METHODS[self.de_solver_name]
        self.de_ctl_pcoeff = de_ctl_pcoeff
        self.de_ctl_icoeff = de_ctl_icoeff
        self.de_ctl_dcoeff = de_ctl_dcoeff
        self.de_compile = tode_compile
        self.de_initial_step = de_initial_step

    def check_solver_support(self):
        if not HAVE_TODE:
            raise RuntimeError(
                "TODE sampler requires torchode installed in venv. Example: pip install torchode"
            )

    def step(self, x):
        s, sn, sigma_down, sigma_up = self.de_get_step(x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x))
        ss = self.ss
        delta = (ss.sigma - sigma_down).item()
        mcc = 0
        pbar = None
        b, c, h, w = x.shape

        def odefn(t, y_flat):
            nonlocal mcc
            if torch.all(t <= 1e-05).item():
                return torch.zeros_like(y_flat)
            if mcc >= self.de_max_nfe:
                raise RuntimeError("TDEStep: Model call limit exceeded")

            pct = (s - t) / delta
            pbar.n = round(pct.min().item() * 999)
            pbar.update(0)
            pbar.set_description(
                f"{self.de_solver_name}({mcc}/{self.de_max_nfe})", refresh=True
            )
            y = y_flat.reshape(-1, c, h, w)
            t32 = t.to(torch.float32)
            del y_flat

            if mcc == 0 and torch.all(t == s):
                mr = ss.hcur
                mcc = 1
            else:
                mr = self.call_model(y, t32.clamp(min=1e-05), call_index=mcc)
                mcc += 1
            result = self.to_d(mr).flatten(start_dim=1)
            for bi in range(t.shape[0]):
                if t[bi] <= 1e-05:
                    result[bi, :] = 0
            return result

        t = torch.stack((s, sigma_down)).to(torch.float64).repeat(b, 1)

        pbar = tqdm.tqdm(
            total=1000, desc=self.de_solver_name, leave=True, disable=ss.disable_status
        )

        term = tode.ODETerm(odefn)
        method = self.de_solver_method(term=term)
        controller = tode.PIDController(
            term=term,
            atol=self.de_atol,
            rtol=self.de_rtol,
            dt_min=1e-05,
            pcoeff=self.de_ctl_pcoeff,
            icoeff=self.de_ctl_icoeff,
            dcoeff=self.de_ctl_dcoeff,
        )
        solver_ = tode.AutoDiffAdjoint(method, controller)
        solver = solver_ if not self.de_compile else torch.compile(solver_)
        problem = tode.InitialValueProblem(
            y0=x.flatten(start_dim=1), t_start=t[:, 0], t_end=t[:, -1]
        )
        dt0 = (
            (t[:, -1] - t[:, 0]) * self.de_initial_step
            if self.de_initial_step
            else None
        )
        solution = solver.solve(problem, dt0=dt0)

        # print("\nSOLUTION", solution.stats, solution.ys.shape)
        result = solution.ys[:, -1].reshape(-1, c, h, w)
        del solution

        sigma_up, result = yield from self.adjusted_step(sn, result, mcc, sigma_up)
        if pbar is not None:
            pbar.n = pbar.total
            pbar.update(0)
            pbar.close()
        yield from self.result(result, sigma_up, sigma_down=sigma_down)


class TSDEStep(DESolverStep):
    name = "tsde"
    model_calls = 2
    allow_alt_cfgpp = True
    de_default_solver = "reversible_heun"
    default_eta = 0.0

    def __init__(
        self,
        *args,
        de_initial_step=0.25,
        de_split=1,
        de_adaptive=False,
        tsde_noise_type="scalar",
        tsde_sde_type="stratonovich",
        tsde_levy_area_approx="none",
        tsde_noise_channels=1,
        tsde_g_multiplier=0.05,
        tsde_g_reverse_time=True,
        tsde_g_derp_mode=False,
        tsde_batch_channels=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.de_initial_step = de_initial_step
        self.de_adaptive = de_adaptive
        self.de_split = de_split
        self.de_noise_type = tsde_noise_type
        self.de_sde_type = tsde_sde_type
        self.de_levy_area_approx = tsde_levy_area_approx
        self.de_g_multiplier = tsde_g_multiplier
        self.de_noise_channels = tsde_noise_channels
        self.de_g_reverse_time = tsde_g_reverse_time
        self.de_g_derp_mode = tsde_g_derp_mode
        self.de_batch_channels = tsde_batch_channels

    def check_solver_support(self):
        pass

    def step(self, x):
        s, sn, sigma_down, sigma_up = self.de_get_step(x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x))
        ss = self.ss
        delta = (ss.sigma - sigma_down).item()
        bidx = 0
        mcc = 0
        pbar = None
        _b, c, h, w = x.shape
        outer_self = self

        class SDE(torch.nn.Module):
            noise_type = outer_self.de_noise_type
            sde_type = outer_self.de_sde_type

            @torch.no_grad()
            def f(self, t_rev, y_flat):
                nonlocal mcc
                t = s - (t_rev - sigma_down)
                # print(f"\nf at t_rev={t_rev}, t={t} :: {y_flat.shape}")
                if torch.all(t <= 1e-05).item():
                    return torch.zeros_like(y_flat)
                if mcc >= outer_self.de_max_nfe:
                    raise RuntimeError("TSDEStep: Model call limit exceeded")

                pct = (s - t) / delta
                pbar.n = round(pct.min().item() * 999)
                pbar.update(0)
                pbar.set_description(
                    f"{outer_self.de_solver_name}({mcc}/{outer_self.de_max_nfe})",
                    refresh=True,
                )
                flat_shape = y_flat.shape
                y = y_flat.view(1, c, h, w)
                t32 = t.to(torch.float32)
                del y_flat

                if mcc == 0 and torch.all(t == s):
                    mr_cached = True
                    mr = ss.hcur
                    mcc = 1
                else:
                    mr_cached = False
                    mr = outer_self.call_model(
                        y, t32.clamp(min=1e-05), call_index=mcc, s_in=t.new_ones(1)
                    )
                    mcc += 1
                return -outer_self.to_d(mr)[bidx if mr_cached else 0].view(*flat_shape)

            @torch.no_grad()
            def g(self, t_rev, y_flat):
                t = (s - sigma_down) - (t_rev - sigma_down)
                pct = t / (s - sigma_down)
                if outer_self.de_g_reverse_time:
                    pct = 1.0 - pct
                multiplier = outer_self.de_g_multiplier
                if outer_self.de_g_derp_mode and mcc % 2 == 0:
                    multiplier *= -1
                val = t * pct * multiplier
                if self.noise_type == "diagonal":
                    out = val.repeat(*y_flat.shape)
                elif self.noise_type == "scalar":
                    out = val.repeat(*y_flat.shape, 1)
                else:
                    out = val.repeat(*y_flat.shape, outer_self.de_noise_channels)
                return out

        t = torch.stack((sigma_down, s)).to(torch.float)

        pbar = tqdm.tqdm(
            total=1000, desc=self.de_solver_name, leave=True, disable=ss.disable_status
        )

        dt0 = (
            delta * self.de_initial_step if self.de_adaptive else delta / self.de_split
        )
        results = []
        for batch in tqdm.trange(
            1,
            x.shape[0] + 1,
            desc="batch",
            leave=False,
            disable=x.shape[0] == 1 or ss.disable_status,
        ):
            bidx = batch - 1
            mcc = 0
            sde = SDE()
            if self.de_batch_channels:
                y_flat = x[bidx].flatten(start_dim=1)
            else:
                y_flat = x[bidx].unsqueeze(0).flatten(start_dim=1)
            if sde.noise_type == "diagonal":
                bm_size = (y_flat.shape[0], y_flat.shape[1])
            elif sde.noise_type == "scalar":
                bm_size = (y_flat.shape[0], 1)
            else:
                bm_size = (y_flat.shape[0], self.de_noise_channels)
            bm = torchsde.BrownianInterval(
                dtype=x.dtype,
                device=x.device,
                t0=-s,
                t1=s,
                entropy=ss.noise.seed,
                levy_area_approximation=self.de_levy_area_approx,
                tol=1e-06,
                size=bm_size,
            )

            ys = torchsde.sdeint(
                sde,
                y_flat,
                t,
                method=self.de_solver_name,
                adaptive=self.de_adaptive,
                atol=self.de_atol,
                rtol=self.de_rtol,
                dt=dt0,
                bm=bm,
            )
            del y_flat
            results.append(ys[-1].view(1, c, h, w))
            del ys
        result = torch.cat(results)
        del results

        sigma_up, result = yield from self.adjusted_step(sn, result, mcc, sigma_up)
        if pbar is not None:
            pbar.n = pbar.total
            pbar.update(0)
            pbar.close()
        yield from self.result(result, sigma_up, sigma_down=sigma_down)


if HAVE_DIFFRAX:

    class RevVirtualBrownianTree(diffrax.VirtualBrownianTree):
        def evaluate(self, t0, t1, *args, **kwargs):
            if t1 is not None:
                return super().evaluate(t1, t0, *args, **kwargs)
            return super().evaluate(t0, t1, *args, **kwargs)

    class StepCallbackTqdmProgressMeter(diffrax.TqdmProgressMeter):
        step_callback: typing.Callable = None

        def _init_bar(self, *args, **kwargs):
            if self.step_callback is None:
                return super()._init_bar(*args, **kwargs)
            bar_format = "{percentage:.2f}%{step_callback}|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            step_callback = self.step_callback

            class WrapTqdm(tqdm.tqdm):
                @property
                def format_dict(self):
                    d = super().format_dict
                    d.update(step_callback=step_callback())
                    return d

            return WrapTqdm(total=100, unit="%", bar_format=bar_format)


class DiffraxStep(DESolverStep):
    name = "diffrax"
    model_calls = 2
    allow_alt_cfgpp = True
    de_default_solver = "dopri5"
    default_eta = 0.0

    def __init__(
        self,
        *args,
        de_split=1,
        de_initial_step=0.25,
        de_ctl_pcoeff=0.3,
        de_ctl_icoeff=0.9,
        de_ctl_dcoeff=0.2,
        diffrax_adaptive=False,
        diffrax_fake_pure_callback=True,
        diffrax_g_multiplier=0.0,
        diffrax_half_solver=False,
        diffrax_batch_channels=False,
        diffrax_levy_area_approx="brownian_increment",
        diffrax_error_order=None,
        diffrax_sde_mode=False,
        diffrax_g_reverse_time=False,
        diffrax_g_time_scaling=False,
        diffrax_g_split_time_mode=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        solvers = dict(
            euler=diffrax.Euler,
            heun=diffrax.Heun,
            midpoint=diffrax.Midpoint,
            ralston=diffrax.Ralston,
            bosh3=diffrax.Bosh3,
            tsit5=diffrax.Tsit5,
            dopri5=diffrax.Dopri5,
            dopri8=diffrax.Dopri8,
            implicit_euler=diffrax.ImplicitEuler,
            # kvaerno3=diffrax.Kvaerno3,
            # kvaerno4=diffrax.Kvaerno4,
            # kvaerno5=diffrax.Kvaerno5,
            semi_implicit_euler=diffrax.SemiImplicitEuler,
            reversible_heun=diffrax.ReversibleHeun,
            leapfrog_midpoint=diffrax.LeapfrogMidpoint,
            euler_heun=diffrax.EulerHeun,
            ito_milstein=diffrax.ItoMilstein,
            stratonovich_milstein=diffrax.StratonovichMilstein,
            sea=diffrax.SEA,
            sra1=diffrax.SRA1,
            shark=diffrax.ShARK,
            general_shark=diffrax.GeneralShARK,
            slow_rk=diffrax.SlowRK,
            spark=diffrax.SPaRK,
        )
        levy_areas = dict(
            brownian_increment=diffrax.BrownianIncrement,
            space_time=diffrax.SpaceTimeLevyArea,
            space_time_time=diffrax.SpaceTimeTimeLevyArea,
        )
        # jax.config.update("jax_disable_jit", True)
        self.de_solver_method = solvers[self.de_solver_name]()
        if diffrax_half_solver:
            self.de_solver_method = diffrax.HalfSolver(self.de_solver_method)
        self.de_ctl_pcoeff = de_ctl_pcoeff
        self.de_ctl_icoeff = de_ctl_icoeff
        self.de_ctl_dcoeff = de_ctl_dcoeff
        self.de_initial_step = de_initial_step
        self.de_adaptive = diffrax_adaptive
        self.de_split = de_split
        self.de_fake_pure_callback = diffrax_fake_pure_callback
        self.de_g_multiplier = diffrax_g_multiplier
        self.de_batch_channels = diffrax_batch_channels
        self.de_levy_area_approx = levy_areas[diffrax_levy_area_approx]
        self.de_error_order = diffrax_error_order
        self.de_sde_mode = diffrax_sde_mode
        self.de_g_reverse_time = diffrax_g_reverse_time
        self.de_g_time_scaling = diffrax_g_time_scaling
        self.de_g_split_time_mode = diffrax_g_split_time_mode

    # As slow and safe as possible.
    @staticmethod
    def t2j(t):
        return jax.block_until_ready(
            jax.numpy.array(numpy.array(t.detach().cpu().contiguous()))
        )

    @staticmethod
    def j2t(t):
        return torch.from_numpy(numpy.array(jax.block_until_ready(t))).contiguous()

    def check_solver_support(self):
        if not HAVE_DIFFRAX:
            raise RuntimeError(
                "Diffrax sampler requires diffrax and jax installed in venv."
            )

    def step(self, x):
        s, sn, sigma_down, sigma_up = self.de_get_step(x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x))
        ss = self.ss
        bidx = 0
        mcc = 0
        _b, c, h, w = x.shape
        interrupted = None
        t0, t1 = sigma_down.item(), s.item()

        def odefn_(t_orig, y_flat, args=()):
            nonlocal mcc, interrupted
            t = self.reverse_time(self.j2t(t_orig).to(s), t0, t1)
            if t <= 1e-05:
                return jax.numpy.zeros_like(y_flat)
            if mcc >= self.de_max_nfe:
                raise RuntimeError("DiffraxStep: Model call limit exceeded")
            y = self.j2t(y_flat.reshape(1, c, h, w)).to(x)
            t32 = t.to(s).clamp(min=1e-05)
            flat_shape = y_flat.shape
            del y_flat

            if not args and mcc == 0 and torch.all(t == s):
                mr_cached = True
                mr = ss.hcur
                mcc = 1
            else:
                mr_cached = False
                try:
                    if not args:
                        mr = self.call_model(y, t32, call_index=mcc, s_in=t.new_ones(1))
                    else:
                        print("TANGENTS")
                        mr = self.call_model(
                            y,
                            t32,
                            call_index=mcc,
                            tangents=args,
                            s_in=t.new_ones(1),
                        )
                except comfy.model_management.InterruptProcessingException as exc:
                    interrupted = exc
                    raise
                mcc += 1
            result = self.to_d(mr)[bidx if mr_cached else 0].reshape(*flat_shape)
            return self.t2j(-result)

        if not self.de_fake_pure_callback:

            def odefn(t, y_flat, args):
                return jax.experimental.io_callback(
                    odefn_, y_flat, t, y_flat, ordered=True
                )

        else:

            def odefn(t, y_flat, args):
                return jax.pure_callback(odefn_, y_flat, t, y_flat)

        def g(t, y, _args):
            if self.de_g_split_time_mode:
                val = jax.lax.cond(
                    t < t0 + (t1 - t0) * 0.5,
                    lambda: self.de_g_multiplier,
                    lambda: -self.de_g_multiplier,
                )
            else:
                val = self.de_g_multiplier
            if self.de_g_time_scaling:
                val *= self.reverse_time(t, t0, t1) if self.de_g_reverse_time else t
            if not self.de_batch_channels:
                return val
            return jax.numpy.float32(val).broadcast((y.shape[0],))

        def progress_callback():
            return f" ({mcc:>3}/{self.de_max_nfe:>3}) {self.de_solver_name}"

        term = diffrax.ODETerm(odefn)
        method = self.de_solver_method
        if self.de_adaptive:
            controller = diffrax.PIDController(
                atol=self.de_atol,
                rtol=self.de_rtol,
                dtmin=1e-05,
                pcoeff=self.de_ctl_pcoeff,
                icoeff=self.de_ctl_icoeff,
                dcoeff=self.de_ctl_dcoeff,
                error_order=self.de_error_order,
            )
        else:
            controller = diffrax.ConstantStepSize()

        if not self.de_adaptive:
            dt0 = (t1 - t0) / self.de_split
        else:
            dt0 = (t1 - t0) * self.de_initial_step
        if self.de_sde_mode:
            bm = diffrax.VirtualBrownianTree(
                t0=ss.sigmas.min().item(),
                t1=ss.sigmas.max().item(),
                tol=1e-06,
                levy_area=self.de_levy_area_approx,
                shape=(c,) if self.de_batch_channels else (),
                key=jax.random.PRNGKey(ss.noise.seed + ss.noise.seed_offset),
            )
            term = diffrax.MultiTerm(term, diffrax.ControlTerm(g, bm))
        results = []
        for batch in tqdm.trange(
            1,
            x.shape[0] + 1,
            desc="batch",
            leave=False,
            disable=x.shape[0] == 1 or ss.disable_status,
        ):
            bidx = batch - 1
            mcc = 0
            if self.de_batch_channels:
                y_flat = x[bidx].flatten(start_dim=1)
            else:
                y_flat = x[bidx].unsqueeze(0).flatten(start_dim=1)
            y_flat = self.t2j(y_flat)
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                try:
                    solution = diffrax.diffeqsolve(
                        terms=term,
                        solver=method,
                        t0=t0,
                        t1=t1,
                        dt0=dt0,
                        y0=y_flat,
                        saveat=diffrax.SaveAt(t1=True),
                        stepsize_controller=controller,
                        progress_meter=StepCallbackTqdmProgressMeter(
                            step_callback=progress_callback,
                            refresh_steps=1,
                        ),
                    )
                except Exception:
                    if interrupted is not None:
                        raise interrupted
                    raise
            results.append(self.j2t(solution.ys).view(1, *x.shape[1:]))
            del solution
        result = torch.cat(results).to(x)
        sigma_up, result = yield from self.adjusted_step(sn, result, mcc, sigma_up)
        yield from self.result(result, sigma_up, sigma_down=sigma_down)


class HeunStep(ReversibleSingleStepSampler):
    name = "heun"
    model_calls = 1
    default_history_limit, max_history = 0, 0
    allow_alt_cfgpp = True

    def reversible_correction(self, d_from, d_to):
        reta, reversible_scale = self.get_reversible_cfg()
        if reversible_scale == 0:
            return 0
        sdr = self.get_ancestral_step(reta)[0]
        dtr = sdr - self.ss.sigma
        return (dtr**2 * (d_to - d_from) / 4) * self.reversible_scale

    def step(self, x):
        ss = self.ss
        s = ss.sigma
        sd, su = self.get_ancestral_step(self.get_dyn_eta())
        dt = sd - s
        hcur = ss.hcur
        d = self.to_d(hcur)
        x_next = hcur.denoised + d * sd
        d_next = self.to_d(self.call_model(x_next, sd, call_index=1))
        result = hcur.denoised + d * s
        result += (dt * (d + d_next)) * 0.5
        result -= self.reversible_correction(d, d_next)
        yield from self.result(result, su, sigma_down=sd)


class Heun1SStep(HeunStep):
    name = "heun_1s"
    model_calls = 1
    allow_alt_cfgpp = True
    default_history_limit, max_history = 1, 1

    def step(self, x):
        ss = self.ss
        s = ss.sigma
        if self.available_history() == 0:
            return (yield from super().step(x))
        hcur, hprev = ss.hcur, ss.hprev
        d_prev = self.to_d(hprev)
        sd, su = self.get_ancestral_step(self.get_dyn_eta())
        dt = sd - s
        d = self.to_d(hcur)
        result = hcur.denoised + hcur.sigma * self.to_d(hcur)
        result += (dt * (d_prev + d)) * 0.5
        result -= self.reversible_correction(d_prev, d)
        yield from self.result(result, su, sigma_down=sd)


class AdapterStep(SingleStepSampler):
    name = "adapter"
    model_calls = 2
    immiscible = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.external_sampler = self.options.pop(
            "SAMPLER", comfy.samplers.sampler_object("euler")
        )
        sig = inspect.signature(self.external_sampler.sampler_function)
        self.external_sampler_options = {
            k: v
            for k, v in self.options.pop("external_sampler", {}).items()
            if k in sig.parameters
        }
        self.external_sampler_uses_noise = "noise_sampler" in sig.parameters
        self.ancestralize = self.options.pop("ancestralize", self.ancestralize) is True

    def step(self, x):
        ss = self.ss
        sigmas = ss.sigmas[ss.idx : ss.idx + 2]
        kwargs = {
            "callback": None,
            "disable": True,
            "extra_args": {"seed": ss.noise.seed + ss.noise.seed_offset},
        } | self.external_sampler_options
        if self.external_sampler_uses_noise:
            kwargs["noise_sampler"] = ss.noise.make_caching_noise_sampler(
                self.options.get("custom_noise"),
                1,
                sigmas[-1],
                sigmas[0],
                immiscible=fallback(self.immiscible, ss.noise.immiscible),
            )

        mcc = 1

        def model_wrapper(x_, sigma_, *args, **kwargs):
            nonlocal mcc
            if torch.equal(x_, x) and sigma_ == ss.sigma:
                return ss.hcur.denoised.clone()
            mr = self.call_model(x_, sigma_, *args, call_index=mcc, **kwargs)
            mcc += 1
            return mr.denoised.clone()

        result = self.external_sampler.sampler_function(
            model_wrapper, x.clone(), sigmas, **kwargs
        )
        yield from self.result(result, ss.sigma.new_zeros(1))


# Based on https://github.com/Extraltodeus/DistanceSampler
class DistanceStep(SingleStepSampler):
    name = "distance"
    allow_alt_cfgpp = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        distance = self.options.get("distance", {})
        self.distance_resample = distance.get("resample", 3)
        self.distance_resample_end = distance.get("resample_end", 1)
        self.distance_resample_eta = distance.get("eta", self.eta)
        self.distance_resample_s_noise = distance.get("s_noise", self.s_noise)
        self.distance_alt_cfgpp_scale = distance.get("alt_cfgpp_scale", 0.0)
        self.distance_first_eta_resample_step = distance.get("first_eta_step", 0)
        self.distance_last_eta_resample_step = distance.get("last_eta_step", -1)

    @property
    def require_uncond(self):
        return super().require_uncond or self.distance_alt_cfgpp_scale != 0

    def distance_resample_steps(self):
        ss = self.ss
        resample, resample_end = self.distance_resample, self.distance_resample_end
        if resample == -1:
            current_resample = min(10, (ss.sigmas.shape[0] - ss.idx) // 2)
        else:
            current_resample = resample
        if resample_end < 0:
            return current_resample
        sigma = ss.sigma
        s_min = (ss.sigmas if ss.sigmas[-1] > 0 else ss.sigmas[:-1]).min()
        s_max = ss.sigmas.max()
        res_mul = max(0, min(1, ((sigma - s_min) / (s_max - s_min)) ** 0.5))
        return max(
            min(current_resample, resample_end),
            min(
                max(current_resample, resample_end),
                int(current_resample * res_mul + resample_end * (1 - res_mul)),
            ),
        )

    @staticmethod
    def distance_weights(t, p):
        batch = t.shape[0]
        d = torch.stack(
            tuple((t - t[idx]).abs().sum(dim=0) for idx in range(batch)),
            dim=0,
        )
        d_min, d_max = d.min(), d.max()
        d = torch.nan_to_num(
            (1 - (d - d_min) / (d_max - d_min)).pow(p),
            nan=1,
            neginf=1,
            posinf=1,
        )
        d /= d.sum(dim=0)
        return d.mul_(t).sum(dim=0)

    def step(self, x):
        resample_steps = self.distance_resample_steps()
        if resample_steps < 1:
            return (yield from self.euler_step(x))
        ss = self.ss
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        rsigma_down, rsigma_up = self.get_ancestral_step(eta=self.distance_resample_eta)
        rsigma_up *= self.distance_resample_s_noise
        sigma, sigma_next = ss.sigma, ss.sigma_next
        zero_up = sigma * 0
        d = self.to_d(ss.hcur)
        can_ancestral = not torch.equal(rsigma_down, sigma_next)
        start_eta_idx, end_eta_idx = (
            max(0, resample_steps + v if v < 0 else v)
            for v in (
                self.distance_first_eta_resample_step,
                self.distance_last_eta_resample_step,
            )
        )
        dt = sigma_down - sigma
        d = self.to_d(ss.hcur)
        x_n = [d]
        for re_step in tqdm.trange(
            resample_steps, desc="distance_resample", disable=ss.disable_status
        ):
            if can_ancestral and start_eta_idx <= re_step <= end_eta_idx:
                curr_sigma_down, curr_sigma_up = rsigma_down, rsigma_up
            else:
                curr_sigma_down, curr_sigma_up = sigma_next, zero_up
            rdt = curr_sigma_down - sigma
            x_new = x + d * rdt
            if curr_sigma_up != 0:
                x_new = yield from self.result(
                    x_new,
                    curr_sigma_up,
                    sigma=sigma,
                    sigma_down=curr_sigma_down,
                    final=False,
                )
            sr = self.call_model(x_new, sigma_next, call_index=re_step + 1)
            new_d = sr.to_d(
                sigma=curr_sigma_down, alt_cfgpp_scale=self.distance_alt_cfgpp_scale
            )
            x_n.append(new_d)
            if re_step == 0:
                d = (new_d + d) / 2
            else:
                d = self.distance_weights(torch.stack(x_n), re_step + 2)
                x_n.append(d)
        yield from self.result(x + d * dt, sigma_up, sigma_down=sigma_down)


class DynamicStep(SingleStepSampler):
    name = "dynamic"
    sample_sigma_zero = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dynamic = self.options.get("dynamic")
        if dynamic is None:
            raise ValueError(
                "Dynamic sampler type requires specifying dynamic block in text parameters"
            )
        if isinstance(dynamic, str):
            dynamic = ({"expression": dynamic},)
        elif not isinstance(dynamic, (tuple, list)):
            raise ValueError(
                "Bad type for dynamic block: must be string or list of objects"
            )
        elif len(dynamic) == 0:
            raise ValueError("Dynamic block as a list cannot be empty")
        dynresult = []
        for idx, item in enumerate(dynamic):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Bad item in dynamic block at index {idx}: must be a dict"
                )
            dyn_when = item.get("when")
            if isinstance(dyn_when, str):
                dyn_when = expr.Expression(dyn_when)
            elif dyn_when is not None:
                raise ValueError(
                    f"Unexpected type for when key in dynamic block at index {idx}, must be string or null/unset"
                )
            dyn_params = item.get("expression")
            if not isinstance(dyn_params, str):
                raise ValueError(
                    f"Missing or incorrectly typed expression key for dynamic block at index {idx}: must be a string"
                )
            dynresult.append((dyn_when, expr.Expression(dyn_params)))
        self.dynamic = tuple(dynresult)

    def step(self, x):
        sampler_params = None
        handlers = filtering.FILTER_HANDLERS.clone(constants=self.ss.refs)
        for idx, (dyn_when, dyn_params) in enumerate(self.dynamic):
            if dyn_when is not None and not bool(dyn_when.eval(handlers)):
                continue
            sampler_params = dyn_params.eval(handlers)
            if sampler_params is not None:
                break
        if sampler_params is None:
            raise RuntimeError(
                "Dynamic sampler could not find matching sampler: all expressions failed to return a result"
            )
        if not isinstance(sampler_params, dict):
            raise TypeError(
                f"Dynamic sampler expression must evaluate to a dict, got type {type(sampler_params)}"
            )
        if bool(sampler_params.get("dynamic_inherit")):
            copy_keys = (
                "s_noise",
                "eta",
                "pre_filter",
                "post_filter",
                "immiscible",
            )
            opts = {k: getattr(self, k) for k in copy_keys}
        else:
            opts = {}
        opts["custom_noise"] = self.custom_noise
        opts |= sampler_params
        opts |= {k: v for k, v in self.options.items() if k.startswith("custom_noise_")}
        # print("\n\nDYN OPTS", opts)
        step_method = opts.get("step_method", "default")
        sampler_class = STEP_SAMPLER_SIMPLE_NAMES.get(step_method)
        if sampler_class is None:
            raise ValueError(f"Unknown step method {step_method} in dynamic sampler")
        sampler = sampler_class(**opts)
        with StepSamplerContext(sampler, self.ss) as sampler:
            yield from sampler.step(x)


STEP_SAMPLERS = {
    "default (euler)": EulerStep,
    "adapter (variable)": AdapterStep,
    "bogacki (2)": BogackiStep,
    "deis": DEISStep,
    "distance (variable)": DistanceStep,
    "dpmpp_2m_sde": DPMPP2MSDEStep,
    "dpmpp_2m": DPMPP2MStep,
    "dpmpp_2s": DPMPP2SStep,
    "dpmpp_3m_sde": DPMPP3MSDEStep,
    "dpmpp_sde (1)": DPMPPSDEStep,
    "dynamic (variable)": DynamicStep,
    "euler_cycle": EulerCycleStep,
    "euler_dancing": EulerDancingStep,
    "euler": EulerStep,
    "heun (1)": HeunStep,
    "heun_1s (1)": Heun1SStep,
    "heunpp (1-2)": HeunPP2Step,
    "ipndm_v": IPNDMVStep,
    "ipndm": IPNDMStep,
    "res (1)": RESStep,
    "reversible_bogacki (2)": ReversibleBogackiStep,
    "reversible_heun (1)": ReversibleHeunStep,
    "reversible_heun_1s": ReversibleHeun1SStep,
    "rk4 (3)": RK4Step,
    "rkf45 (4)": RKF45Step,
    "rk_dynamic": RKDynamicStep,
    "solver_diffrax (variable)": DiffraxStep,
    "solver_torchdiffeq (variable)": TDEStep,
    "solver_torchode (variable)": TODEStep,
    "solver_torchsde (variable)": TSDEStep,
    "trapezoidal (1)": TrapezoidalStep,
    "trapezoidal_cycle (1)": TrapezoidalCycleStep,
    "ttm_jvp (1)": TTMJVPStep,
}

STEP_SAMPLER_SIMPLE_NAMES = {k.split(None, 1)[0]: v for k, v in STEP_SAMPLERS.items()}

__all__ = ("STEP_SAMPLERS", "STEP_SAMPLER_SIMPLE_NAMES")
