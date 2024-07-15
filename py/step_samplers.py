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

from .latent import Normalize
from .noise import ImmiscibleNoise
from .res_support import _de_second_order
from .utils import find_first_unsorted, extract_pred, fallback


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
        self.sampler = sampler
        self.sigma_up = fallback(sigma_up, ss.sigma.new_zeros(1))
        self.s_noise = fallback(s_noise, sampler.s_noise)
        self.sigma = fallback(sigma, ss.sigma)
        self.sigma_next = fallback(sigma_next, ss.sigma_next)
        self.sigma_down = fallback(sigma_down, self.sigma_next)
        self.noise_sampler = fallback(noise_sampler, sampler.noise_sampler)
        self.final = final
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

    def get_noise(self, scaled=True):
        if self.sigma_next == 0 or self.noise_scale == 0:
            return torch.zeros_like(self.x_)
        refs = {
            k: getattr(self, ak)
            for k, ak in (
                ("x", "x"),
                ("noise", "noise_pred"),
                ("denoised", "denoised"),
                ("uncond", "denoised_uncond"),
                ("cond", "denoised_cond"),
                ("sigma", "sigma"),
                ("sigma_next", "sigma_next"),
                ("sigma_down", "sigma_down"),
                ("sigma_up", "sigma_up"),
            )
            if getattr(self, ak, None) is not None
        }
        return self.noise_sampler(
            self.sigma,
            self.sigma_next,
            out_hw=self.x.shape[-2:],
            x_ref=self.x,
            refs=refs,
        ).mul_(self.noise_scale if scaled else 1.0)

    def extract_pred(self, ss):
        if self.denoised is None or self.noise_pred is None:
            self.denoised, self.noise_pred = extract_pred(
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

    def noise_x(self, x=None, scale=1.0):
        x = fallback(x, self.x)
        if self.sigma_next == 0 or self.noise_scale == 0:
            return x
        x = x + self.get_noise() * scale
        return x

    def clone(self):
        obj = self.__new__(self.__class__)
        for k in (
            "sampler",
            "sigma_up",
            "s_noise",
            "sigma",
            "sigma_next",
            "sigma_down",
            "noise_sampler",
            "final",
            "denoised",
            "denoised_uncond",
            "denoised_cond",
            "noise_pred",
            "x_",
        ):
            if hasattr(self, k):
                setattr(obj, k, getattr(self, k))
        return obj


class CFGPPStepMixin:
    allow_cfgpp = False
    allow_alt_cfgpp = False

    def __init__(self):
        self.cfgpp = self.allow_cfgpp and self.options.pop("cfgpp", False)
        alt_cfgpp_scale = self.options.pop("alt_cfgpp_scale", 0.0)
        self.alt_cfgpp_scale = 0.0 if not self.allow_alt_cfgpp else alt_cfgpp_scale

    def to_d(self, mr, **kwargs):
        return mr.to_d(alt_cfgpp_scale=self.alt_cfgpp_scale, cfgpp=self.cfgpp, **kwargs)


class SingleStepSampler(CFGPPStepMixin):
    name = None
    self_noise = 0
    model_calls = 0
    ancestralize = False
    sample_sigma_zero = False
    immiscible = None

    def __init__(
        self,
        *,
        noise_sampler=None,
        substeps=1,
        s_noise=1.0,
        eta=1.0,
        dyn_eta_start=None,
        dyn_eta_end=None,
        weight=1.0,
        normalize=(),
        immiscible=None,
        **kwargs,
    ):
        self.options = kwargs
        super().__init__()
        self.s_noise = s_noise
        self.eta = eta
        self.dyn_eta_start = dyn_eta_start
        self.dyn_eta_end = dyn_eta_end
        self.noise_sampler = noise_sampler
        self.immiscible = (
            ImmiscibleNoise(**immiscible)
            if immiscible not in (False, None)
            else immiscible
        )
        self.weight = weight
        self.substeps = substeps
        if isinstance(normalize, dict):
            normalize = (normalize,)
        if normalize:
            normalize = tuple(Normalize(**cfg) for cfg in normalize)
        else:
            normalize = ()
        self.normalize = normalize

    def __call__(self, x, ss):
        orig_x = x
        if not self.sample_sigma_zero and ss.sigma_next == 0:
            return (yield from self.denoised_result(ss))
        for n in self.normalize:
            x = n(ss, ss.sigma, x, "before")
        next_x = None
        sg = self.step(x, ss)
        with contextlib.suppress(StopIteration):
            while True:
                sr = sg.send(next_x)
                if sr.final:
                    if self.ancestralize:
                        sr = self.ancestralize_result(ss, sr)
                    curr_x = sr.x
                    for n in self.normalize:
                        curr_x = n(ss, ss.sigma, curr_x, "after", orig_x=orig_x)
                    if not torch.equal(curr_x, sr.x):
                        sr.x_ = curr_x
                    return (yield sr)
                next_x = sr.x
                yield sr

    def step(self, x, ss):
        raise NotImplementedError

    # Euler - based on original ComfyUI implementation
    def euler_step(self, x, ss):
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        d = self.to_d(ss.hcur)
        return (yield from self.result(ss, ss.denoised + d * sigma_down, sigma_up))

    def denoised_result(self, ss, **kwargs):
        return (
            yield SamplerResult(ss, self, ss.denoised, ss.sigma.new_zeros(1), **kwargs)
        )

    def result(self, ss, x, noise_scale=None, **kwargs):
        return (yield SamplerResult(ss, self, x, noise_scale, **kwargs))

    def split_result(
        self, ss, denoised, noise_pred, sigma_up=None, sigma_down=None, **kwargs
    ):
        return (
            yield SamplerResult(
                ss,
                self,
                None,
                sigma_up,
                sigma_down=sigma_down,
                split_result=(denoised, noise_pred),
                **kwargs,
            )
        )

    def ancestralize_result(self, ss, sr):
        new_sr = sr.clone()
        if new_sr.sigma_down is not None and new_sr.sigma_down != new_sr.sigma_next:
            return sr
        eta = self.get_dyn_eta(ss)
        if sr.sigma_next == 0 or eta == 0:
            return sr
        sd, su = ss.get_ancestral_step(eta, sigma=sr.sigma, sigma_next=sr.sigma_next)
        _ = new_sr.extract_pred(ss)
        new_sr.x_ = None
        new_sr.sigma_up = su
        new_sr.sigma_down = sd
        return new_sr

    def __str__(self):
        return f"<SS({self.name}): s_noise={self.s_noise}, eta={self.eta}>"

    def get_dyn_value(self, ss, start, end):
        if None in (start, end):
            return 1.0
        if start == end:
            return start
        main_idx = getattr(ss, "main_idx", ss.idx)
        main_sigmas = getattr(ss, "main_sigmas", ss.sigmas)
        step_pct = main_idx / (len(main_sigmas) - 1)
        dd_diff = end - start
        return start + dd_diff * step_pct

    def get_dyn_eta(self, ss):
        return self.eta * self.get_dyn_value(ss, self.dyn_eta_start, self.dyn_eta_end)

    def max_noise_samples(self):
        return (1 + self.self_noise) * self.substeps


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

    def available_history(self, ss):
        return max(
            0, min(ss.idx, self.history_limit, self.max_history, len(ss.hist) - 1)
        )


class ReversibleSingleStepSampler(HistorySingleStepSampler):
    def __init__(
        self,
        *,
        reversible_scale=1.0,
        reta=1.0,
        dyn_reta_start=None,
        dyn_reta_end=None,
        reversible_start_step=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reversible_scale = reversible_scale
        self.reta = reta
        self.reversible_start_step = reversible_start_step
        self.dyn_reta_start = dyn_reta_start
        self.dyn_reta_end = dyn_reta_end

    def reversible_correction(self, ss):
        raise NotImplementedError

    def get_dyn_reta(self, ss):
        if ss.step < self.reversible_start_step:
            return 0.0
        return self.reta * self.get_dyn_value(
            ss, self.dyn_reta_start, self.dyn_reta_end
        )

    def get_reversible_cfg(self, ss):
        if ss.step < self.reversible_start_step:
            return 0.0, 0.0
        return self.get_dyn_reta(ss), self.reversible_scale


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

    def adjusted_step(self, ss, sn, result, mcc, sigma_up):
        if sn == ss.sigma_next:
            return sigma_up, result
        # FIXME: Make sure we're noising from the right sigma.
        result = yield from self.result(
            ss, result, sigma_up, sigma=ss.sigma, sigma_next=sn, final=False
        )
        mr = ss.model(result, sn, model_call_idx=mcc)
        dt = ss.sigma_next - sn
        result = result + self.to_d(mr) * dt
        return sigma_up.new_zeros(1), result


class EulerStep(SingleStepSampler):
    name = "euler"
    allow_cfgpp = True
    step = SingleStepSampler.euler_step


class CycleSingleStepSampler(SingleStepSampler):
    def __init__(self, *, cycle_pct=0.25, **kwargs):
        super().__init__(**kwargs)
        self.cycle_pct = cycle_pct

    def get_cycle_scales(self, sigma_next):
        keep_scale = sigma_next * (1.0 - self.cycle_pct)
        add_scale = ((sigma_next**2.0 - keep_scale**2.0) ** 0.5) * (
            0.95 + 0.25 * self.cycle_pct
        )
        # print(f">> keep={keep_scale}, add={add_scale}")
        return keep_scale, add_scale


class EulerCycleStep(CycleSingleStepSampler):
    name = "euler_cycle"
    allow_alt_cfgpp = True
    allow_cfgpp = True

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.denoised_result(ss))
        d = self.to_d(ss.hcur)
        keep_scale, add_scale = self.get_cycle_scales(ss.sigma_next)
        yield from self.result(ss, ss.denoised + d * keep_scale, add_scale)


class DPMPP2MStep(HistorySingleStepSampler, DPMPPStepMixin):
    name = "dpmpp_2m"
    default_history_limit, max_history = 1, 1
    ancestralize = True

    def step(self, x, ss):
        s, sn = ss.sigma, ss.sigma_next
        t, t_next = self.t_fn(s), self.t_fn(sn)
        h = t_next - t
        st, st_next = self.sigma_fn(t), self.sigma_fn(t_next)
        if self.available_history(ss) > 0:
            h_last = t - self.t_fn(ss.sigma_prev)
            r = h_last / h
            denoised, old_denoised = ss.denoised, ss.hprev.denoised
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
        else:
            denoised_d = ss.denoised
        yield from self.result(ss, (st_next / st) * x - (-h).expm1() * denoised_d)


class DPMPP2MSDEStep(HistorySingleStepSampler):
    name = "dpmpp_2m_sde"
    default_history_limit, max_history = 1, 1

    def __init__(self, *, solver_type="midpoint", **kwargs):
        super().__init__(**kwargs)
        self.solver_type = solver_type

    def step(self, x, ss):
        denoised = ss.denoised
        # DPM-Solver++(2M) SDE
        t, s = -ss.sigma.log(), -ss.sigma_next.log()
        h = s - t
        eta_h = self.get_dyn_eta(ss) * h

        x = (
            ss.sigma_next / ss.sigma * (-eta_h).exp() * x
            + (-h - eta_h).expm1().neg() * denoised
        )
        noise_strength = ss.sigma_next * (-2 * eta_h).expm1().neg().sqrt()
        if self.available_history(ss) == 0:
            return (yield from self.result(ss, x, noise_strength))
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
        yield from self.result(ss, x, noise_strength)


class DPMPP3MSDEStep(HistorySingleStepSampler):
    name = "dpmpp_3m_sde"
    default_history_limit, max_history = 2, 2

    def step(self, x, ss):
        denoised = ss.denoised
        t, s = -ss.sigma.log(), -ss.sigma_next.log()
        h = s - t
        eta = self.get_dyn_eta(ss)
        h_eta = h * (eta + 1)

        x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised
        noise_strength = ss.sigma_next * (-2 * h * eta).expm1().neg().sqrt()
        ah = self.available_history(ss)
        if ah == 0:
            return (yield from self.result(ss, x, noise_strength))
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
        yield from self.result(ss, x, noise_strength)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class ReversibleHeunStep(ReversibleSingleStepSampler):
    name = "reversible_heun"
    model_calls = 1
    allow_alt_cfgpp = True
    allow_cfgpp = True

    def step(self, x, ss):
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        reta, reversible_scale = self.get_reversible_cfg(ss)
        sigma_down_reversible, _sigma_up_reversible = ss.get_ancestral_step(reta)
        dt_reversible = sigma_down_reversible - ss.sigma

        # Calculate the derivative using the model
        d = self.to_d(ss.hcur)

        # Predict the sample at the next sigma using Euler step
        x_pred = ss.denoised + d * sigma_down

        # Denoised sample at the next sigma
        mr_next = ss.model(x_pred, sigma_down, model_call_idx=1)

        # Calculate the derivative at the next sigma
        d_next = self.to_d(mr_next)

        # Update the sample using the Reversible Heun formula
        correction = dt_reversible**2 * (d_next - d) / 4
        x = (
            mr_next.denoised
            + (sigma_down * (d + d_next) / 2)
            - correction * reversible_scale
        )
        yield from self.result(ss, x, sigma_up)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class ReversibleHeun1SStep(ReversibleSingleStepSampler):
    name = "reversible_heun_1s"
    model_calls = 1
    default_history_limit, max_history = 1, 1
    allow_alt_cfgpp = True
    allow_cfgpp = True

    def step(self, x, ss):
        if self.available_history(ss) < 1:
            return (yield from ReversibleHeunStep.step(self, x, ss))
        s = ss.sigma
        # Reversible Heun-inspired update (first-order)
        sd, su = ss.get_ancestral_step(self.get_dyn_eta(ss))
        reta, reversible_scale = self.get_reversible_cfg(ss)
        sdr, _sur = ss.get_ancestral_step(reta)
        dt, dtr = sd - s, sdr - s
        # eff_x = ss.hist[-1].x if ah > 0 else x
        eff_x = x

        # Calculate the derivative using the model
        # d_prev = self.to_d(
        #     ss.hist[-2] if ah > 0 else ss.hist[-1],
        #     x=eff_x,
        #     sigma=s,
        # )
        prev_mr = ss.hist[-2]

        # d_prev = self.to_d(prev_mr, x=eff_x, sigma=s)
        d_prev = self.to_d(prev_mr, sigma=ss.sigma_prev)

        # Predict the sample at the next sigma using Euler step
        # x_pred = ss.denoised + d_prev * sd
        x_pred = eff_x + d_prev * dt
        # x_pred = ss.denoised + d_prev * sd

        # Calculate the derivative at the next sigma
        d_next = self.to_d(ss.hcur, x=x_pred, sigma=sd)

        # Update the sample using the Reversible Heun formula
        correction = dtr**2 * (d_next - d_prev) / 4
        x = x + (dt * (d_prev + d_next) / 2) - correction * reversible_scale
        yield from self.result(ss, x, su)

    def __step(self, x, ss):
        if ss.sigma_next == 0:
            return self.euler_step(x, ss)
        # Reversible Heun-inspired update (first-order)
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        sigma_down_reversible, sigma_up_reversible = ss.get_ancestral_step(
            self.get_dyn_reta(ss)
        )
        sigma_i, sigma_i_plus_1 = ss.sigma, sigma_down
        dt = sigma_i_plus_1 - sigma_i
        dt_reversible = sigma_down_reversible - sigma_i

        eff_x = ss.hist[-2 if len(ss.hist) > 1 else -1].x
        # eff_x = ss.hist[-2].x if len(ss.hist) > 1 else x

        # Calculate the derivative using the model
        eff_mr = ss.hprev if len(ss.hist) > 1 else ss.hcur
        d_i_old = self.to_d(eff_mr)
        # d_i_old = self.to_d(ss.hprev if len(ss.hist) > 1 else ss.hcur)
        # d_i_old = to_d(
        #     eff_x,
        #     sigma_i if len(ss.hist) == 1 else ss.sigma_prev,
        #     ss.hist[-2].denoised
        #     if len(ss.hist) > 1
        #     else ss.model(eff_x, sigma_i, model_call_idx=1).denoised,
        # )

        # Predict the sample at the next sigma using Euler step
        x_pred = eff_x + d_i_old * dt

        # Calculate the derivative at the next sigma
        d_i_plus_1 = to_d(x_pred, sigma_i_plus_1, ss.denoised)

        # Update the sample using the Reversible Heun formula
        x = (
            x
            + dt * (d_i_old + d_i_plus_1) / 2
            - dt_reversible**2 * (d_i_plus_1 - d_i_old) / 4
        )
        yield from self.result(ss, x, sigma_up)
        # return x, sigma_up

    def _step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
        ah = self.available_history(ss)
        s = ss.sigma
        # Reversible Heun-inspired update (first-order)
        sd, su = ss.get_ancestral_step(self.get_dyn_eta(ss))
        sdr, _sur = ss.get_ancestral_step(self.get_dyn_reta(ss))
        dt, dtr = sd - s, sdr - s
        # eff_mr = ss.hprev if ah > 0 else ss.hcur
        # eff_x = ss.hist[-1].x if ah > 0 else x  # This probably doesn't make sense.

        # Calculate the derivative using the model
        # mr_prev = ss.hist[-2] if ah > 0 else ss.model(eff_x, s, model_call_idx=1)
        mr_prev = ss.hist[-2 if ah > 0 else -1]
        d_prev = self.to_d(mr_prev, x=x, sigma=ss.sigma)
        # d_prev = self.to_d(
        #     mr_prev, sigma=ss.sigma_prev if ss.sigma_prev is not None else ss.sigma
        # )

        # Predict the sample at the next sigma using Euler step
        x_pred = ss.denoised + d_prev * sd
        # x_pred = mr_prev.denoised + d_prev * sd
        # x_pred = eff_x + d_prev * dt

        # Calculate the derivative at the next sigma
        d_next = self.to_d(ss.hcur, x=x_pred, sigma=sd)

        # Update the sample using the Reversible Heun formula
        correction = dtr**2 * (d_next - d_prev) / 4
        # x = x + (dt * (d_prev + d_next) / 2) - correction * self.reversible_scale
        x = (
            ss.denoised
            + (sd * (d_prev + d_next) / 2)
            - correction * self.reversible_scale
        )
        yield from self.result(ss, x, su)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class RESStep(SingleStepSampler):
    name = "res"
    model_calls = 1

    def __init__(self, *, res_simple_phi=False, res_c2=0.5, **kwargs):
        super().__init__(**kwargs)
        self.simple_phi = res_simple_phi
        self.c2 = res_c2

    def step(self, x, ss):
        eta = self.get_dyn_eta(ss)
        sigma_down, sigma_up = ss.get_ancestral_step(eta)
        denoised = ss.denoised
        lam_next = sigma_down.log().neg() if eta != 0 else ss.sigma_next.log().neg()
        lam = ss.sigma.log().neg()

        h = lam_next - lam
        a2_1, b1, b2 = _de_second_order(
            h=h, c2=self.c2, simple_phi_calc=self.simple_phi
        )

        c2_h = 0.5 * h

        x_2 = math.exp(-c2_h) * x + a2_1 * h * denoised
        lam_2 = lam + c2_h
        sigma_2 = lam_2.neg().exp()

        denoised2 = ss.model(x_2, sigma_2, model_call_idx=1).denoised

        x = math.exp(-h) * x + h * (b1 * denoised + b2 * denoised2)
        yield from self.result(ss, x, sigma_up)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class TrapezoidalStep(SingleStepSampler):
    name = "trapezoidal"
    model_calls = 1
    allow_alt_cfgpp = True

    def step(self, x, ss):
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))

        # Calculate the derivative using the model
        d_i = self.to_d(ss.hcur)

        # Predict the sample at the next sigma using Euler step
        x_pred = x + d_i * ss.dt

        # Denoised sample at the next sigma
        mr_next = ss.model(x_pred, ss.sigma_next, model_call_idx=1)

        # Calculate the derivative at the next sigma
        d_next = self.to_d(mr_next)
        dt_2 = sigma_down - ss.sigma

        # Update the sample using the Trapezoidal rule
        x = x + dt_2 * (d_i + d_next) / 2
        yield from self.result(ss, x, sigma_up)


class TrapezoidalCycleStep(CycleSingleStepSampler):
    name = "trapezoidal_cycle"
    model_calls = 1
    allow_alt_cfgpp = True

    def step(self, x, ss):
        # Calculate the derivative using the model
        d_i = self.to_d(ss.hcur)

        # Predict the sample at the next sigma using Euler step
        x_pred = x + d_i * ss.dt

        # Denoised sample at the next sigma
        mr_next = ss.model(x_pred, ss.sigma_next, model_call_idx=1)

        # Calculate the derivative at the next sigma
        d_next = self.to_d(mr_next)

        # Update the sample using the Trapezoidal rule
        keep_scale, add_scale = self.get_cycle_scales(ss.sigma_next)
        noise_pred = (d_i + d_next) * 0.5  # Combined noise prediction
        denoised_pred = x - noise_pred * ss.sigma  # Denoised prediction
        yield from self.result(ss, denoised_pred + noise_pred * keep_scale, add_scale)


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

    def step(self, x, ss):
        s = ss.sigma
        sd, su = ss.get_ancestral_step(self.get_dyn_eta(ss))
        reta, reversible_scale = self.get_reversible_cfg(ss)
        sdr, _sur = ss.get_ancestral_step(reta)
        dt, dtr = sd - s, sdr - s

        # Calculate the derivative using the model
        d = self.to_d(ss.hcur)

        # Bogacki-Shampine steps
        k1 = d * dt
        k2 = self.to_d(ss.model(x + k1 / 2, s + dt / 2, model_call_idx=1)) * dt
        k3 = (
            self.to_d(
                ss.model(x + 3 * k1 / 4 + k2 / 4, s + 3 * dt / 4, model_call_idx=2)
            )
            * dt
        )

        # Reversible correction term (inspired by Reversible Heun)
        correction = dtr**2 * (k3 - k2) / 6

        # Update the sample
        x = (x + 2 * k1 / 9 + k2 / 3 + 4 * k3 / 9) - correction * reversible_scale
        yield from self.result(ss, x, su)


class ReversibleBogackiStep(BogackiStep):
    name = "reversible_bogacki"
    reversible = True


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class RK4Step(SingleStepSampler):
    name = "rk4"
    model_calls = 3
    allow_alt_cfgpp = True

    def step(self, x, ss):
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        sigma = ss.sigma
        # Calculate the derivative using the model
        d = to_d(x, sigma, ss.denoised)
        dt = sigma_down - sigma

        # Runge-Kutta steps
        k1 = d * dt
        k2 = self.to_d(ss.model(x + k1 / 2, sigma + dt / 2, model_call_idx=1)) * dt
        k3 = self.to_d(ss.model(x + k2 / 2, sigma + dt / 2, model_call_idx=2)) * dt
        k4 = self.to_d(ss.model(x + k3, sigma + dt, model_call_idx=3)) * dt

        # Update the sample
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        yield from self.result(ss, x, sigma_up)


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

    def step(self, x, ss):
        eta = self.eta
        deta = self.deta
        leap_sigmas = ss.sigmas[ss.idx :]
        leap_sigmas = leap_sigmas[: find_first_unsorted(leap_sigmas)]
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
            return (yield from self.result(ss, x, sigma_up))
        noise_strength = self.ds_noise * sigma_up
        if noise_strength != 0:
            x = yield from self.result(
                ss, x, sigma_up, sigma_next=sigma_leap, final=False
            )

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
        yield from self.result(ss, x, sigma_up2)

    def _step(self, x, ss):
        eta = self.get_dyn_eta(ss)
        leap_sigmas = ss.sigmas[ss.idx :]
        leap_sigmas = leap_sigmas[: find_first_unsorted(leap_sigmas)]
        zero_idx = (leap_sigmas <= 0).nonzero().flatten()[:1]
        max_leap = (zero_idx.item() if len(zero_idx) else len(leap_sigmas)) - 1
        is_danceable = max_leap > 1 and ss.sigma_next != 0
        curr_leap = max(1, min(self.leap, max_leap))
        sigma_leap = leap_sigmas[curr_leap] if is_danceable else ss.sigma_next
        # DANCE 35 6 tensor(10.0947, device='cuda:0') -- tensor([21.9220,
        # print("DANCE", max_leap, curr_leap, sigma_leap, "--", leap_sigmas)
        del leap_sigmas
        sigma_down, sigma_up = get_ancestral_step(ss.sigma, sigma_leap, eta)
        d = to_d(x, ss.sigma, ss.denoised)
        # Euler method
        dt = sigma_down - ss.sigma
        x = x + d * dt
        if curr_leap == 1:
            return x, sigma_up
        dance_scale = self.get_dyn_value(ss, self.dyn_deta_start, self.dyn_deta_end)
        if curr_leap == 1 or not is_danceable or abs(dance_scale) < 1e-04:
            print("NODANCE", dance_scale, self.deta, is_danceable, ss.sigma_next)
            yield SamplerResult(ss, self, x, sigma_up)
        print(
            "DANCE", dance_scale, self.deta, self.dyn_deta_mode, self.ds_noise, sigma_up
        )
        sigma_down_normal, sigma_up_normal = get_ancestral_step(
            ss.sigma, ss.sigma_next, eta
        )
        if self.dyn_deta_mode == "lerp":
            dt_normal = sigma_down_normal - ss.sigma
            x_normal = x + d * dt_normal
        else:
            x_normal = x
        sigma_down2, sigma_up2 = get_ancestral_step(
            sigma_leap,
            ss.sigma_next,
            eta=self.deta * (1.0 if self.dyn_deta_mode != "deta" else dance_scale),
        )
        print(
            "-->",
            sigma_down2,
            sigma_up2,
            "--",
            self.deta * (1.0 if self.dyn_deta_mode != "deta" else dance_scale),
        )
        x = x + self.noise_sampler(ss.sigma, sigma_leap).mul_(self.ds_noise * sigma_up)
        d_2 = to_d(x, sigma_leap, ss.denoised)
        dt_2 = sigma_down2 - sigma_leap
        result = x + d_2 * dt_2
        # SIGMA: norm_up=9.062416076660156, up=10.703859329223633, up2=19.376544952392578, str=21.955078125
        noise_strength = sigma_up2 + ((sigma_up - sigma_up_normal) ** 5.0)
        noise_strength = sigma_up2 + ((sigma_up2 - sigma_up) * 0.5)
        # noise_strength = sigma_up2 + (
        #     (sigma_up2 - sigma_up) ** (1.0 - (sigma_up_normal / sigma_up2))
        # )
        noise_diff = (
            sigma_up - sigma_up_normal
            if sigma_up > sigma_up_normal
            else sigma_up_normal - sigma_up
        )
        noise_div = (
            sigma_up / sigma_up_normal
            if sigma_up > sigma_up_normal
            else sigma_up_normal / sigma_up
        )
        noise_diff = sigma_up2 - sigma_up_normal
        noise_div = sigma_up2 / sigma_up_normal
        noise_div = ss.sigma / sigma_leap

        # noise_strength = sigma_up2 + (noise_diff * noise_div)
        # noise_strength = sigma_up2 + ((noise_diff * 0.5) ** 2.0)
        # noise_strength = sigma_up2 + ((1.0 - noise_diff) ** 0.5)
        # noise_strength = sigma_up2 + (((sigma_up2 - sigma_up) * 0.5) ** 2.0)
        # noise_strength = sigma_up2 + (((sigma_up2 - sigma_up_normal) * 0.5) ** 1.5)
        # noise_strength = sigma_up2 + (
        #     (noise_diff * 0.1875) ** (1.0 / (noise_div - 0.0))
        # )
        # noise_strength = sigma_up2 + (
        #     (noise_diff * 0.125) ** (1.0 / (noise_div * 1.25))
        # )
        # noise_strength = sigma_up2 + ((noise_diff * 0.2) ** (1.0 / (noise_div * 1.0)))
        noise_strength = sigma_up2 + (noise_diff * 0.9 * max(0.0, noise_div - 0.8))
        noise_strength = sigma_up2 + (
            (noise_diff / (curr_leap * 0.4))
            * ((noise_div - (curr_leap / 2.0)).clamp(min=0, max=1.5) * 1.0)
        )
        # (1.0 / (noise_div * 1.25)))
        # noise_strength = sigma_up2 + ((noise_diff * 0.5) ** noise_div)
        print(
            f"SIGMA: norm_up={sigma_up_normal}, up={sigma_up}, up2={sigma_up2}, str={noise_strength}",
            # noise_diff,
            noise_div,
        )
        return result, noise_strength

        noise_diff = sigma_up2 - sigma_up * dance_scale
        noise_scale = sigma_up2 + noise_diff * (0.025 * curr_leap)
        # noise_scale = sigma_up2 * self.ds_noise
        if self.dyn_deta_mode == "deta" or dance_scale == 1.0:
            return result, noise_scale
        result = torch.lerp(x_normal, result, dance_scale)
        # FIXME: Broken for noise samplers that care about s/sn
        return result, noise_scale

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


class DPMPP2SStep(SingleStepSampler, DPMPPStepMixin):
    name = "dpmpp_2s"
    model_calls = 1

    def step(self, x, ss):
        t_fn, sigma_fn = self.t_fn, self.sigma_fn
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        # DPM-Solver++(2S)
        t, t_next = t_fn(ss.sigma), t_fn(sigma_down)
        r = 1 / 2
        h = t_next - t
        s = t + r * h
        x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * ss.denoised
        denoised_2 = ss.model(x_2, sigma_fn(s), model_call_idx=1).denoised
        x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        yield from self.result(ss, x, sigma_up)


class DPMPPSDEStep(SingleStepSampler, DPMPPStepMixin):
    name = "dpmpp_sde"
    self_noise = 1
    model_calls = 1

    def __init__(self, *args, r=1 / 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r

    def step(self, x, ss):
        t_fn, sigma_fn = self.t_fn, self.sigma_fn
        r, eta = self.r, self.get_dyn_eta(ss)
        # DPM-Solver++
        t, t_next = t_fn(ss.sigma), t_fn(ss.sigma_next)
        h = t_next - t
        s = t + h * r
        fac = 1 / (2 * r)

        # Step 1
        sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
        s_ = t_fn(sd)
        x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * ss.denoised
        x_2 = yield from self.result(
            ss, x_2, su, sigma=sigma_fn(t), sigma_next=sigma_fn(s), final=False
        )
        denoised_2 = ss.model(x_2, sigma_fn(s), model_call_idx=1).denoised

        # Step 2
        sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
        t_next_ = t_fn(sd)
        denoised_d = (1 - fac) * ss.denoised + fac * denoised_2
        x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
        yield from self.result(ss, x, su)


# Based on implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
# Which was originally written by Katherine Crowson
class TTMJVPStep(SingleStepSampler):
    name = "ttm_jvp"
    model_calls = 1

    def __init__(self, *args, alternate_phi_2_calc=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.alternate_phi_2_calc = alternate_phi_2_calc

    def step(self, x, ss):
        eta = self.get_dyn_eta(ss)
        sigma, sigma_next = ss.sigma, ss.sigma_next
        # 2nd order truncated Taylor method
        t, s = -sigma.log(), -sigma_next.log()
        h = s - t
        h_eta = h * (eta + 1)

        eps = to_d(x, sigma, ss.denoised)
        denoised_prime = ss.model(
            x, sigma, tangents=(eps * -sigma, -sigma), model_call_idx=1
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
        yield from self.result(ss, x, noise_scale)


# Adapted from https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
# under Apache 2 license
class IPNDMStep(HistorySingleStepSampler):
    name = "ipndm"
    ancestralize = True
    default_history_limit, max_history = 1, 3
    allow_alt_cfgpp = True

    IPNDM_MULTIPLIERS = (
        ((1,), 1),
        ((3, -1), 2),
        ((23, -16, 5), 12),
        ((55, -59, 37, -9), 24),
    )

    def step(self, x, ss):
        order = self.available_history(ss) + 1
        if order > 1:
            hd = tuple(self.to_d(ss.hist[-hidx]) for hidx in range(order, 1, -1))
        (dm, *hms), divisor = self.IPNDM_MULTIPLIERS[order - 1]
        noise = dm * self.to_d(ss.hcur)
        for hidx, hm in enumerate(hms, start=1):
            noise += hm * hd[-hidx]
        noise /= divisor
        yield from self.result(ss, x + ss.dt * noise)


# Adapted from https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
# under Apache 2 license
class IPNDMVStep(HistorySingleStepSampler):
    name = "ipndm_v"
    ancestralize = True
    default_history_limit, max_history = 1, 3
    allow_alt_cfgpp = True

    def step(self, x, ss):
        dt = ss.dt
        d = self.to_d(ss.hcur)
        order = self.available_history(ss) + 1
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
        yield from self.result(ss, x + ss.dt * noise)


class DEISStep(HistorySingleStepSampler):
    name = "deis"
    ancestralize = True
    default_history_limit, max_history = 1, 3
    allow_alt_cfgpp = True

    def __init__(self, *args, deis_mode="tab", **kwargs):
        super().__init__(*args, **kwargs)
        self.deis_mode = deis_mode
        self.deis_coeffs_key = None
        self.deis_coeffs = None

    def get_deis_coeffs(self, ss):
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

    def step(self, x, ss):
        dt = ss.dt
        d = self.to_d(ss.hcur)
        order = self.available_history(ss) + 1
        if order < 2:
            noise = dt * d  # Euler
        else:
            c = self.get_deis_coeffs(ss)[ss.idx]
            hd = tuple(self.to_d(ss.hist[-hidx]) for hidx in range(order, 1, -1))
            noise = c[0] * d
            for i in range(1, order):
                noise += c[i] * hd[-i]
        yield from self.result(ss, x + noise)


class HeunPP2Step(SingleStepSampler):
    name = "heunpp2"
    ancestralize = True
    model_calls = 2
    allow_alt_cfgpp = True

    def __init__(self, *args, max_order=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_order = max(1, min(self.model_calls + 1, max_order))

    def step(self, x, ss):
        steps_remain = max(0, len(ss.sigmas) - (ss.idx + 2))
        order = min(self.max_order, steps_remain + 1)
        sn = ss.sigma_next
        if order == 1:
            return (yield from self.euler_step(x, ss))
        d = self.to_d(ss.hcur)
        dt = ss.dt
        w = order * ss.sigma
        w2 = sn / w
        x_2 = x + d * dt
        d_2 = self.to_d(ss.model(x_2, sn, model_call_idx=1))
        if order == 2:
            # Heun's method (ish)
            w1 = 1 - w2
            d_prime = d * w1 + d_2 * w2
        else:
            # Heun++ (ish)
            snn = ss.sigmas[ss.idx + 2]
            dt_2 = snn - sn
            x_3 = x_2 + d_2 * dt_2
            d_3 = self.to_d(ss.model(x_3, snn, model_call_idx=2))
            w3 = snn / w
            w1 = 1 - w2 - w3
            d_prime = w1 * d + w2 * d_2 + w3 * d_3
        yield from self.result(ss, x + d_prime * dt)


class DESolverStep(SingleStepSampler, MinSigmaStepMixin):
    de_default_solver = None
    sample_sigma_zero = True

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

    def de_get_step(self, ss, x):
        eta = self.get_dyn_eta(ss)
        s, sn = ss.sigma, ss.sigma_next
        sn = self.adjust_step(sn, self.de_min_sigma)
        sigma_down, sigma_up = ss.get_ancestral_step(eta, sigma_next=sn)
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

    def step(self, x, ss):
        s, sn, sigma_down, sigma_up = self.de_get_step(ss, x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x, ss))
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
                mr = ss.hcur
                mcc = 1
            else:
                mr = ss.model(y.unsqueeze(0), t, model_call_idx=mcc, s_in=t.new_ones(1))
                mcc += 1
            return self.to_d(mr)[0]

        result = torch.zeros_like(x)
        t = sigma_down.new_zeros(self.de_split + 1)
        torch.linspace(ss.sigma, sigma_down, t.shape[0], out=t)

        for batch in tqdm.trange(
            1,
            x.shape[0] + 1,
            desc="batch",
            leave=True,
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

        sigma_up, result = yield from self.adjusted_step(ss, sn, result, mcc, sigma_up)
        if pbar is not None:
            pbar.n = pbar.total
            pbar.update(0)
            pbar.close()
        yield from self.result(ss, result, sigma_up, sigma_down=sigma_down)


class TODEStep(DESolverStep):
    name = "tode"
    model_calls = 2
    allow_alt_cfgpp = True
    de_default_solver = "dopri5"

    def __init__(
        self,
        *args,
        de_initial_step=0.25,
        de_compile=False,
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
        self.de_compile = de_compile
        self.de_initial_step = de_initial_step

    def check_solver_support(self):
        if not HAVE_TODE:
            raise RuntimeError(
                "TODE sampler requires torchode installed in venv. Example: pip install torchode"
            )

    def step(self, x, ss):
        s, sn, sigma_down, sigma_up = self.de_get_step(ss, x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x, ss))
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
                mr = ss.model(y, t32.clamp(min=1e-05), model_call_idx=mcc)
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

        sigma_up, result = yield from self.adjusted_step(ss, sn, result, mcc, sigma_up)
        if pbar is not None:
            pbar.n = pbar.total
            pbar.update(0)
            pbar.close()
        yield from self.result(ss, result, sigma_up, sigma_down=sigma_down)


class TSDEStep(DESolverStep):
    name = "tsde"
    model_calls = 2
    allow_alt_cfgpp = True
    de_default_solver = "reversible_heun"

    def __init__(
        self,
        *args,
        de_initial_step=0.25,
        de_split=1,
        de_adaptive=False,
        de_noise_type="scalar",
        de_sde_type="stratonovich",
        de_levy_area_approx="none",
        de_noise_channels=1,
        de_g_multiplier=0.05,
        de_g_reverse_time=True,
        de_g_derp_mode=False,
        de_batch_channels=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.de_initial_step = de_initial_step
        self.de_adaptive = de_adaptive
        self.de_split = de_split
        self.de_noise_type = de_noise_type
        self.de_sde_type = de_sde_type
        self.de_levy_area_approx = de_levy_area_approx
        self.de_g_multiplier = de_g_multiplier
        self.de_noise_channels = de_noise_channels
        self.de_g_reverse_time = de_g_reverse_time
        self.de_g_derp_mode = de_g_derp_mode
        self.de_batch_channels = de_batch_channels

    def check_solver_support(self):
        pass

    def step(self, x, ss):
        s, sn, sigma_down, sigma_up = self.de_get_step(ss, x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x, ss))
        delta = (ss.sigma - sigma_down).item()
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
                    mr = ss.hcur
                    mcc = 1
                else:
                    mr = ss.model(y, t32.clamp(min=1e-05), model_call_idx=mcc)
                    mcc += 1
                return -outer_self.to_d(mr).view(*flat_shape)

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
        for bidx in range(x.shape[0]):
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
            # print("DONE", ys.shape)
            results.append(ys[-1].view(1, c, h, w))
            # result = ys[-1].reshape(-1, c, h, w)
            del ys
        result = torch.cat(results)
        del results

        sigma_up, result = yield from self.adjusted_step(ss, sn, result, mcc, sigma_up)
        if pbar is not None:
            pbar.n = pbar.total
            pbar.update(0)
            pbar.close()
        yield from self.result(ss, result, sigma_up, sigma_down=sigma_down)


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

    def __init__(
        self,
        *args,
        de_split=1,
        de_adaptive=False,
        de_fake_pure_callback=True,
        de_initial_step=0.25,
        de_ctl_pcoeff=0.3,
        de_ctl_icoeff=0.9,
        de_ctl_dcoeff=0.2,
        de_g_multiplier=0.0,
        de_half_solver=False,
        de_batch_channels=False,
        de_levy_area_approx="brownian_increment",
        de_error_order=None,
        de_sde_mode=False,
        de_g_reverse_time=False,
        de_g_time_scaling=False,
        de_g_split_time_mode=False,
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
        if de_half_solver:
            self.de_solver_method = diffrax.HalfSolver(self.de_solver_method)
        self.de_ctl_pcoeff = de_ctl_pcoeff
        self.de_ctl_icoeff = de_ctl_icoeff
        self.de_ctl_dcoeff = de_ctl_dcoeff
        self.de_initial_step = de_initial_step
        self.de_adaptive = de_adaptive
        self.de_split = de_split
        self.de_fake_pure_callback = de_fake_pure_callback
        self.de_g_multiplier = de_g_multiplier
        self.de_batch_channels = de_batch_channels
        self.de_levy_area_approx = levy_areas[de_levy_area_approx]
        self.de_error_order = de_error_order
        self.de_sde_mode = de_sde_mode
        self.de_g_reverse_time = de_g_reverse_time
        self.de_g_time_scaling = de_g_time_scaling
        self.de_g_split_time_mode = de_g_split_time_mode

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

    def step(self, x, ss):
        s, sn, sigma_down, sigma_up = self.de_get_step(ss, x)
        if self.de_min_sigma is not None and s <= self.de_min_sigma:
            return (yield from self.euler_step(x, ss))
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
                mr = ss.hcur
                mcc = 1
            else:
                try:
                    if not args:
                        mr = ss.model(y, t32, model_call_idx=mcc)
                    else:
                        print("TANGENTS")
                        mr = ss.model(y, t32, model_call_idx=mcc, tangents=args)
                except comfy.model_management.InterruptProcessingException as exc:
                    interrupted = exc
                    raise
                mcc += 1
            result = self.to_d(mr).reshape(*flat_shape)
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
                # t0,
                # t1 + 1e-06,
                t0=ss.sigmas.min().item(),
                t1=ss.sigmas.max().item(),
                tol=1e-06,
                levy_area=self.de_levy_area_approx,
                shape=(c,) if self.de_batch_channels else (),
                key=jax.random.PRNGKey(ss.noise.seed + ss.noise.seed_offset),
            )
            term = diffrax.MultiTerm(term, diffrax.ControlTerm(g, bm))
        results = []
        for bidx in range(x.shape[0]):
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
            results.append(self.j2t(solution.ys).view(*x.shape))
            del solution
        result = torch.cat(results).to(x)
        sigma_up, result = yield from self.adjusted_step(ss, sn, result, mcc, sigma_up)
        yield from self.result(ss, result, sigma_up, sigma_down=sigma_down)


class HeunStep(ReversibleSingleStepSampler):
    name = "heun"
    model_calls = 1
    default_history_limit, max_history = 0, 0
    allow_alt_cfgpp = True

    def reversible_correction(self, ss, d_from, d_to):
        reta, reversible_scale = self.get_reversible_cfg(ss)
        if reversible_scale == 0:
            return 0
        sdr = ss.get_ancestral_step(reta)[0]
        dtr = sdr - ss.sigma
        return (dtr**2 * (d_to - d_from) / 4) * self.reversible_scale

    def step(self, x, ss):
        s = ss.sigma
        sd, su = ss.get_ancestral_step(self.get_dyn_eta(ss))
        dt = sd - s
        hcur = ss.hcur
        d = self.to_d(hcur)
        x_next = hcur.denoised + d * sd
        d_next = self.to_d(ss.model(x_next, sd))
        result = hcur.denoised + d * s
        result += (dt * (d + d_next)) * 0.5
        result -= self.reversible_correction(ss, d, d_next)
        yield from self.result(ss, result, su)


class Heun1SStep(HeunStep):
    name = "heun_1s"
    model_calls = 1
    allow_alt_cfgpp = True
    default_history_limit, max_history = 1, 1

    def step(self, x, ss):
        s = ss.sigma
        if self.available_history(ss) == 0:
            return (yield from super().step(x, ss))
        hcur, hprev = ss.hcur, ss.hprev
        d_prev = self.to_d(hprev)
        sd, su = ss.get_ancestral_step(self.get_dyn_eta(ss))
        dt = sd - s
        d = self.to_d(hcur)
        result = hcur.denoised + hcur.sigma * self.to_d(hcur)
        result += (dt * (d_prev + d)) * 0.5
        result -= self.reversible_correction(ss, d_prev, d)
        yield from self.result(ss, result, su)


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

    def step(self, x, ss):
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
            mr = ss.model(x_, sigma_, *args, model_call_idx=mcc, **kwargs)
            mcc += 1
            return mr.denoised.clone()

        result = self.external_sampler.sampler_function(
            model_wrapper, x.clone(), sigmas, **kwargs
        )
        yield from self.result(ss, result, ss.sigma.new_zeros(1))


STEP_SAMPLERS = {
    "default (euler)": EulerStep,
    "adapter (variable)": AdapterStep,
    "bogacki (2)": BogackiStep,
    "deis": DEISStep,
    "dpmpp_2m_sde": DPMPP2MSDEStep,
    "dpmpp_2m": DPMPP2MStep,
    "dpmpp_2s": DPMPP2SStep,
    "dpmpp_3m_sde": DPMPP3MSDEStep,
    "dpmpp_sde (1)": DPMPPSDEStep,
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
    "solver_diffrax (variable)": DiffraxStep,
    "solver_torchdiffeq (variable)": TDEStep,
    "solver_torchode (variable)": TODEStep,
    "solver_torchsde (variable)": TSDEStep,
    "trapezoidal (1)": TrapezoidalStep,
    "trapezoidal_cycle (1)": TrapezoidalCycleStep,
    "ttm_jvp (1)": TTMJVPStep,
}

__all__ = (
    "STEP_SAMPLERS",
    "EulerStep",
    "EulerCycleStep",
    "DPMPP2MStep",
    "DPMPP2MSDEStep",
    "DPMPP3MSDEStep",
    "DPMPP2SStep",
    "ReversibleHeunStep",
    "ReversibleHeun1SStep",
    "RESStep",
    "TrapezoidalCycleStep",
    "TrapezoidalStep",
    "BogackiStep",
    "ReversibleBogackiStep",
    "EulerDancingStep",
    "TTMJVPStep",
    "IPNDMStep",
    "IPNDMVStep",
    "TDEStep",
)
