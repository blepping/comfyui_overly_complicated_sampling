import torch

import comfy
from comfy.k_diffusion.sampling import get_ancestral_step

from .base import (
    SingleStepSampler,
    DPMPPStepMixin,
    HistorySingleStepSampler,
    ReversibleSingleStepSampler,
    registry,
)


class EulerStep(SingleStepSampler):
    name = "euler"
    allow_cfgpp = True
    allow_alt_cfgpp = True
    step = SingleStepSampler.euler_step


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


class DPMPP2MSDEStep(ReversibleSingleStepSampler):
    name = "dpmpp_2m_sde"
    default_history_limit, max_history = 1, 1
    default_reversible_scale = 0.0

    def __init__(self, *, solver_type="midpoint", **kwargs):
        super().__init__(**kwargs)
        solver_type = solver_type.lower().strip()
        if solver_type not in ("midpoint", "heun"):
            raise ValueError("Bad solver_type: must be one of midpoint, heun")
        self.solver_type = solver_type

    def step(self, x):
        ss = self.ss
        sigma, sigma_next = ss.sigma, ss.sigma_next
        denoised = ss.denoised
        # DPM-Solver++(2M) SDE
        t, s = -sigma.log(), -sigma_next.log()
        h = s - t
        eta_h = self.get_dyn_eta() * h
        ratio = sigma_next / sigma
        x = ((ratio * (-eta_h).exp()) * x).add_((-h - eta_h).expm1().neg() * denoised)
        noise_strength = sigma_next * (-2 * eta_h).expm1().neg().sqrt()
        if self.available_history() == 0:
            return (yield from self.result(x, noise_strength))
        sigma_prev, old_denoised = ss.hprev.sigma, ss.hprev.denoised
        h_last = (-sigma.log()) - (-sigma_prev.log())
        r = h_last / h
        if self.solver_type == "midpoint":
            multiplier = 0.5 * (-h - eta_h).expm1().neg()
        else:
            multiplier = (-h - eta_h).expm1().neg() / (-h - eta_h) + 1
        reta, reversible_scale = self.get_reversible_cfg()
        if reversible_scale != 0:
            multiplier *= 0.5
        x += (denoised - old_denoised).mul_((1 / r) * multiplier)
        if reversible_scale != 0:
            reta_h = reta * h
            if self.solver_type == "midpoint":
                rmultiplier = 0.5 * (-h - reta_h).expm1().neg()
            else:
                rmultiplier = (-h - reta_h).expm1().neg() / (-h - reta_h) + 1
            rmultiplier = ((1 / r) * (rmultiplier**2 / 2)) * reversible_scale
            x -= (old_denoised - denoised).mul_(rmultiplier)
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

    def step_(self, x):
        sr = next(super().step(x))
        if self.available_history() < 2:
            yield sr
            return
        ss = self.ss
        sigma, sigma_next = ss.sigma, ss.sigma_next
        hprev = ss.hist[-2]
        hprevprev = ss.hist[-3]
        t, s = -sigma.log(), -sigma_next.log()
        h = s - t
        eta = self.get_dyn_eta()
        h_eta = h * (eta + 1)
        h_2 = (-ss.sigma_prev.log()) - (-hprevprev.sigma.log())
        denoised = ss.denoised
        denoised_1 = hprev.denoised
        denoised_2 = hprevprev.denoised
        h_1 = (-ss.sigma.log()) - (-hprev.sigma.log())
        r0 = h_1 / h
        r1 = h_2 / h
        d1_0 = (denoised - denoised_1).div_(r0)
        d1_1 = (denoised_1 - denoised_2).div_(r1)
        d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
        d2 = (d1_0 - d1_1).div_(r0 + r1)
        phi_2 = h_eta.neg().expm1() / h_eta + 1
        phi_3 = phi_2 / h_eta - 0.5
        sr.x_ += phi_2 * d1 - phi_3 * d2
        yield sr
        # x = x + phi_2 * d1 - phi_3 * d2
        # yield from self.result(sr.x + phi_2 * d1 - phi_3 * d2, sr.sigma_up)


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
    uses_alt_noise = True

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
            x_2,
            su,
            sigma=sigma_fn(t),
            sigma_next=sigma_fn(s),
            noise_sampler=self.alt_noise_sampler,
            final=False,
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


# https://openreview.net/pdf?id=o2ND9v0CeK
# Implementation referenced from ComfyUI
class GradientEstimationStep(HistorySingleStepSampler):
    name = "gradient_estimation"
    ancestralize = False
    default_history_limit, max_history = 1, 1
    default_eta = 0.0

    def __init__(self, *args, ge_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ge_gamma = ge_gamma

    def step(self, x):
        ss = self.ss
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        dt = sigma_down - ss.sigma
        d = self.to_d(ss.hcur)
        if self.available_history() < 1:
            noise_pred = dt * d  # Euler
        else:
            gamma = self.ge_gamma
            noise_pred = dt * (gamma * d + (1 - gamma) * ss.hist[-2].d)
        yield from self.result(x + noise_pred, sigma_up, sigma_down=sigma_down)


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


# Referenced from ComfyUI implementation
class DPM2Step(SingleStepSampler):
    name = "dpm_2"
    model_calls = 1

    def step(self, x):
        ss = self.ss
        sigma, sigma_next = ss.sigma, ss.sigma_next
        eta = self.get_dyn_eta()
        sigma_down, sigma_up = self.get_ancestral_step(eta)
        sigma_mid = sigma.log().lerp(sigma_next.log(), 0.5).exp()
        dt_1, dt_2 = sigma_mid - sigma, sigma_down - sigma
        d = self.to_d(ss.hcur)
        mr_2 = self.call_model(x + d * dt_1, sigma_mid, call_index=1)
        d_2 = self.to_d(mr_2)
        yield from self.result(x + d_2 * dt_2, sigma_up)


# Referenced from ComfyUI implementation
class RESMultistepStep(HistorySingleStepSampler, DPMPPStepMixin):
    name = "res_multistep"
    default_history_limit, max_history = 1, 1
    default_eta = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def phi1_fn(t: torch.Tensor) -> torch.Tensor:
        return t.expm1() / t

    @classmethod
    def phi2_fn(cls, t: torch.Tensor) -> torch.Tensor:
        return (cls.phi1_fn(t) - 1.0) / t

    def step(self, x):
        ss = self.ss
        sigma = ss.sigma
        eta = self.get_dyn_eta()
        sigma_down, sigma_up = self.get_ancestral_step(eta)
        if self.available_history() == 0:
            dt = sigma_down - sigma
            d = self.to_d(ss.hcur)
            return (yield from self.result(x + dt * d, sigma_up, sigma_down=sigma_down))
        prev_mr = ss.hist[-2]
        prev_sigma_down = self.get_ancestral_step(
            sigma=prev_mr.sigma, sigma_next=sigma, eta=eta
        )[0]
        # Second order multistep method in https://arxiv.org/pdf/2308.02157
        t, t_old, t_next, t_prev = (
            self.t_fn(sigma),
            self.t_fn(prev_sigma_down),
            self.t_fn(sigma_down),
            self.t_fn(prev_mr.sigma),
        )
        h = t_next - t
        h_s = self.sigma_fn(h)
        c2 = (t_prev - t_old) / h

        phi1_val, phi2_val = self.phi1_fn(-h), self.phi2_fn(-h)
        b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
        b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
        result = h_s * x + h * (b1 * ss.denoised + b2 * prev_mr.denoised)
        yield from self.result(result, sigma_up, sigma_down=sigma_down)


registry.add(
    DEISStep,
    DPMPP2MSDEStep,
    DPMPP2MStep,
    DPMPP3MSDEStep,
    DPMPPSDEStep,
    EulerStep,
    HeunPP2Step,
    IPNDMStep,
    IPNDMVStep,
    GradientEstimationStep,
    DPM2Step,
    DPMPP2SStep,
    RESMultistepStep,
)
