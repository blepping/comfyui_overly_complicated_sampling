import math

import torch

from comfy.k_diffusion.sampling import (
    get_ancestral_step,
    to_d,
)

from .res_support import _de_second_order
from .utils import find_first_unsorted


class SamplerResult:
    def __init__(
        self,
        ss,
        sampler,
        x,
        strength=None,
        *,
        sigma=None,
        sigma_next=None,
        s_noise=None,
        noise_sampler=None,
        final=True,
    ):
        self.x = x
        self.sampler = sampler
        self.strength = strength if strength is not None else ss.sigma_up
        self.s_noise = s_noise if s_noise is not None else sampler.s_noise
        self.sigma = sigma if sigma is not None else ss.sigma
        self.sigma_next = sigma_next if sigma_next is not None else ss.sigma_next
        self.noise_sampler = noise_sampler if noise_sampler else sampler.noise_sampler
        self.final = final

    def get_noise(self, scaled=True):
        return self.noise_sampler(
            self.sigma, self.sigma_next, out_hw=self.x.shape[-2:]
        ).mul_(self.noise_scale if scaled else 1.0)

    @property
    def noise_scale(self):
        return self.strength * self.s_noise

    def noise_x(self, x=None, scale=1.0):
        if x is None:
            x = self.x
        else:
            self.x = x
        if self.sigma_next == 0 or self.noise_scale == 0:
            return x
        self.x = x + self.get_noise() * scale
        return self.x


class SingleStepSampler:
    name = None
    self_noise = 0
    model_calls = 0

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
        **kwargs,
    ):
        self.s_noise = s_noise
        self.eta = eta
        self.dyn_eta_start = dyn_eta_start
        self.dyn_eta_end = dyn_eta_end
        self.noise_sampler = noise_sampler
        self.weight = weight
        self.substeps = substeps
        self.options = kwargs

    def step(self, x, ss):
        raise NotImplementedError

    # Euler - based on original ComfyUI implementation
    def euler_step(self, x, ss):
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        d = to_d(x, ss.sigma, ss.denoised)
        dt = sigma_down - ss.sigma
        yield SamplerResult(ss, self, x + d * dt, sigma_up)
        # return x + d * dt, sigma_up

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


class ReversibleSingleStepSampler(SingleStepSampler):
    def __init__(self, *, reta=1.0, dyn_reta_start=None, dyn_reta_end=None, **kwargs):
        super().__init__(**kwargs)
        self.reta = reta
        self.dyn_reta_start = dyn_reta_start
        self.dyn_reta_end = dyn_reta_end

    def get_dyn_reta(self, ss):
        return self.reta * self.get_dyn_value(
            ss, self.dyn_reta_start, self.dyn_reta_end
        )


class EulerStep(SingleStepSampler):
    name = "euler"
    step = SingleStepSampler.euler_step


class DPMPPStepBase(SingleStepSampler):
    @staticmethod
    def sigma_fn(t):
        return t.neg().exp()

    @staticmethod
    def t_fn(t):
        return t.log().neg()


class DPMPP2MStep(DPMPPStepBase):
    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
        t, t_next = self.t_fn(ss.sigma), self.t_fn(ss.sigma_next)
        h = t_next - t
        st, st_next = self.sigma_fn(t), self.sigma_fn(t_next)
        if len(ss.dhist) == 0 or ss.sigma_prev is None:
            return (
                yield SamplerResult(
                    ss,
                    self,
                    (st_next / st) * x - (-h).expm1() * ss.denoised,
                    ss.sigma.new_zeros(1),
                )
            )
        h_last = t - self.t_fn(ss.sigma_prev)
        r = h_last / h
        denoised, old_denoised = ss.denoised, ss.dhist[-1]
        denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised

        yield SamplerResult(
            ss, self, (st_next / st) * x - (-h).expm1() * denoised_d, 0.0
        )


class DPMPP2MSDEStep(SingleStepSampler):
    name = "dpmpp_2m_sde"

    def __init__(self, *, solver_type="midpoint", **kwargs):
        super().__init__(**kwargs)
        self.solver_type = solver_type

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
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
        if len(ss.dhist) == 0 or ss.sigma_prev is None:
            return (yield SamplerResult(ss, self, x, noise_strength))
        h_last = (-ss.sigma.log()) - (-ss.sigma_prev.log())
        r = h_last / h
        old_denoised = ss.dhist[-1]
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
        yield SamplerResult(ss, self, x, noise_strength)


class DPMPP3MSDEStep(SingleStepSampler):
    name = "dpmpp_3m_sde"

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
        denoised = ss.denoised
        # if ss.sigma_next == 0:
        #     return denoised, 0
        t, s = -ss.sigma.log(), -ss.sigma_next.log()
        h = s - t
        eta = self.get_dyn_eta(ss)
        h_eta = h * (eta + 1)

        x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised
        noise_strength = ss.sigma_next * (-2 * h * eta).expm1().neg().sqrt()
        if len(ss.dhist) == 0 or ss.sigma_prev is None:
            return (yield SamplerResult(ss, self, x, noise_strength))
        h_1 = (-ss.sigma.log()) - (-ss.sigma_prev.log())
        denoised_1 = ss.dhist[-1]
        if len(ss.dhist) == 1:
            r = h_1 / h
            d = (denoised - denoised_1) / r
            phi_2 = h_eta.neg().expm1() / h_eta + 1
            x = x + phi_2 * d
        else:
            h_2 = (-ss.sigma_prev.log()) - (-ss.sigmas[ss.idx - 2].log())
            denoised_2 = ss.dhist[-2]
            r0 = h_1 / h
            r1 = h_2 / h
            d1_0 = (denoised - denoised_1) / r0
            d1_1 = (denoised_1 - denoised_2) / r1
            d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
            d2 = (d1_0 - d1_1) / (r0 + r1)
            phi_2 = h_eta.neg().expm1() / h_eta + 1
            phi_3 = phi_2 / h_eta - 0.5
            x = x + phi_2 * d1 - phi_3 * d2
        yield SamplerResult(ss, self, x, noise_strength)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class ReversibleHeunStep(ReversibleSingleStepSampler):
    name = "reversible_heun"
    model_calls = 1

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        sigma_down_reversible, _sigma_up_reversible = ss.get_ancestral_step(
            self.get_dyn_reta(ss)
        )
        dt = sigma_down - ss.sigma
        dt_reversible = sigma_down_reversible - ss.sigma

        # Calculate the derivative using the model
        d = to_d(x, ss.sigma, ss.denoised)

        # Predict the sample at the next sigma using Euler step
        x_pred = x + d * dt

        # Denoised sample at the next sigma
        denoised_next = ss.model(x_pred, sigma_down, model_call_idx=1)

        # Calculate the derivative at the next sigma
        d_next = to_d(x_pred, sigma_down, denoised_next)

        # Update the sample using the Reversible Heun formula
        x = x + dt * (d + d_next) / 2 - dt_reversible**2 * (d_next - d) / 4
        yield SamplerResult(ss, self, x, sigma_up)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class ReversibleHeun1SStep(ReversibleSingleStepSampler):
    name = "reversible_heun_1s"
    model_calls = 1

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
        # Reversible Heun-inspired update (first-order)
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        sigma_down_reversible, _sigma_up_reversible = ss.get_ancestral_step(
            self.get_dyn_reta(ss)
        )
        sigma_i, sigma_i_plus_1 = ss.sigma, sigma_down
        dt = sigma_i_plus_1 - sigma_i
        dt_reversible = sigma_down_reversible - sigma_i

        eff_x = ss.xhist[-1] if len(ss.xhist) else x

        # Calculate the derivative using the model
        d_i_old = to_d(
            eff_x,
            sigma_i,
            ss.dhist[-1]
            if len(ss.dhist)
            else ss.model(eff_x, sigma_i, model_call_idx=1),
        )

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
        yield SamplerResult(ss, self, x, sigma_up)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class RESStep(SingleStepSampler):
    name = "res"
    model_calls = 1

    def __init__(self, *, res_simple_phi=False, res_c2=0.5, **kwargs):
        super().__init__(**kwargs)
        self.simple_phi = res_simple_phi
        self.c2 = res_c2
        pass

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
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

        denoised2 = ss.model(x_2, sigma_2, model_call_idx=1)

        x = math.exp(-h) * x + h * (b1 * denoised + b2 * denoised2)
        yield SamplerResult(ss, self, x, sigma_up)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class TrapezoidalStep(SingleStepSampler):
    name = "trapezoidal"
    model_calls = 1

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        dt = ss.sigma_next - ss.sigma
        denoised = ss.denoised

        # Calculate the derivative using the model
        d_i = to_d(x, ss.sigma, denoised)

        # Predict the sample at the next sigma using Euler step
        x_pred = x + d_i * dt

        # Denoised sample at the next sigma
        denoised_next = ss.model(x_pred, ss.sigma_next, model_call_idx=1)

        # Calculate the derivative at the next sigma
        d_next = to_d(x_pred, ss.sigma_next, denoised_next)

        dt_2 = sigma_down - ss.sigma
        # Update the sample using the Trapezoidal rule
        x = x + dt_2 * (d_i + d_next) / 2
        yield SamplerResult(ss, self, x, sigma_up)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class BogackiStep(ReversibleSingleStepSampler):
    name = "bogacki"
    reversible = False
    model_calls = 2

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        sigma_down_reversible, _sigma_up_reversible = ss.get_ancestral_step(
            self.get_dyn_reta(ss)
        )
        sigma, sigma_next = ss.sigma, sigma_down
        dt = sigma_next - sigma
        dt_reversible = sigma_down_reversible - sigma
        denoised = ss.denoised

        # Calculate the derivative using the model
        d = to_d(x, sigma, denoised)

        # Bogacki-Shampine steps
        k1 = d * dt
        k2 = (
            to_d(
                x + k1 / 2,
                sigma + dt / 2,
                ss.model(x + k1 / 2, sigma + dt / 2, model_call_idx=1),
            )
            * dt
        )
        k3 = (
            to_d(
                x + 3 * k1 / 4 + k2 / 4,
                sigma + 3 * dt / 4,
                ss.model(x + 3 * k1 / 4 + k2 / 4, sigma + 3 * dt / 4, model_call_idx=2),
            )
            * dt
        )

        # Reversible correction term (inspired by Reversible Heun)
        correction = dt_reversible**2 * (k3 - k2) / 6 if self.reversible else 0.0

        # Update the sample
        x = x + 2 * k1 / 9 + k2 / 3 + 4 * k3 / 9 - correction
        yield SamplerResult(ss, self, x, sigma_up)


class ReversibleBogackiStep(BogackiStep):
    name = "reversible_bogacki"
    reversible = True


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class RK4Step(SingleStepSampler):
    name = "rk4"
    model_calls = 3

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        sigma = ss.sigma
        # Calculate the derivative using the model
        d = to_d(x, sigma, ss.denoised)
        dt = sigma_down - sigma

        # Runge-Kutta steps
        k1 = d * dt
        k2 = (
            to_d(
                x + k1 / 2,
                sigma + dt / 2,
                ss.model(x + k1 / 2, sigma + dt / 2, model_call_idx=1),
            )
            * dt
        )
        k3 = (
            to_d(
                x + k2 / 2,
                sigma + dt / 2,
                ss.model(x + k2 / 2, sigma + dt / 2, model_call_idx=2),
            )
            * dt
        )
        k4 = (
            to_d(
                x + k3,
                sigma + dt,
                ss.model(x + k3, sigma + dt, model_call_idx=3),
            )
            * dt
        )

        # Update the sample
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        yield SamplerResult(ss, self, x, sigma_up)


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
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
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
            return (yield SamplerResult(ss, self, x, sigma_up))
        noise_strength = self.ds_noise * sigma_up
        if noise_strength != 0:
            x = yield SamplerResult(
                ss, self, x, sigma_up, sigma_next=sigma_leap, final=False
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
        yield SamplerResult(ss, self, x, sigma_up2)

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


class DPMPP2SStep(DPMPPStepBase):
    name = "dpmpp_2s"
    model_calls = 1

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
        t_fn, sigma_fn = self.t_fn, self.sigma_fn
        sigma_down, sigma_up = ss.get_ancestral_step(self.get_dyn_eta(ss))
        # DPM-Solver++(2S)
        t, t_next = t_fn(ss.sigma), t_fn(sigma_down)
        r = 1 / 2
        h = t_next - t
        s = t + r * h
        x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * ss.denoised
        denoised_2 = ss.model(x_2, sigma_fn(s), model_call_idx=0)
        x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        yield SamplerResult(ss, self, x, sigma_up)


class DPMPPSDEStep(DPMPPStepBase):
    name = "dpmpp_sde"
    self_noise = 1
    model_calls = 1

    def __init__(self, *args, r=1 / 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield from self.euler_step(x, ss))
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
        x_2 = yield SamplerResult(
            ss, self, x_2, su, sigma=sigma_fn(t), sigma_next=sigma_fn(s), final=False
        )
        denoised_2 = ss.model(x_2, sigma_fn(s), model_call_idx=1)

        # Step 2
        sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
        t_next_ = t_fn(sd)
        denoised_d = (1 - fac) * ss.denoised + fac * denoised_2
        x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
        yield SamplerResult(ss, self, x, su)


# Based on implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
# Which was originally written by Katherine Crowson
class TTMJVPStep(SingleStepSampler):
    name = "ttm_jvp"
    model_calls = 1

    def __init__(self, *args, alternate_phi_2_calc=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.alternate_phi_2_calc = alternate_phi_2_calc

    def step(self, x, ss):
        if ss.sigma_next == 0:
            return (yield SamplerResult(ss, self, ss.denoised, ss.sigma.new_zeros(1)))
        eta = self.get_dyn_eta(ss)
        sigma, sigma_next = ss.sigma, ss.sigma_next
        # 2nd order truncated Taylor method
        t, s = -sigma.log(), -sigma_next.log()
        h = s - t
        h_eta = h * (eta + 1)

        eps = to_d(x, sigma, ss.denoised)
        _denoised, denoised_prime = ss.model(
            x, sigma, tangents=(eps * -sigma, -sigma), model_call_idx=1
        )

        phi_1 = -torch.expm1(-h_eta)
        if self.alternate_phi_2_calc:
            phi_2 = torch.expm1(-h) + h  # seems to work better with eta > 0
        else:
            phi_2 = torch.expm1(-h_eta) + h_eta
        x = torch.exp(-h_eta) * x + phi_1 * ss.denoised + phi_2 * denoised_prime

        if not eta:
            return (yield SamplerResult(ss, self, x, ss.sigma.new_zeros(1)))

        phi_1_noise = torch.sqrt(-torch.expm1(-2 * h * eta))
        yield SamplerResult(ss, self, x, sigma_next * phi_1_noise)


STEP_SAMPLERS = {
    "euler": EulerStep,
    "dpmpp_sde": DPMPPSDEStep,
    "dpmpp_2m": DPMPP2MStep,
    "dpmpp_2m_sde": DPMPP2MSDEStep,
    "dpmpp_3m_sde": DPMPP3MSDEStep,
    "dpmpp_2s": DPMPP2SStep,
    "reversible_heun": ReversibleHeunStep,
    "reversible_heun_1s": ReversibleHeun1SStep,
    "res": RESStep,
    "trapezoidal": TrapezoidalStep,
    "bogacki": BogackiStep,
    "reversible_bogacki": ReversibleBogackiStep,
    "rk4": RK4Step,
    "euler_dancing": EulerDancingStep,
    "ttm_jvp": TTMJVPStep,
}

__all__ = (
    "STEP_SAMPLERS",
    "EulerStep",
    "DPMPP2MStep",
    "DPMPP2MSDEStep",
    "DPMPP3MSDEStep",
    "DPMPP2SStep",
    "ReversibleHeunStep",
    "ReversibleHeun1SStep",
    "RESStep",
    "TrapezoidalStep",
    "BogackiStep",
    "ReversibleBogackiStep",
    "EulerDancingStep",
    "TTMJVPStep",
)
