import math

import torch

from comfy.k_diffusion.sampling import (
    get_ancestral_step,
    to_d,
)

from .res_support import _de_second_order
from .utils import find_first_unsorted


class SingleStepSampler:
    name = None

    def __init__(
        self, *, noise_sampler=None, s_noise=1.0, eta=1.0, weight=1.0, **kwargs
    ):
        self.s_noise = s_noise
        self.eta = eta
        self.noise_sampler = noise_sampler
        self.weight = weight
        self.kwargs = kwargs

    def step(self, x, ss):
        raise NotImplementedError

    # Euler - based on original ComfyUI implementation
    def final_step(self, x, ss):
        sigma_down, sigma_up = ss.get_ancestral_step(self.eta)
        d = to_d(x, ss.sigma, ss.denoised)
        dt = sigma_down - ss.sigma
        return x + d * dt, sigma_up

    def __str__(self):
        return f"<SS({self.name}): s_noise={self.s_noise}, eta={self.eta}, kwargs={self.kwargs}>"


class ReversibleSingleStepSampler(SingleStepSampler):
    def __init__(self, *, reta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.reta = reta


class EulerStep(SingleStepSampler):
    name = "euler"

    def step(self, x, ss):
        return self.final_step(x, ss)


class DPMPP2MStep(SingleStepSampler):
    @staticmethod
    def sigma_fn(t):
        return t.neg().exp()

    @staticmethod
    def t_fn(t):
        return t.log().neg()

    def step(self, x, ss):
        t, t_next = self.t_fn(ss.sigma), self.t_fn(ss.sigma_next)
        h = t_next - t
        st, st_next = self.sigma_fn(t), self.sigma_fn(t_next)
        if len(ss.dhist) == 0 or ss.sigma_prev is None:
            return (st_next / st) * x - (-h).expm1() * ss.denoised, 0.0
        h_last = t - self.t_fn(ss.sigma_prev)
        r = h_last / h
        denoised, old_denoised = ss.denoised, ss.dhist[-1]
        denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
        return (st_next / st) * x - (-h).expm1() * denoised_d, 0.0


class DPMPP2MSDEStep(SingleStepSampler):
    name = "dpmpp_2m_sde"

    def __init__(self, *, solver_type="midpoint", **kwargs):
        super().__init__(**kwargs)
        self.solver_type = solver_type

    def step(self, x, ss):
        denoised = ss.denoised
        if ss.sigma_next == 0:
            return denoised, None
        # DPM-Solver++(2M) SDE
        t, s = -ss.sigma.log(), -ss.sigma_next.log()
        h = s - t
        eta_h = self.eta * h

        x = (
            ss.sigma_next / ss.sigma * (-eta_h).exp() * x
            + (-h - eta_h).expm1().neg() * denoised
        )
        noise_strength = ss.sigma_next * (-2 * eta_h).expm1().neg().sqrt()
        if len(ss.dhist) == 0 or ss.sigma_prev is None:
            return x, noise_strength
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
        return x, noise_strength


class DPMPP3MSDEStep(SingleStepSampler):
    name = "dpmpp_3m_sde"

    def step(self, x, ss):
        denoised = ss.denoised
        if ss.sigma_next == 0:
            return denoised, 0
        t, s = -ss.sigma.log(), -ss.sigma_next.log()
        h = s - t
        h_eta = h * (self.eta + 1)

        x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised
        noise_strength = ss.sigma_next * (-2 * h * self.eta).expm1().neg().sqrt()
        if len(ss.dhist) == 0 or ss.sigma_prev is None:
            return x, noise_strength
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
        return x, noise_strength


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class ReversibleHeunStep(ReversibleSingleStepSampler):
    name = "reversible_heun"

    def step(self, x, ss):
        sigma_down, sigma_up = ss.get_ancestral_step(self.eta)
        sigma_down_reversible, sigma_up_reversible = ss.get_ancestral_step(self.reta)
        dt = sigma_down - ss.sigma
        dt_reversible = sigma_down_reversible - ss.sigma

        # Calculate the derivative using the model
        d = to_d(x, ss.sigma, ss.denoised)

        # Predict the sample at the next sigma using Euler step
        x_pred = x + d * dt

        # Denoised sample at the next sigma
        denoised_next = ss.model(x_pred, sigma_down, model_call_idx=0)

        # Calculate the derivative at the next sigma
        d_next = to_d(x_pred, sigma_down, denoised_next)

        # Update the sample using the Reversible Heun formula
        x = x + dt * (d + d_next) / 2 - dt_reversible**2 * (d_next - d) / 4
        return x, sigma_up


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class ReversibleHeun1SStep(ReversibleSingleStepSampler):
    name = "reversible_heun_1s"

    def step(self, x, ss):
        # Reversible Heun-inspired update (first-order)
        sigma_down, sigma_up = ss.get_ancestral_step(self.eta)
        sigma_down_reversible, sigma_up_reversible = ss.get_ancestral_step(self.reta)
        sigma_i, sigma_i_plus_1 = ss.sigma, sigma_down
        dt = sigma_i_plus_1 - sigma_i
        dt_reversible = sigma_down_reversible - sigma_i

        eff_x = ss.xhist[-1] if len(ss.xhist) else x

        # Calculate the derivative using the model
        print("Can skip", len(ss.dhist))
        d_i_old = to_d(
            eff_x,
            sigma_i,
            ss.dhist[-1]
            if len(ss.dhist)
            else ss.model(eff_x, sigma_i, model_call_idx=0),
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
        return x, sigma_up


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class RESStep(SingleStepSampler):
    name = "res"

    def __init__(self, *, res_simple_phi=False, res_c2=0.5, **kwargs):
        super().__init__(**kwargs)
        self.simple_phi = res_simple_phi
        self.c2 = res_c2
        pass

    def step(self, x, ss):
        sigma_down, sigma_up = ss.get_ancestral_step(self.eta)
        denoised = ss.denoised
        lam_next = (
            sigma_down.log().neg() if self.eta != 0 else ss.sigma_next.log().neg()
        )
        lam = ss.sigma.log().neg()

        h = lam_next - lam
        a2_1, b1, b2 = _de_second_order(
            h=h, c2=self.c2, simple_phi_calc=self.simple_phi
        )

        c2_h = 0.5 * h

        x_2 = math.exp(-c2_h) * x + a2_1 * h * denoised
        lam_2 = lam + c2_h
        sigma_2 = lam_2.neg().exp()

        denoised2 = ss.model(x_2, sigma_2, model_call_idx=0)

        x = math.exp(-h) * x + h * (b1 * denoised + b2 * denoised2)
        return x, sigma_up


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class TrapezoidalStep(SingleStepSampler):
    name = "trapezoidal"

    def step(self, x, ss):
        sigma_down, sigma_up = ss.get_ancestral_step(self.eta)
        dt = ss.sigma_next - ss.sigma
        denoised = ss.denoised

        # Calculate the derivative using the model
        d_i = to_d(x, ss.sigma, denoised)

        # Predict the sample at the next sigma using Euler step
        x_pred = x + d_i * dt

        # Denoised sample at the next sigma
        denoised_next = ss.model(x_pred, ss.sigma_next, model_call_idx=0)

        # Calculate the derivative at the next sigma
        d_next = to_d(x_pred, ss.sigma_next, denoised_next)

        dt_2 = sigma_down - ss.sigma
        # Update the sample using the Trapezoidal rule
        x = x + dt_2 * (d_i + d_next) / 2
        return x, sigma_up


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class BogackiStep(ReversibleSingleStepSampler):
    name = "bogacki"
    reversible = False

    def step(self, x, ss):
        sigma_down, sigma_up = ss.get_ancestral_step(self.eta)
        sigma_down_reversible, sigma_up_reversible = ss.get_ancestral_step(self.reta)
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
                ss.model(x + k1 / 2, sigma + dt / 2, model_call_idx=0),
            )
            * dt
        )
        k3 = (
            to_d(
                x + 3 * k1 / 4 + k2 / 4,
                sigma + 3 * dt / 4,
                ss.model(x + 3 * k1 / 4 + k2 / 4, sigma + 3 * dt / 4, model_call_idx=1),
            )
            * dt
        )

        # Reversible correction term (inspired by Reversible Heun)
        correction = dt_reversible**2 * (k3 - k2) / 6 if self.reversible else 0.0

        # Update the sample
        x = x + 2 * k1 / 9 + k2 / 3 + 4 * k3 / 9 - correction
        return x, sigma_up


class ReversibleBogackiStep(BogackiStep):
    name = "reversible_bogacki"
    reversible = True


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class RK4Step(SingleStepSampler):
    name = "rk4"

    def step(self, x, ss):
        sigma_down, sigma_up = ss.get_ancestral_step(self.eta)
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
                ss.model(x + k1 / 2, sigma + dt / 2, model_call_idx=0),
            )
            * dt
        )
        k3 = (
            to_d(
                x + k2 / 2,
                sigma + dt / 2,
                ss.model(x + k2 / 2, sigma + dt / 2, model_call_idx=1),
            )
            * dt
        )
        k4 = (
            to_d(
                x + k3,
                sigma + dt,
                ss.model(x + k3, sigma + dt, model_call_idx=2),
            )
            * dt
        )

        # Update the sample
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x, sigma_up


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class EulerDancingStep(SingleStepSampler):
    name = "euler_dancing"

    def __init__(
        self,
        *,
        deta=1.0,
        ds_noise=1.0,
        leap=2,
        dyn_deta_start=None,
        dyn_deta_end=None,
        dyn_deta_mode="lerp",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.deta = deta
        self.ds_noise = ds_noise
        self.leap = leap
        self.dyn_deta_start = dyn_deta_start
        self.dyn_deta_end = dyn_deta_end
        if dyn_deta_mode not in ("lerp", "deta"):
            raise ValueError("Bad dyn_deta_mode")
        self.dyn_deta_mode = dyn_deta_mode

    def step(self, x, ss):
        leap_sigmas = ss.sigmas[ss.idx :]
        leap_sigmas = leap_sigmas[: find_first_unsorted(leap_sigmas)]
        zero_idx = (leap_sigmas <= 0).nonzero().flatten()[:1]
        max_leap = (zero_idx.item() if len(zero_idx) else len(leap_sigmas)) - 1
        is_danceable = max_leap > 1 and ss.sigma_next != 0
        curr_leap = max(1, min(self.leap, max_leap))
        sigma_leap = leap_sigmas[curr_leap] if is_danceable else ss.sigma_next
        print("DANCE", max_leap, curr_leap, sigma_leap, "--", leap_sigmas)
        del leap_sigmas
        sigma_down, sigma_up = get_ancestral_step(ss.sigma, sigma_leap, self.eta)
        d = to_d(x, ss.sigma, ss.denoised)
        # Euler method
        dt = sigma_down - ss.sigma
        x = x + d * dt
        if None not in (self.dyn_deta_start, self.dyn_deta_end):
            if self.dyn_deta_start == self.dyn_deta_end:
                dance_scale = self.dyn_deta_start
            else:
                main_idx = getattr(ss, "main_idx", ss.idx)
                main_sigmas = getattr(ss, "main_sigmas", ss.sigmas)
                step_pct = main_idx / (len(main_sigmas) - 1)
                dd_diff = self.dyn_deta_end - self.dyn_deta_start
                dance_scale = self.dyn_deta_start + dd_diff * step_pct
        else:
            dance_scale = 1.0
        print("DANCE?", dance_scale, ss.idx, is_danceable, curr_leap)
        if not is_danceable or abs(dance_scale) < 1e-04:
            return x, sigma_up
        orig_x = x
        x = x + self.noise_sampler(ss.sigma, sigma_leap) * self.s_noise * sigma_up
        sigma_down2, sigma_up2 = get_ancestral_step(
            sigma_leap,
            ss.sigma_next,
            eta=self.deta * (1.0 if self.dyn_deta_mode == "lerp" else dance_scale),
        )
        d_2 = to_d(x, sigma_leap, ss.denoised)
        dt_2 = sigma_down2 - sigma_leap
        result = x + d_2 * dt_2
        if self.dyn_deta_mode == "deta" or dance_scale == 1.0:
            return result, sigma_up2
        result = torch.lerp(orig_x, result, dance_scale)
        # FIXME: Broken for noise samplers that care about s/sn
        return result, torch.lerp(sigma_up, sigma_up2, dance_scale)


STEP_SAMPLERS = {
    "euler": EulerStep,
    "dpmpp_2m": DPMPP2MStep,
    "dpmpp_2m_sde": DPMPP2MSDEStep,
    "dpmpp_3m_sde": DPMPP3MSDEStep,
    "reversible_heun": ReversibleHeunStep,
    "reversible_heun_1s": ReversibleHeun1SStep,
    "res": RESStep,
    "trapezoidal": TrapezoidalStep,
    "bogacki": BogackiStep,
    "reversible_bogacki": ReversibleBogackiStep,
    "rk4": RK4Step,
    "euler_dancing": EulerDancingStep,
}

__all__ = (
    "STEP_SAMPLERS",
    "EulerStep",
    "DPMPP2MStep",
    "DPMPP2MSDEStep",
    "DPMPP3MSDEStep",
    "ReversibleHeunStep",
    "ReversibleHeun1SStep",
    "RESStep",
    "TrapezoidalStep",
    "BogackiStep",
    "ReversibleBogackiStep",
    "EulerDancingStep",
)