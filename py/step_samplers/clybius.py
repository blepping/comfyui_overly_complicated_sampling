# Samplers based on Clybius' designs, mostly from https://github.com/Clybius/ComfyUI-Extra-Samplers/

import math
import torch

from comfy.k_diffusion.sampling import get_ancestral_step, to_d

from .base import SingleStepSampler, ReversibleConfig, ReversibleSingleStepSampler
from .builtins import DPMPP2MSDEStep
from . import res_support
from . import registry

from .. import filtering
from .. import utils


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
        blend_mode = self.options.get("blend_mode", "lerp").strip()
        self.blend = (
            filtering.BLENDING_MODES[blend_mode] if blend_mode != "lerp" else torch.lerp
        )

    def reversible_correction(self, d, d_next, dt_reversible):
        if dt_reversible == 0 or self.reversible.scale == 0:
            return None
        return d_next.sub_(d).div_(4).mul_(dt_reversible**2).mul_(self.reversible.scale)

    def step_internal(self, x, *, history_mode=False):
        ss = self.ss
        history_mode = history_mode and self.available_history() > 0
        if not history_mode:
            mr_1, mr_2 = ss.hcur, None
        else:
            mr_1, mr_2 = ss.hprev, ss.hcur
        sigma = ss.sigma
        denoised, uncond = mr_1.denoised, mr_1.denoised_uncond
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        ratio = sigma_down / sigma
        dratio = 1 - (sigma / sigma_down) * 0.5
        if mr_2 is None:
            x_2 = self.step_mix(x, denoised, uncond, ratio, blend=self.blend)
            mr_2 = self.call_model(x_2, sigma_down, call_index=1)
            del x_2
        denoised_2, uncond_2 = mr_2.denoised, mr_2.denoised_uncond
        denoised_prime = self.blend(denoised_2, denoised, dratio)
        if self.cfgpp:
            denoised_prime += denoised * 0.5
            uncond_prime = (uncond_2 * (1 - dratio)).add_(uncond)
        elif self.alt_cfgpp_scale != 0:
            uncond_prime = self.blend(uncond_2, uncond, dratio)
        else:
            uncond_prime = uncond
        x = self.step_mix(x, denoised_prime, uncond_prime, ratio, blend=self.blend)
        if self.reversible.scale != 0:
            correction = self.reversible_correction(
                d=self.to_d(mr_1, use_cfgpp=self.reversible.use_cfgpp),
                d_next=self.to_d(mr_2, use_cfgpp=self.reversible.use_cfgpp),
                dt_reversible=self.get_ancestral_step(self.dyn_reta)[0] - sigma,
            )
            if correction is not None:
                x -= correction
        yield from self.result(x, sigma_up)

    def step(self, x):
        return self.step_internal(x, history_mode=False)


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class ReversibleHeun1SStep(ReversibleHeunStep):
    name = "reversible_heun_1s"
    model_calls = (0, 1)
    default_history_limit, max_history = 1, 1
    allow_alt_cfgpp = True
    allow_cfgpp = True

    def step(self, x):
        return self.step_internal(x, history_mode=True)


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


# Based on original implementation from https://github.com/Clybius/ComfyUI-Extra-Samplers
class BogackiStep(ReversibleSingleStepSampler):
    name = "bogacki"
    reversible = False
    model_calls = 2
    allow_alt_cfgpp = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.reversible:
            self.reversible.scale = 0

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
    model_calls = (0, 3)
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
    name = "clybius_euler_dancing"
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
        return (dtr**2 * (d_to - d_from) / 4) * self.reversible.scale

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
    model_calls = (0, 1)
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


class ClybiusSENSStep(DPMPP2MSDEStep):
    name = "clybius_sens"
    default_history_limit, max_history = 2, 2
    allow_alt_cfgpp = False
    default_reta = 1.0
    default_reversible_scale = 1.0

    def __init__(self, *, tsde_reversible=None, **kwargs):
        super().__init__(**kwargs)
        self.tsde_reversible = ReversibleConfig.build(
            default_eta=self.default_reta,
            default_scale=self.default_reversible_scale,
            **utils.fallback(tsde_reversible, {}),
        )

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
        if self.available_history() > 1:
            tsde_reta, tsde_reversible_scale = self.get_reversible_cfg(
                reversible=self.tsde_reversible
            )
            tsde_reta_h = tsde_reta * h
            sigma_prev_2 = ss.hist[-3].sigma
            h_last_2 = (-sigma_prev.log()) - (-sigma_prev_2.log())
            r = h_last_2 / h
            old_denoised_2 = ss.hist[-3].denoised
            d = (old_denoised - old_denoised_2).div_(r)
            d_2 = (old_denoised - denoised).div_(r)

            d_rev = (denoised - old_denoised).div_(r)
            d_2_rev = (old_denoised_2 - old_denoised).div_(r)

            rphi = tsde_reta_h.neg().expm1() / tsde_reta_h + 1
            tsde_adjustment = rphi * (d + d_2) / 2
            if tsde_reversible_scale != 0:
                tsde_adjustment -= (rphi**2 * (d_rev + d_2_rev) / 2).mul_(
                    tsde_reversible_scale
                )
            x += tsde_adjustment
        yield from self.result(x, noise_strength)


registry.add(
    BogackiStep,
    ClybiusSENSStep,
    EulerDancingStep,
    ReversibleBogackiStep,
    ReversibleHeunStep,
    ReversibleHeun1SStep,
    RESStep,
    TTMJVPStep,
    TrapezoidalStep,
    RK4Step,
    RKDynamicStep,
    RKF45Step,
    Heun1SStep,
    HeunStep,
)
