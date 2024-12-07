# Samplers based on design from https://github.com/Extraltodeus/

import typing

import torch
import tqdm

from .base import SingleStepSampler
from . import registry


class DistanceConfig(typing.NamedTuple):
    resample: int = 3
    resample_end: int = 1
    eta: float = 0.0
    s_noise: float = 1.0
    alt_cfgpp_scale: float = 0.0
    first_eta_step: int = 0
    last_eta_step: int = -1
    custom_noise_name: str = "alt"
    immiscible: dict | bool | None = None


# Based on https://github.com/Extraltodeus/DistanceSampler
class DistanceStep(SingleStepSampler):
    name = "extraltodeus_distance"
    allow_alt_cfgpp = True
    model_calls = -1
    uses_alt_noise = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance = DistanceConfig(**self.options.get("distance", {}))

    @property
    def require_uncond(self):
        return super().require_uncond or self.distance.alt_cfgpp_scale != 0

    def distance_resample_steps(self):
        ss = self.ss
        resample, resample_end = self.distance.resample, self.distance.resample_end
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
        distance = self.distance
        ss = self.ss
        sigma_down, sigma_up = self.get_ancestral_step(self.get_dyn_eta())
        rsigma_down, rsigma_up = self.get_ancestral_step(eta=distance.eta)
        rsigma_up *= distance.s_noise
        sigma, sigma_next = ss.sigma, ss.sigma_next
        zero_up = sigma * 0
        d = self.to_d(ss.hcur)
        can_ancestral = not torch.equal(rsigma_down, sigma_next)
        start_eta_idx, end_eta_idx = (
            max(0, resample_steps + v if v < 0 else v)
            for v in (
                distance.first_eta_step,
                distance.last_eta_step,
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
                    noise_sampler=self.alt_noise_sampler,
                    final=False,
                )
            sr = self.call_model(x_new, sigma_next, call_index=re_step + 1)
            new_d = sr.to_d(
                sigma=curr_sigma_down, alt_cfgpp_scale=distance.alt_cfgpp_scale
            )
            x_n.append(new_d)
            if re_step == 0:
                d = (new_d + d) / 2
            else:
                d = self.distance_weights(torch.stack(x_n), re_step + 2)
                x_n.append(d)
        yield from self.result(x + d * dt, sigma_up, sigma_down=sigma_down)


registry.add(DistanceStep)
