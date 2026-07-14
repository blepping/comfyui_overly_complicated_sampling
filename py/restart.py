from __future__ import annotations

from typing import NamedTuple

import torch

# from tqdm import tqdm


class RestartScaleFactors(NamedTuple):
    latent_scale: float
    noise_scale: float

    @classmethod
    def build(
        cls,
        sigma_from: float | torch.Tensor,
        sigma_to: float | torch.Tensor,
        *,
        is_flow: bool,
    ) -> RestartScaleFactors:
        if isinstance(sigma_from, torch.Tensor):
            sigma_from = sigma_from.max().item()
        if isinstance(sigma_to, torch.Tensor):
            sigma_to = sigma_to.max().item()
        if not is_flow:
            return cls(
                1.0,
                max(0.0, (sigma_to**2 - sigma_from**2)) ** 0.5,
            )
        alpha_from = 1.0 - sigma_from
        alpha_to = 1.0 - sigma_to
        if alpha_to <= 0:
            latent_scale = 0.0
            noise_scale = sigma_to
        else:
            latent_scale = alpha_to / alpha_from
            noise_scale = (
                max(0.0, (sigma_to**2) - (latent_scale * sigma_from) ** 2) ** 0.5
            )
        return cls(latent_scale, noise_scale)


class Restart:
    def __init__(
        self,
        *,
        s_noise=1.0,
        custom_noise=None,
        immiscible=False,
        normalized=True,
        normalize_dims: tuple[int, ...] | None = None,
        is_flow=False,
    ):
        from .noise import ImmiscibleNoise

        self.s_noise = s_noise
        if immiscible is not False:
            immiscible = ImmiscibleNoise(**immiscible)
        self.immiscible = immiscible
        self.custom_noise = custom_noise
        self.normalized = normalized
        self.normalize_dims = normalize_dims
        self.is_flow = is_flow

    def get_noise_sampler(self, nsc):
        return nsc.make_caching_noise_sampler(
            self.custom_noise,
            1,
            nsc.max_sigma,
            nsc.min_sigma,
            immiscible=self.immiscible,
        )

    @staticmethod
    def get_segment(sigmas: torch.Tensor) -> torch.Tensor:
        last_sigma = sigmas[0]
        for idx in range(1, len(sigmas)):
            sigma = sigmas[idx]
            if sigma > last_sigma:
                return sigmas[:idx]
            last_sigma = sigma
        return sigmas

    def split_sigmas(self, sigmas: torch.Tensor):
        prev_seg = None
        while len(sigmas) > 1:
            seg = self.get_segment(sigmas)
            sigmas = sigmas[len(seg) :]
            if prev_seg is not None and seg[0] > prev_seg[-1]:
                scale_factors = RestartScaleFactors.build(
                    sigma_from=prev_seg[-1], sigma_to=seg[0], is_flow=self.is_flow
                )
            else:
                scale_factors = None
            prev_seg = seg
            yield (scale_factors, seg)

    def get_noise_scale(
        self, s_min: float | torch.Tensor, s_max: float | torch.Tensor
    ) -> float:
        result = (s_max**2 - s_min**2) ** 0.5
        if isinstance(result, torch.Tensor):
            return result.item()
        return result

    def add_noise(
        self,
        x: torch.Tensor,
        sigma_from: float,
        sigma_to: float,
        *,
        nsc,
        refs,
        scale_factors: RestartScaleFactors | None = None,
        in_place: bool = False,
    ) -> torch.Tensor:
        if self.is_flow:
            sigma_from = min(1.0, max(0.0, sigma_from))
            sigma_to = min(1.0, max(0.0, sigma_to))
        if sigma_from >= sigma_to:
            raise ValueError(
                f"sigma_from ({sigma_from:.4f}) must be less than sigma_to ({sigma_to:.4f})"
            )
        scale_factors = scale_factors or RestartScaleFactors.build(
            sigma_from, sigma_to, is_flow=self.is_flow
        )
        ns = self.get_noise_sampler(nsc)
        sigma_empty = nsc.min_sigma * 0
        noise = nsc.scale_noise(
            ns(sigma_empty + sigma_from, sigma_empty + sigma_to, refs=refs),
            normalized=self.normalized,
            normalize_dims=self.normalize_dims,
        )
        noise *= scale_factors.noise_scale * self.s_noise
        if scale_factors.latent_scale != 1.0:
            x = (
                x.mul_(scale_factors.latent_scale)
                if in_place
                else scale_factors.latent_scale * x
            )
        return noise.add_(x)

    def __repr__(self):
        return f"<Restart: s_noise={self.s_noise:.04}, immiscible={self.immiscible}>"

    @classmethod
    def simple_schedule(cls, sigmas, start_step, schedule=(), max_iter=1000):
        if sigmas.ndim != 1:
            raise ValueError("Bad number of dimensions for sigmas")
        siglen = len(sigmas) - 1
        if siglen <= start_step or not len(schedule):
            return sigmas
        siglist = sigmas.cpu().tolist()
        out = siglist[:start_step]
        sched_len = len(schedule)
        sched_idx = 0
        sig_idx = start_step
        iter_count = 0
        while 0 <= sched_idx < sched_len:
            # print(f"LOOP: sched_idx={sched_idx}, sig_idx={sig_idx}: {out}")
            iter_count += 1
            if iter_count > max_iter:
                raise RuntimeError("Hit max iteration count. Loop in schedule?")
            item = schedule[sched_idx]
            if not isinstance(item, (list, tuple)):
                if item < 0:
                    item = sched_len + item
                if item < 0 or item >= sched_len:
                    raise ValueError("Schedule jump index out of range")
                sched_idx = item
                continue
            sched_frac = round(sig_idx - int(sig_idx), ndigits=5)
            sig_idx = int(sig_idx if sched_frac == 0 else sig_idx + 1)
            if sig_idx >= siglen or sig_idx < 0:
                break
            interval, jump = item
            chunk = siglist[sig_idx : sig_idx + interval + 1]
            if sched_frac != 0:
                chunk[0] -= (chunk[0] - chunk[1]) * (1.0 - sched_frac)
            # print(f"{out}  +  {chunk}")
            out += chunk
            if jump >= 0:
                sig_idx += 1
            sig_idx += interval + jump
            sched_idx += 1
        sched_frac = round(sig_idx - int(sig_idx), ndigits=5)
        sig_idx = int(sig_idx if sched_frac == 0 else sig_idx + 1)
        if sig_idx < siglen and sig_idx >= 0:
            chunk = siglist[sig_idx:]
            if sched_frac != 0:
                chunk[0] -= (chunk[0] - chunk[1]) * (1.0 - sched_frac)
            out += chunk
        if out[-1] > siglist[-1]:
            out.append(siglist[-1])
        return torch.tensor(out).to(sigmas)
