import torch

from comfy.k_diffusion.sampling import to_d


def scale_noise(
    noise,
    factor=1.0,
    *,
    normalized=True,
    normalize_dims=(-3, -2, -1),
):
    if not normalized or noise.numel() == 0:
        return noise.mul_(factor) if factor != 1 else noise
    mean, std = (
        noise.mean(dim=normalize_dims, keepdim=True),
        noise.std(dim=normalize_dims, keepdim=True),
    )
    return noise.sub_(mean).div_(std).mul_(factor)


def find_first_unsorted(tensor, desc=True):
    if not (len(tensor.shape) and tensor.shape[0]):
        return None
    fun = torch.gt if desc else torch.lt
    first_unsorted = fun(tensor[1:], tensor[:-1]).nonzero().flatten()[:1].add_(1)
    return None if not len(first_unsorted) else first_unsorted.item()


def fallback(val, default, exclude=None):
    return val if val is not exclude else default


# From Gaeros. Thanks!
def extract_pred(x_before, x_after, sigma_before, sigma_after):
    if sigma_after == 0:
        return x_after, torch.zeros_like(x_after)
    alpha = sigma_after / sigma_before
    denoised = (x_after - alpha * x_before) / (1 - alpha)
    return denoised, to_d(x_after, sigma_after, denoised)


class Restart:
    def __init__(self, *, s_noise=1.0, custom_noise=None, immiscible=False):
        from .noise import ImmiscibleNoise

        self.s_noise = s_noise
        if immiscible is not False:
            immiscible = ImmiscibleNoise(**immiscible)
        self.immiscible = immiscible
        self.custom_noise = custom_noise

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

    def split_sigmas(self, sigmas):
        prev_seg = None
        while len(sigmas) > 1:
            seg = self.get_segment(sigmas)
            sigmas = sigmas[len(seg) :]
            if prev_seg is not None and seg[0] > prev_seg[-1]:
                noise_scale = self.get_noise_scale(prev_seg[-1], seg[0])
            else:
                noise_scale = 0.0
            prev_seg = seg
            yield (noise_scale, seg)

    def get_noise_scale(self, s_min, s_max):
        result = (s_max**2 - s_min**2) ** 0.5
        if isinstance(result, torch.Tensor):
            result = result.item()
        return result * self.s_noise

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
            if sig_idx >= siglen or sig_idx < 0:
                break
            interval, jump = item
            chunk = siglist[sig_idx : sig_idx + interval + 1]
            # print(f"{out}  +  {chunk}")
            out += chunk
            sig_idx += interval + jump
            if jump >= 0:
                sig_idx += 1
            sched_idx += 1
        if sig_idx < siglen and sig_idx >= 0:
            out += siglist[sig_idx:]
        if out[-1] > siglist[-1]:
            out.append(siglist[-1])
        return torch.tensor(out).to(sigmas)
