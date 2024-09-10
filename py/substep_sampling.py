import torch

from comfy.k_diffusion.sampling import get_ancestral_step

from .filtering import FilterRefs
from .model import History
from .utils import fallback


class Items:
    def __init__(self, items=None):
        self.items = [] if items is None else items

    def clone(self):
        return self.__class__(items=self.items.copy())

    def append(self, item):
        self.items.append(item)
        return item

    def __getitem__(self, key):
        return self.items[key]

    def __setitem__(self, key, value):
        self.items[key] = value

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return self.items.__iter__()


class CommonOptionsItems(Items):
    def __init__(self, *, s_noise=1.0, eta=1.0, items=None, **kwargs):
        super().__init__(items=items)
        self.options = kwargs
        self.s_noise = s_noise
        self.eta = eta

    def clone(self):
        obj = super().clone()
        obj.options = self.options.copy()
        obj.s_noise = self.s_noise
        obj.eta = self.eta
        return obj


class StepSamplerChain(CommonOptionsItems):
    def __init__(
        self,
        *,
        merge_method="divide",
        time_mode="step",
        time_start=0,
        time_end=999,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.merge_method = merge_method
        if time_mode not in ("step", "step_pct", "sigma"):
            raise ValueError("Bad time mode")
        self.time_mode = time_mode
        self.time_start, self.time_end = time_start, time_end

    def clone(self):
        obj = super().clone()
        obj.merge_method = self.merge_method
        obj.time_mode = self.time_mode
        obj.time_start, obj.time_end = self.time_start, self.time_end
        obj.options = self.options.copy()
        return obj


class ParamGroup(Items):
    pass


class StepSamplerGroups(CommonOptionsItems):
    pass


class SamplerState:
    CLONE_KEYS = (
        "cfg_scale_override",
        "model",
        "hist",
        "extra_args",
        "disable_status",
        "eta",
        "reta",
        "s_noise",
        "sigmas",
        "callback_",
        "noise_sampler",
        "noise",
        "idx",
        "total_steps",
        "step",
        "substep",
        "sigma",
        "sigma_next",
        "sigma_prev",
        "sigma_down",
        "sigma_up",
        "refs",
    )

    def __init__(
        self,
        model,
        sigmas,
        idx,
        extra_args,
        *,
        step=0,
        substep=0,
        noise_sampler,
        callback=None,
        denoised=None,
        noise=None,
        eta=1.0,
        reta=1.0,
        s_noise=1.0,
        disable_status=False,
        history_size=4,
        cfg_scale_override=None,
    ):
        self.model = model
        self.hist = History(max(1, history_size))
        self.extra_args = extra_args
        self.eta = eta
        self.reta = reta
        self.s_noise = s_noise
        self.sigmas = sigmas
        self.callback_ = callback
        self.noise_sampler = noise_sampler
        self.noise = noise
        self.disable_status = disable_status
        self.step = 0
        self.substep = 0
        self.total_steps = len(sigmas) - 1
        self.cfg_scale_override = cfg_scale_override
        self.update(idx)  # Sets idx, sigma_prev, sigma, sigma_down, refs

    @property
    def hcur(self):
        return self.hist[-1]

    @property
    def hprev(self):
        return self.hist[-2]

    @property
    def denoised(self):
        return self.hcur.denoised

    @property
    def dt(self):
        return self.sigma_next - self.sigma

    @property
    def d(self):
        return self.hcur.d

    def update(self, idx=None, step=None, substep=None):
        idx = self.idx if idx is None else idx
        self.idx = idx
        self.sigma_prev = None if idx < 1 else self.sigmas[idx - 1]
        self.sigma, self.sigma_next = self.sigmas[idx], self.sigmas[idx + 1]
        self.sigma_down, self.sigma_up = get_ancestral_step(
            self.sigma, self.sigma_next, eta=self.eta
        )
        if step is not None:
            self.step = step
        if substep is not None:
            self.substep = substep
        self.refs = FilterRefs.from_ss(self)

    def get_ancestral_step(
        self, eta=1.0, sigma=None, sigma_next=None, retry_increment=0
    ):
        if self.model.is_rectified_flow:
            return self.get_ancestral_step_rf(
                eta=eta,
                sigma=sigma,
                sigma_next=sigma_next,
                retry_increment=retry_increment,
            )
        sigma = fallback(sigma, self.sigma)
        sigma_next = fallback(sigma_next, self.sigma_next)
        if eta <= 0 or sigma_next <= 0:
            return sigma_next, sigma_next.new_zeros(1)
        while eta > 0:
            sd, su = (
                v if isinstance(v, torch.Tensor) else sigma.new_full((1,), v)
                for v in get_ancestral_step(
                    sigma, sigma_next, eta=eta if sigma_next != 0 else 0
                )
            )
            if sd > 0 and su > 0:
                return sd, su
            if retry_increment <= 0:
                break
            # print(f"\nETA {eta} failed, retrying with {eta - retry_increment}")
            eta -= retry_increment
        return sigma_next, sigma_next.new_zeros(1)

    # Referenced from Comfy dpmpp_2s_ancestral_RF
    def get_ancestral_step_rf(
        self, eta=1.0, sigma=None, sigma_next=None, retry_increment=0
    ):
        sigma = fallback(sigma, self.sigma)
        sigma_next = fallback(sigma_next, self.sigma_next)
        if eta <= 0 or sigma_next <= 0:
            return sigma_next, sigma_next.new_zeros(1)
        while eta > 0:
            sigma_down = sigma_next * (1 + (sigma_next / sigma - 1) * eta)
            alpha_ip1, alpha_down = 1 - sigma_next, 1 - sigma_down
            sigma_up = (
                sigma_next**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2
            ) ** 0.5
            if sigma_down > 0 and sigma_up > 0:
                return sigma_down, sigma_up
            if retry_increment <= 0:
                break
            eta -= retry_increment
        return sigma_next, sigma_next.new_zeros(1)
        # print(f"\nRF ancestral: down={sigma_down}, up={sigma_up}")

    def clone_edit(self, **kwargs):
        obj = self.__class__.__new__(self.__class__)
        for k in self.CLONE_KEYS:
            setattr(obj, k, kwargs[k] if k in kwargs else getattr(self, k))
        obj.update()
        return obj

    def callback(self, hi=None, *, preview_mode="denoised"):
        if not self.callback_:
            return None
        hi = self.hcur if hi is None else hi
        if preview_mode == "cond":
            preview = fallback(hi.denoised_cond, hi.denoised)
        elif preview_mode == "uncond":
            preview = fallback(hi.denoised_uncond, hi.denoised)
        elif preview_mode == "raw":
            preview = hi.x
        elif (
            preview_mode == "diff"
            and hi.denoised_uncond is not None
            and hi.denoised_cond is not None
        ):
            preview = (
                hi.denoised_uncond * 0.25 + (hi.denoised_uncond - hi.denoised_cond) * 16
            )
        elif preview_mode == "noisy":
            preview = (hi.x - hi.denoised) * 0.1 + hi.denoised
        else:
            preview = hi.denoised
        return self.callback_({
            "x": hi.x,
            "i": self.step,
            "sigma": hi.sigma,
            "sigma_hat": hi.sigma,
            "denoised": preview,
        })

    def reset(self):
        self.hist.reset()
        self.denoised = None

    def call_model(self, *args, **kwargs):
        cfg_scale_override = kwargs.pop("cfg_scale_override", self.cfg_scale_override)
        return self.model(*args, cfg_scale_override=cfg_scale_override, **kwargs)
