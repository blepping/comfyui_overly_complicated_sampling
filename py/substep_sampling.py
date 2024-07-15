import torch

from comfy.k_diffusion.sampling import get_ancestral_step

from .model import History


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
        # step, step_pct, sigma
        self.merge_method = merge_method
        self.time_mode = time_mode
        self.time_start, self.time_end = time_start, time_end

    def check_time(self, sigma, step, steps):
        step_pct = step / steps if steps != 0 else 0.0
        if self.time_mode == "step":
            return self.time_start <= step <= self.time_end
        if self.time_mode == "step_pct":
            return self.time_start <= step_pct <= self.time_end
        if self.time_mode == "sigma":
            return self.time_start >= sigma >= self.time_end
        raise ValueError("Bad time mode")

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
    def find_match(self, sigma, step, steps):
        for idx, item in enumerate(self.items):
            if item.check_time(sigma, step, steps):
                return idx
        return None


class SamplerState:
    def __init__(
        self,
        model,
        sigmas,
        idx,
        extra_args,
        *,
        step=0,
        noise_sampler,
        callback=None,
        denoised=None,
        noise=None,
        eta=1.0,
        reta=1.0,
        s_noise=1.0,
        disable_status=False,
    ):
        self.model = model
        self.hist = History(4)
        self.extra_args = extra_args
        self.eta = eta
        self.reta = reta
        self.s_noise = s_noise
        self.sigmas = sigmas
        self.callback_ = callback
        self.noise_sampler = noise_sampler
        self.noise = noise
        self.disable_status = disable_status
        self.update(idx)

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

    def update(self, idx=None, step=None):
        idx = self.idx if idx is None else idx
        self.idx = idx
        self.sigma_prev = None if idx < 1 else self.sigmas[idx - 1]
        self.sigma, self.sigma_next = self.sigmas[idx], self.sigmas[idx + 1]
        self.sigma_down, self.sigma_up = get_ancestral_step(
            self.sigma, self.sigma_next, eta=self.eta
        )
        if step is not None:
            self.step = step

    def get_ancestral_step(self, eta=1.0, sigma=None, sigma_next=None):
        sigma = self.sigma if sigma is None else sigma
        sigma_next = self.sigma_next if sigma_next is None else sigma_next
        sd, su = (
            v if isinstance(v, torch.Tensor) else sigma.new_full((1,), v)
            for v in get_ancestral_step(
                sigma, sigma_next, eta=eta if sigma_next != 0 else 0
            )
        )
        return sd, su

    def clone_edit(self, **kwargs):
        obj = self.__class__.__new__(self.__class__)
        for k in (
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
            "step",
            "sigma",
            "sigma_next",
            "sigma_prev",
            "sigma_down",
            "sigma_up",
        ):
            setattr(obj, k, kwargs[k] if k in kwargs else getattr(self, k))
        obj.update()
        return obj

    def callback(self, hi=None):
        if not self.callback_:
            return None
        hi = self.hcur if hi is None else hi
        return self.callback_({
            "x": hi.x,
            "i": self.step,
            "sigma": hi.sigma,
            "sigma_hat": hi.sigma,
            "denoised": hi.denoised,
        })

    def reset(self):
        self.hist.reset()
        self.denoised = None
