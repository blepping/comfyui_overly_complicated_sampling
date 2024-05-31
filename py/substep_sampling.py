import torch

from comfy.k_diffusion.sampling import get_ancestral_step


class StepSamplerChain:
    def __init__(self, items=None):
        self.items = [] if items is None else items

    def clone(self):
        return self.__class__(items=self.items.copy())


class History:
    def __init__(self, x, size):
        self.history = torch.zeros(size, *x.shape, device=x.device, dtype=x.dtype)
        self.size = size
        self.pos = 0
        self.last = None

    def __len__(self):
        return min(self.pos, self.size)

    def __getitem__(self, k):
        idx = (self.pos + k if k < 0 else self.pos + -self.size + k) % self.size
        # print(f"\nFETCH {k}: pos={self.pos}, size={self.size}, at={idx}")
        return self.history[idx]

    def push(self, val):
        # print(f"\nPUSH {self.pos % self.size}: pos={self.pos}, size={self.size}")
        self.last = self.pos % self.size
        self.history[self.last] = val
        self.pos += 1

    def reset(self):
        self.pos = 0
        self.last = None


class ModelCallCache:
    def __init__(
        self, model, x, s_in, extra_args, *, size=0, max_use=1000000, threshold=1
    ):
        self.size = size
        self.model = model
        self.threshold = threshold
        self.s_in = s_in
        self.extra_args = extra_args
        self.max_use = max_use
        if self.size < 1:
            return
        self.mcc = torch.zeros(size, *x.shape, device=x.device, dtype=x.dtype)
        self.jmcc = torch.zeros_like(self.mcc)
        self.reset_cache()

    def reset_cache(self):
        size = self.size
        self.slot = [None] * size
        self.jslot = [None] * size
        self.slot_use = [self.max_use] * size

    def get(self, idx, *, jvp=False):
        idx -= self.threshold
        if (
            idx >= self.size
            or idx < 0
            or self.slot[idx] is None
            or self.slot_use[idx] < 1
        ):
            return None
        if jvp and self.jslot[idx] is None:
            return None
        self.slot_use[idx] -= 1
        return self.slot[idx] if not jvp else (self.slot[idx], self.jslot[idx])

    def set(self, idx, denoised, jdenoised=None):
        idx -= self.threshold
        if idx < 0 or idx >= self.size:
            return
        self.slot_use[idx] = self.max_use
        self.slot[idx] = denoised
        self.jslot[idx] = jdenoised

    def call_model(self, x, sigma, **kwargs):
        return self.model(x, sigma * self.s_in, **self.extra_args, **kwargs)

    def __call__(self, x, sigma, *, model_call_idx=0, tangents=None, **kwargs):
        result = self.get(model_call_idx, jvp=tangents is not None)
        # print(
        #     f"MODEL: idx={model_call_idx}, size={self.size}, threshold={self.threshold}, cached={result is not None}"
        # )
        if result is not None:
            return result
        if tangents is None:
            denoised = self.call_model(x, sigma, **kwargs)
            self.set(model_call_idx, denoised)
            return denoised
        denoised, denoised_prime = torch.func.jvp(self.call_model, (x, sigma), tangents)
        self.set(model_call_idx, denoised, jdenoised=denoised_prime)
        return denoised, denoised_prime


class SamplerState:
    def __init__(
        self,
        model,
        sigmas,
        idx,
        dhist,
        xhist,
        extra_args,
        *,
        noise_sampler,
        callback=None,
        denoised=None,
        eta=1.0,
        reta=1.0,
        s_noise=1.0,
    ):
        self.model = model
        self.dhist = dhist
        self.xhist = xhist
        self.extra_args = extra_args
        self.eta = eta
        self.reta = reta
        self.s_noise = s_noise
        self.sigmas = sigmas
        self.denoised = denoised
        self.callback_ = callback
        self.noise_sampler = noise_sampler
        self.update(idx)

    def update(self, idx=None):
        idx = self.idx if idx is None else idx
        self.idx = idx
        self.sigma_prev = None if idx < 1 else self.sigmas[idx - 1]
        self.sigma, self.sigma_next = self.sigmas[idx], self.sigmas[idx + 1]
        # if self.sigma_prev is not None and self.sigma < self.sigma_prev:
        #     self.dhist.reset()
        #     self.xhist.reset()
        self.sigma_down, self.sigma_up = get_ancestral_step(
            self.sigma, self.sigma_next, eta=self.eta
        )
        self.sigma_down_reversible, self.sigma_up_reversible = get_ancestral_step(
            self.sigma, self.sigma_next, eta=self.reta
        )

    def get_ancestral_step(self, eta=1.0):
        return get_ancestral_step(self.sigma, self.sigma_next, eta=eta)

    def clone_edit(self, **kwargs):
        obj = self.__class__.__new__(self.__class__)
        for k in (
            "model",
            "dhist",
            "xhist",
            "extra_args",
            "eta",
            "reta",
            "s_noise",
            "sigmas",
            "denoised",
            "callback_",
            "noise_sampler",
            "idx",
            "sigma",
            "sigma_next",
            "sigma_prev",
            "sigma_down",
            "sigma_up",
            "sigma_down_reversible",
            "sigma_up_reversible",
        ):
            setattr(obj, k, kwargs[k] if k in kwargs else getattr(self, k))
        obj.update()
        return obj

    def callback(self, x):
        if not self.callback_:
            return None
        return self.callback_(
            {
                "x": x,
                "i": self.idx,
                "sigma": self.sigma,
                "sigma_hat": self.sigma,
                "denoised": self.dhist[-1],
            }
        )
