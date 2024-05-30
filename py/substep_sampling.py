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


class SamplerState:
    def __init__(
        self,
        model,
        sigmas,
        idx,
        s_in,
        dhist,
        xhist,
        extra_args,
        *,
        noise_sampler,
        callback=None,
        denoised=None,
        model_call_cache=None,
        eta=1.0,
        reta=1.0,
        s_noise=1.0,
    ):
        self.model_ = model
        self.dhist = dhist
        self.xhist = xhist
        self.extra_args = extra_args
        self.s_in = s_in
        self.eta = eta
        self.reta = reta
        self.s_noise = s_noise
        self.sigmas = sigmas
        self.denoised = denoised
        self.callback_ = callback
        self.noise_sampler = noise_sampler
        self.model_call_cache = model_call_cache
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

    def model(self, x, sigma, *, model_call_idx=0, **kwargs):
        mcc = self.model_call_cache
        if mcc is None or model_call_idx >= mcc.size:
            return self.model_(x, sigma * self.s_in, **self.extra_args, **kwargs)
        if model_call_idx < mcc.pos:
            print("CACHED MODEL CALL", model_call_idx)
            return mcc.history[model_call_idx]
        result = self.model_(x, sigma * self.s_in, **self.extra_args, **kwargs)
        mcc.push(result)
        print("CACHING MODEL CALL", model_call_idx, mcc.size, mcc.pos)
        return result

    def get_ancestral_step(self, eta=1.0):
        return get_ancestral_step(self.sigma, self.sigma_next, eta=eta)

    def clone_edit(self, **kwargs):
        obj = self.__class__.__new__(self.__class__)
        for k in (
            "model_",
            "dhist",
            "xhist",
            "model_call_cache",
            "extra_args",
            "s_in",
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
