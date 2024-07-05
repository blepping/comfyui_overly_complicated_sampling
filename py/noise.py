import collections
import gc
import random

import scipy
import torch

from .utils import scale_noise, fallback


class ImmiscibleNoise(
    collections.namedtuple(
        "ImmiscibleConfig",
        ("size", "mode", "batching", "scale_ref", "normalize_ref", "strength"),
        defaults=(0, "x", "channels", 1.0, False, 1.0),
    )
):
    def __call__(self, noise_sampler, x_ref, refs=None):
        if (
            self.size == 0
            or self.strength == 0
            or (self.size < 2 and self.batching == "batch")
        ):
            return noise_sampler()
        noise = torch.cat(tuple(noise_sampler() for _ in range(self.size)))
        ref = self.get_ref(x_ref, refs=refs)
        result = self.unbatch(self.choose(*self.batch(noise, ref)), x_ref)
        if self.strength == 1:
            return result
        return (self.strength * result) + (
            (1.0 - self.strength) * noise[: x_ref.shape[0]]
        )  # LERP

    def get_ref(self, x_ref, refs=None):
        ref = fallback(refs, {}).get(self.mode, x_ref)
        ref = scale_noise(
            ref.clone(),
            self.scale_ref,
            normalized=self.normalize_ref not in (False, None),
            normalize_dims=self.normalize_ref
            if isinstance(self.normalize_ref, (tuple, list))
            else (-3, -2, -1),
        )
        return ref

    def batch(self, noise, ref):
        if self.batching == "batch":
            return noise, ref
        nsz, rsz = noise.shape, ref.shape
        if len(nsz) != 4 or len(rsz) != 4:
            raise ValueError("Both noise and reference must be four-dimensional")
        if self.batching == "channel":
            noise = noise.view(nsz[0] * nsz[1], *nsz[2:])
            ref = ref.view(rsz[0] * rsz[1], *rsz[2:])
            return noise, ref
        if self.batching == "row":
            noise = noise.view(nsz[0] * nsz[1] * nsz[2], nsz[3])
            ref = ref.view(rsz[0] * rsz[1] * rsz[2], rsz[3])
            return noise, ref
        if self.batching == "column":
            noise = noise.permute(0, 1, 3, 2).reshape(nsz[0] * nsz[1] * nsz[3], nsz[2])
            ref = ref.permute(0, 1, 3, 2).reshape(rsz[0] * rsz[1] * rsz[3], rsz[2])
            return noise, ref
        raise ValueError("Bad Immmiscible noise batching type")

    def unbatch(self, noise, x_ref):
        xsz = x_ref.shape
        if self.batching == "column":
            return noise.view(*xsz[:2], xsz[3], xsz[2]).permute(0, 1, 3, 2)
        return noise.view(*xsz)

    # Based on implementation from https://github.com/kohya-ss/sd-scripts/pull/1395
    # Idea from https://github.com/Clybius
    @staticmethod
    def choose(noise, latents):
        # "Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment" (2024) Li et al. arxiv.org/abs/2406.12303
        # Minimize latent-noise pairs over a batch
        n = noise.shape[0]
        latents_expanded = latents.half().unsqueeze(1).expand(-1, n, *latents.shape[1:])
        noise_expanded = (
            noise.half().unsqueeze(0).expand(latents.shape[0], *noise.shape)
        )
        dist = (latents_expanded - noise_expanded) ** 2
        dist = dist.mean(list(range(2, dist.dim()))).cpu()
        assign_mat = scipy.optimize.linear_sum_assignment(dist)
        print("IMM IDX", assign_mat[1])
        return noise[assign_mat[1]]


class NoiseSamplerCache:
    def __init__(
        self,
        x,
        seed,
        min_sigma,
        max_sigma,
        *,
        normalize_noise=True,
        cpu_noise=True,
        batch_size=32,
        caching=True,
        cache_reset_interval=1,
        set_seed=False,
        scale=1.0,
        normalize_dims=(-3, -2, -1),
        immiscible=None,
        **_unused,
    ):
        self.x = x
        self.mega_x = None
        self.seed = seed
        self.seed_offset = 0
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.cache = {}
        self.batch_size = max(1, batch_size)
        self.normalize_noise = normalize_noise
        self.cpu_noise = cpu_noise
        self.caching = caching
        self.cache_reset_interval = max(1, cache_reset_interval)
        self.scale = float(scale)
        self.normalize_dims = tuple(int(v) for v in normalize_dims)
        self.immiscible = ImmiscibleNoise(**fallback(immiscible, {}))
        print("IMM", self.immiscible)
        self.update_x(x)
        if set_seed:
            random.seed(seed)
            torch.manual_seed(seed)

    def reset_cache(self):
        self.cache = {}
        gc.collect()

    def scale_noise(self, noise, factor=1.0, normalized=None, normalize_dims=None):
        normalized = self.normalize_noise if normalized is None else normalized
        normalize_dims = (
            self.normalize_dims if normalize_dims is None else normalize_dims
        )
        return scale_noise(
            noise, factor, normalized=normalized, normalize_dims=normalize_dims
        )

    def update_x(self, x):
        if self.x.shape == x.shape and self.mega_x is not None:
            self.x = x
            return
        self.x = x
        self.cache = {}
        self.mega_x = None
        if self.batch_size == 1:
            self.mega_x = x
            return
        self.mega_x = x.repeat(x.shape[0] * self.batch_size, *((1,) * (x.dim() - 1)))

    def set_cache(self, key, noise_sampler):
        if not self.caching:
            return
        self.cache[key] = noise_sampler

    def make_caching_noise_sampler(self, nsobj, size, sigma, sigma_next):
        size = min(size, self.batch_size)
        cache_key = (nsobj, size)
        if self.caching:
            noise_sampler = self.cache.get(cache_key)
            if noise_sampler:
                return noise_sampler
        curr_seed = self.seed + self.seed_offset
        self.seed_offset += 1
        curr_x = self.mega_x[: self.x.shape[0] * size, ...]
        if nsobj is None:

            def ns(_s, _sn, *_unused, **_unusedkwargs):
                return torch.randn_like(curr_x)
        else:
            ns = nsobj.make_noise_sampler(
                curr_x,
                self.min_sigma,
                self.max_sigma,
                seed=curr_seed,
                normalized=False,
                cpu=self.cpu_noise,
            )

        orig_h, orig_w = self.x.shape[-2:]
        remain = 0
        noise = None

        def noise_sampler_(
            *_unused,
            out_hw=(orig_h, orig_w),
            **_unusedkwargs,
        ):
            nonlocal remain, noise
            if out_hw != (orig_h, orig_w):
                raise NotImplementedError(
                    f"Noise size mismatch: {out_hw} vs {(orig_h, orig_w)}"
                )
            if remain < 1:
                noise = self.scale_noise(ns(sigma, sigma_next)).view(
                    size,
                    *self.x.shape,
                )
                remain = size
            result = noise[-remain]
            remain -= 1
            return result

        def noise_sampler(*args, x_ref=None, refs=None, **kwargs):
            return self.immiscible(
                lambda args=args, kwargs=kwargs: noise_sampler_(*args, **kwargs),
                fallback(x_ref, self.x),
                refs=refs,
            )

        self.set_cache(cache_key, noise_sampler)
        return noise_sampler
