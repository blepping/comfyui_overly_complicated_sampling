import gc
import math
import random

import scipy
import torch

from .filtering import Filter, make_filter
from .utils import scale_noise, fallback


class ImmiscibleNoise(Filter):
    name = "immiscible"
    uses_ref = True
    default_options = Filter.default_options | {
        "size": 0,
        "batching": "channel",
        "maximize": False,
        "distance_scale": 0.0,
        "distance_scale_ref": None,
    }

    def __call__(self, noise_sampler, x_ref, *, refs=None):
        if self.size == 0 or self.strength == 0 or not self.check_applies(refs):
            return noise_sampler()
        size = self.size if self.strength == 1.0 else self.size + 1
        return self.apply(
            torch.cat(tuple(noise_sampler() for _ in range(size))),
            default_ref=x_ref,
            refs=refs,
            output_shape=x_ref.shape,
        )

    def filter(self, latent, ref_latent, *, refs, output_shape):
        if self.size == 0:
            return latent
        offset = 0 if self.strength == 1.0 else output_shape[0]
        return self.unbatch(
            self.immiscible(self.batch(latent[offset:]), self.batch(ref_latent)),
            output_shape,
        )

    def batch(self, latent):
        if self.batching == "batch":
            return latent
        sz = latent.shape
        if self.batching == "channel":
            return latent.reshape(sz[0] * sz[1], *sz[2:])
        if self.batching == "frame":
            if latent.ndim != 5:
                raise ValueError(
                    "Both latent and reference must be five-dimensional for frame mode"
                )
            return latent.permute(0, 2, 1, 3, 4).reshape(sz[0] * sz[2], sz[1], *sz[3:])
        if self.batching == "row":
            return latent.reshape(math.prod(sz[:-1]), sz[-1])
        if self.batching == "column":
            if latent.ndim != 4:
                raise ValueError(
                    "Both latent and reference must be four-dimensional for column mode"
                )
            return latent.permute(0, 1, 3, 2).reshape(sz[0] * sz[1] * sz[3], sz[2])
        raise ValueError("Bad Immmiscible noise batching type")

    def unbatch(self, latent, sz):
        if self.batching == "column":
            return latent.reshape(*sz[:2], sz[3], sz[2]).permute(0, 1, 3, 2)
        if self.batching == "frame":
            return latent.reshape(sz[0], sz[2], sz[1], *sz[3:]).permute(0, 2, 1, 3, 4)
        return latent.reshape(*sz)

    # Based on implementation from https://github.com/kohya-ss/sd-scripts/pull/1395
    # Idea for use with inference as well as implementation help from https://github.com/Clybius
    def immiscible(self, latent, ref_latent):
        # "Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment" (2024) Li et al. arxiv.org/abs/2406.12303
        # Minimize latent-noise pairs over a batch
        n = latent.shape[0]
        ref_latent = ref_latent.detach().clone()
        if self.distance_scale == 0:
            ref_latent_expanded = (
                ref_latent.half().unsqueeze(1).expand(-1, n, *ref_latent.shape[1:])
            )
            latent_expanded = (
                latent.half().unsqueeze(0).expand(ref_latent.shape[0], *latent.shape)
            )
            dist = (ref_latent_expanded - latent_expanded) ** 2
            del ref_latent_expanded, latent_expanded
            dist = dist.mean(list(range(2, dist.dim())))
        else:
            dist = torch.linalg.vector_norm(
                fallback(self.distance_scale_ref, self.distance_scale)
                * ref_latent.half().flatten(start_dim=1).unsqueeze(1)
                - self.distance_scale * latent.half().flatten(start_dim=1).unsqueeze(0),
                dim=2,
            )
        try:
            assign_mat = scipy.optimize.linear_sum_assignment(
                dist.cpu(), maximize=self.maximize
            )
        except ValueError:
            return latent[: ref_latent.shape[0]]
        return latent[assign_mat[1]]


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
        batch_size=1,
        caching=True,
        cache_reset_interval=9999,
        set_seed=True,
        seed_offset=1,
        scale=1.0,
        normalize_dims=(-3, -2, -1),
        immiscible=None,
        filter=None,
        **_unused,
    ):
        self.x = x
        self.mega_x = None
        self.seed = seed
        self.seed_offset = seed_offset
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
        if filter is None:
            self.filter = None
        else:
            self.filter = make_filter(filter)
        self.update_x(x)
        if set_seed:
            random.seed(seed)
            torch.manual_seed(seed)
            if self.seed_offset > 0:
                for _ in range(self.seed_offset):
                    _ = torch.randn_like(x)
        else:
            self.seed_offset = 0

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
        self.mega_x = None
        self.reset_cache()
        if self.batch_size == 1:
            self.mega_x = x
            return
        self.mega_x = x.repeat(x.shape[0] * self.batch_size, *((1,) * (x.dim() - 1)))

    def set_cache(self, key, noise_sampler):
        if not self.caching:
            return
        self.cache[key] = noise_sampler

    def make_caching_noise_sampler(
        self,
        nsobj,
        size,
        sigma,
        sigma_next,
        *,
        immiscible=None,
        sigmas=None,
    ):
        size = min(size, self.batch_size)
        if immiscible is None:
            immiscible = self.immiscible
        cache_key = (nsobj, size, hash(immiscible))
        if self.caching:
            noise_sampler = self.cache.get(cache_key)
            if noise_sampler:
                return noise_sampler
        curr_seed = self.seed + self.seed_offset
        self.seed_offset += 1
        curr_x = self.mega_x[: self.x.shape[0] * size, ...]
        if nsobj is None:

            def ns(*_unused, **_unusedkwargs):
                noise = torch.randn(
                    curr_x.shape,
                    dtype=curr_x.dtype,
                    layout=curr_x.layout,
                    device="cpu" if self.cpu_noise else curr_x.device,
                )
                if noise.device != curr_x.device:
                    return noise.to(curr_x.device)
                return noise

        else:
            if sigmas is not None:
                sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
            else:
                sigma_min, sigma_max = self.min_sigma, self.max_sigma
            ns = nsobj.make_noise_sampler(
                curr_x,
                sigma_min,
                sigma_max,
                seed=curr_seed,
                normalized=False,
                cpu=self.cpu_noise,
            )

        orig_h, orig_w = self.x.shape[-2:]
        remain = 0
        noise = None

        def noise_sampler_(
            curr_sigma,
            curr_sigma_next,
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
                curr_sigma = fallback(curr_sigma, sigma)
                curr_sigma_next = fallback(curr_sigma_next, sigma_next)
                noise = self.scale_noise(ns(curr_sigma, curr_sigma_next)).view(
                    size,
                    *self.x.shape,
                )
                remain = size
            result = noise[-remain]
            remain -= 1
            return result

        def noise_sampler(*args, x_ref=None, refs=None, **kwargs):
            if immiscible is False:
                noise = noise_sampler_(*args, **kwargs)
            else:
                noise = immiscible(
                    lambda args=args, kwargs=kwargs: noise_sampler_(*args, **kwargs),
                    fallback(x_ref, self.x),
                    refs=refs,
                )
            return (
                self.filter.apply(noise, refs=refs)
                if self.filter is not None
                else noise
            )

        self.set_cache(cache_key, noise_sampler)
        return noise_sampler
