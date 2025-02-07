import torch

from ..noise import scale_noise, ImmiscibleNoise
from .base import CustomNoiseItemBase


class ImmiscibleReferenceItem(CustomNoiseItemBase):
    def __init__(
        self,
        factor,
        *,
        size: int,
        batching: str,
        normalize_ref_scale: float,
        normalize_noise_scale: float,
        maximize: bool,
        distance_scale: float,
        distance_scale_ref: float,
        blend: float,
        noise,
        reference,
        blend_function,
        normalized=None,
    ):
        super().__init__(
            factor,
            size=size,
            batching=batching,
            normalize_ref_scale=normalize_ref_scale,
            normalize_noise_scale=normalize_noise_scale,
            maximize=maximize,
            distance_scale=distance_scale,
            distance_scale_ref=distance_scale_ref,
            blend=blend,
            blend_function=blend_function,
            noise=noise,
            reference=reference,
            normalized=normalized,
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        elif k == "reference":
            return self.reference.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        norm_noise_scale = self.normalize_noise_scale
        normalize = self.get_normalize("normalize", normalized)

        ns = self.noise.make_noise_sampler(x, *args, normalized=False, **kwargs)
        immiscible = ImmiscibleNoise(
            size=self.size,
            batching=self.batching,
            distance_scale=self.distance_scale,
            distance_scale_ref=self.distance_scale_ref,
            maximize=self.maximize,
        )
        ref_latent = self.reference.detach().clone().to(x)
        if self.normalize_ref_scale:
            ref_latent = scale_noise(
                ref_latent, self.normalize_ref_scale, normalized=True
            )
        blend = self.blend
        blend_function = self.blend_function
        blending = self.blend != 1.0
        repeat_count = max(1, self.size) + int(blending)
        batch_size = x.shape[0]

        def noise_sampler(s, sn, *args, **kwargs):
            noise_batch = torch.cat(tuple(ns(s, sn) for _ in range(repeat_count)))
            immiscible_noise = immiscible.unbatch(
                immiscible.immiscible(
                    immiscible.batch(
                        scale_noise(
                            noise_batch[batch_size * int(blending) :],
                            1.0 if norm_noise_scale == 0 else norm_noise_scale,
                            normalized=norm_noise_scale != 0,
                        )
                    ),
                    immiscible.batch(ref_latent),
                ),
                ref_latent.shape,
            )
            immiscible_noise = scale_noise(immiscible_noise, normalized=normalize)
            if blend != 1:
                immiscible_noise = blend_function(
                    noise_batch[: batch_size * int(blending)],
                    immiscible_noise,
                    blend,
                )
            return scale_noise(immiscible_noise, normalized=normalize)

        return noise_sampler
