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
        blend_function,
        normalize=None,
        reference=None,
        custom_noise_blend=None,
        custom_noise_ref=None,
    ):
        if reference is None and custom_noise_ref is None:
            raise ValueError(
                "Either the reference latent or custom_noise_ref need to be supplied."
            )
        if reference is not None and custom_noise_ref is not None:
            raise ValueError(
                "One of reference latent or custom_noise_ref need to be supplied, but not both."
            )
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
            normalize=normalize,
            custom_noise_ref=custom_noise_ref,
            custom_noise_blend=custom_noise_blend,
        )

    def clone_key(self, k):
        if k == "noise":
            return self.noise.clone()
        if k == "reference" and self.reference is not None:
            return self.reference.clone()
        if k == "custom_noise_ref" and self.custom_noise_ref is not None:
            return self.custom_noise_ref.clone()
        if k == "custom_noise_blend" and self.custom_noise_blend is not None:
            return self.custom_noise_blend.clone()
        return super().clone_key(k)

    def make_noise_sampler(self, x, *args, normalized=True, **kwargs):
        factor = self.factor
        norm_noise_scale = self.normalize_noise_scale
        normalize = self.get_normalize("normalize", normalized)
        norm_ref = self.normalize_ref_scale

        ns = self.noise.make_noise_sampler(x, *args, normalized=False, **kwargs)
        batching = self.batching
        if "_" in batching:
            batchings = batching.split("_")
            batching = batchings[0]
            dual_mode = True
        else:
            dual_mode = False
        immiscible = ImmiscibleNoise(
            size=self.size,
            batching=batching,
            distance_scale=self.distance_scale,
            distance_scale_ref=self.distance_scale_ref,
            maximize=self.maximize,
        )
        if dual_mode:
            immiscible2 = immiscible = ImmiscibleNoise(
                size=self.size,
                batching=batchings[-1],
                distance_scale=self.distance_scale,
                distance_scale_ref=self.distance_scale_ref,
                maximize=self.maximize,
            )
        if self.reference is not None:
            ns_ref = None
            ref_latent = self.reference.detach().clone().to(x)
            if norm_ref:
                ref_latent = scale_noise(ref_latent, norm_ref, normalized=True)
        else:
            ref_latent = None
            ns_ref = self.custom_noise_ref.make_noise_sampler(
                x, *args, normalized=False, **kwargs
            )
        blend = self.blend
        blend_function = self.blend_function
        blending = self.blend != 1.0
        if self.custom_noise_blend is not None and blending:
            ns_blend = self.custom_noise_blend.make_noise_sampler(
                x, *args, normalized=False, **kwargs
            )
        else:
            ns_blend = None
        repeat_count = max(1, self.size) + int(blending)
        batch_size = x.shape[0]
        blend_in_batch = blending and ns_blend is None

        def noise_sampler(s, sn, *args, **kwargs):
            if ns_ref is not None:
                ref_latent = ns_ref(s, sn)
                if norm_ref:
                    ref_latent = scale_noise(ref_latent, norm_ref, normalized=True)

            noise_batch = torch.cat(tuple(ns(s, sn) for _ in range(repeat_count)))
            nb_input = scale_noise(
                noise_batch[batch_size * int(blend_in_batch) :],
                1.0 if norm_noise_scale == 0 else norm_noise_scale,
                normalized=norm_noise_scale != 0,
            )
            immiscible_noise = immiscible.unbatch(
                immiscible.immiscible(
                    immiscible.batch(nb_input),
                    immiscible.batch(ref_latent),
                ),
                ref_latent.shape,
            )
            if dual_mode:
                immiscible_noise = (
                    immiscible2.unbatch(
                        immiscible2.immiscible(
                            immiscible2.batch(nb_input),
                            immiscible2.batch(ref_latent),
                        ),
                        ref_latent.shape,
                    )
                    .add_(immiscible_noise)
                    .mul_(0.5)
                )
            immiscible_noise = scale_noise(immiscible_noise, normalized=normalize)
            if blend != 1:
                immiscible_noise = blend_function(
                    noise_batch[:batch_size] if ns_blend is None else ns_blend(s, sn),
                    immiscible_noise,
                    blend,
                )
            return scale_noise(immiscible_noise, factor, normalized=normalize)

        return noise_sampler
