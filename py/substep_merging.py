import torch

from .utils import scale_noise, find_first_unsorted
from .substep_sampling import History


class MergeSubstepsSampler:
    def __init__(self, ss, samplers, **_kwargs):
        self.ss = ss
        self.samplers = samplers

    def step(self, x):
        raise NotImplementedError

    def merge_steps(self, _x, result):
        return result


class NormalMergeSubstepsSampler(MergeSubstepsSampler):
    def __init__(self, ss, samplers, **kwargs):
        super().__init__(ss, samplers, **kwargs)
        self.ss = ss

    def step(self, x):
        ss = self.ss
        substeps = len(self.samplers)
        renoise_weight = 1.0 / substeps
        z_avg = torch.zeros_like(x)
        noise = torch.zeros_like(x)
        noise_total = 0.0
        for subidx, ssampler in enumerate(self.samplers):
            print("  SUBSTEP", subidx, ssampler.name)
            ss.denoised = ss.model(x, ss.sigma * ss.s_in)
            z_k, noise_strength = ssampler.step(x, ss)
            z_avg += renoise_weight * z_k
            if ss.sigma_next == 0:
                continue
            noise_strength *= ssampler.s_noise
            noise_curr = ssampler.noise_sampler(ss.sigma, ss.sigma_next)
            x = z_k
            if subidx != substeps - 1:
                x += noise_curr * noise_strength
            noise_total += noise_strength.item() * renoise_weight
            noise += noise_curr * noise_strength
        ss.dhist.push(ss.denoised)
        ss.denoised = None
        x = self.merge_steps(x, z_avg)
        if ss.sigma_next != 0 and noise_total != 0:
            x += scale_noise(noise, noise_total * ss.s_noise)
        ss.xhist.push(x)
        ss.callback(x)
        return x


class AverageMergeSubstepsSampler(NormalMergeSubstepsSampler):
    def __init__(self, ss, samplers, *, avgmerge_stretch=0.4, **kwargs):
        super().__init__(ss, samplers, **kwargs)
        self.ss = ss
        self.stretch = avgmerge_stretch

    def step(self, x):
        ss = orig_ss = self.ss
        substeps = len(self.samplers)
        renoise_weight = 1.0 / substeps
        z_avg = torch.zeros_like(x)
        noise = torch.zeros_like(x)
        stretch = (ss.sigma - ss.sigma_next) * self.stretch
        sig_adj = ss.sigma + stretch
        ss = self.ss.clone_edit(sigma=sig_adj)
        orig_x = x
        x = x + ss.noise_sampler(orig_ss.sigma, ss.sigma_next) * stretch * ss.s_noise
        ss.denoised = ss.model(x, sig_adj * ss.s_in)
        noise_total = 0.0
        for subidx, ssampler in enumerate(self.samplers):
            print("  SUBSTEP", subidx, ssampler.name)
            curr_x = orig_x + scale_noise(
                ssampler.noise_sampler(sig_adj, ss.sigma_next),
                ssampler.s_noise * stretch,
            )
            z_k, noise_strength = ssampler.step(curr_x, ss)
            z_avg += renoise_weight * z_k
            if ss.sigma_next == 0:
                continue
            noise_strength *= ssampler.s_noise
            noise_curr = ssampler.noise_sampler(ss.sigma, ss.sigma_next)
            noise_total += noise_strength.item() * renoise_weight
            noise += noise_curr * noise_strength
        ss.dhist.push(ss.denoised)
        ss.denoised = None
        x = self.merge_steps(x, z_avg)
        if ss.sigma_next != 0 and noise_total != 0:
            x += scale_noise(noise, noise_total * ss.s_noise)
        ss.xhist.push(x)
        ss.callback(x)
        return x


class SampleMergeSubstepsSampler(AverageMergeSubstepsSampler):
    cache_model = True

    def __init__(self, ss, samplers, *, merge_sampler, **kwargs):
        super().__init__(ss, samplers, **kwargs)
        self.merge_sampler = merge_sampler
        self.merge_ss = None

    def step(self, x):
        ss = self.ss
        substeps = len(self.samplers)
        renoise_weight = 1.0 / substeps
        z_avg = torch.zeros_like(x)
        curr_x = x
        ss.denoised = None
        stretch = (ss.sigma - ss.sigma_next) * self.stretch
        sig_adj = ss.sigma + stretch
        ss = self.ss.clone_edit(sigma=sig_adj)
        for subidx, ssampler in enumerate(self.samplers):
            if subidx == 0 or not self.cache_model:
                ss.denoised = ss.model(
                    curr_x,
                    # + ss.noise_sampler(sig_adj.sigma, ss.sigma_next) * stretch * ss.s_noise,
                    ss.s_in * sig_adj,
                )
            curr_x = (
                x
                + ssampler.noise_sampler(sig_adj, ss.sigma_next)
                * ssampler.s_noise
                * stretch
            )
            print("  SUBSTEP", subidx, ssampler.name, stretch)
            z_k, noise_strength = ssampler.step(curr_x, ss)
            z_avg += renoise_weight * z_k
            curr_x = z_k
            if not noise_strength or ss.sigma_next == 0:
                continue
            curr_x += (
                ssampler.noise_sampler(ss.sigma, ss.sigma_next)
                * ssampler.s_noise
                * noise_strength
            )
        ss.dhist.push(ss.denoised)
        ss.denoised = None
        x = self.merge_steps(curr_x, z_avg)
        ss.xhist.push(x)
        ss.callback(x)
        return x

    def merge_steps(self, x, result):
        if self.ss.model_call_cache is not None:
            self.ss.model_call_cache.reset()
        msampler = self.merge_sampler
        if self.merge_ss is None:
            merge_ss = self.merge_ss = self.ss.clone_edit(
                denoised=result,
                dhist=History(x, 3),
                xhist=History(x, 2),
                s_noise=msampler.s_noise,
                eta=msampler.eta,
                model_call_cache=None,
            )
        else:
            merge_ss = self.merge_ss
            merge_ss.denoised = result
        merge_ss.update(self.ss.idx)
        final = merge_ss.sigma_next == 0
        merged, noise_strength = msampler.step(x, merge_ss)
        if not final:
            ss = self.ss
            merged = (
                merged
                + msampler.noise_sampler(ss.sigma, ss.sigma_next)
                * msampler.s_noise
                * ss.sigma_up
            )
        merge_ss.dhist.push(result)
        merge_ss.xhist.push(merged)
        merge_ss.denoised = None
        return merged


class SampleUncachedMergeSubstepsSampler(SampleMergeSubstepsSampler):
    cache_model = False


class DivideMergeSubstepsSampler(MergeSubstepsSampler):
    def __init__(self, ss, samplers, *, schedule_multiplier=4, **kwargs):
        super().__init__(ss, samplers, **kwargs)
        self.schedule_multiplier = schedule_multiplier

    def step(self, x):
        ss = self.ss
        samplers = self.samplers
        substeps = len(samplers)
        max_steps = len(self.ss.sigmas) - 1
        sigmas_slice = ss.sigmas[
            ss.idx : min(max_steps + 1, ss.idx + self.schedule_multiplier)
        ]
        # print("SLICE", sigmas_slice)
        unsorted_idx = find_first_unsorted(sigmas_slice)
        if unsorted_idx is not None:
            sigmas_slice = sigmas_slice[:unsorted_idx]
        # print("SLICE ADJ", sigmas_slice)
        chunks = tuple(
            torch.linspace(
                sigmas_slice[idx],
                sigmas_slice[idx + 1],
                steps=substeps + 1,
                device=sigmas_slice.device,
                dtype=sigmas_slice.dtype,
            )[0 if not idx else 1 :]
            for idx in range(len(sigmas_slice) - 1)
        )
        # print("CHUNKS", chunks)
        subsigmas = torch.cat(chunks)
        # print("SUBSIGMAS", subsigmas)
        subss = self.ss.clone_edit(idx=0, sigmas=subsigmas)
        subss.main_idx = ss.idx
        subss.main_sigmas = ss.sigmas
        for subidx in range(substeps):
            subss.update(subidx)
            subss.denoised = subss.model(x, subss.sigma)
            ssampler = samplers[subidx]
            x, noise_strength = ssampler.step(x, subss)
            if not noise_strength or subss.sigma_next == 0:
                continue
            x = (
                x
                + ssampler.noise_sampler(subss.sigma, subss.sigma_next)
                * ssampler.s_noise
                * noise_strength
            )
            subss.xhist.push(x)
            subss.dhist.push(subss.denoised)
            subss.denoised = None
        ss.callback(x)
        return x


MERGE_SUBSTEPS_CLASSES = {
    "normal": NormalMergeSubstepsSampler,
    "divide": DivideMergeSubstepsSampler,
    "average": AverageMergeSubstepsSampler,
    "sample": SampleMergeSubstepsSampler,
    "sample_uncached": SampleUncachedMergeSubstepsSampler,
}
