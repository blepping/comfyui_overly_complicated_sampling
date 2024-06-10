import torch
import contextlib

from .utils import find_first_unsorted
from .substep_samplers import STEP_SAMPLERS
from .substep_sampling import History


class MergeSubstepsSampler:
    def __init__(self, ss, sitems, **kwargs):
        samplers = tuple(
            STEP_SAMPLERS[sitem["step_method"]](**sitem) for sitem in sitems
        )
        self.ss = ss
        self.samplers = samplers
        self.substeps = sum(sampler.substeps for sampler in samplers)
        self.options = kwargs

    def step(self, x):
        raise NotImplementedError

    def substep(self, x, sampler, ss=None):
        if ss is None:
            ss = self.ss
        sg = sampler.step(x, ss)
        next_x = None
        with contextlib.suppress(StopIteration):
            while True:
                sr = sg.send(next_x)
                next_x = sr.x
                yield sr

    def simple_substep(self, x, sampler, ss=None):
        for sr in self.substep(x, sampler, ss=ss):
            if not sr.final:
                sr.noise_x()
        return sr

    def merge_steps(self, x, result=None, *, noise=None, ss=None):
        if result is None:
            result = x
        ss = ss if ss is not None else self.ss
        ss.dhist.push(ss.denoised)
        ss.denoised = None
        ss.callback(result)
        if noise is not None:
            result = result + noise
        ss.xhist.push(result)
        return result

    def step_max_noise_samples(self):
        return sum(
            (1 + sampler.self_noise) * sampler.substeps for sampler in self.samplers
        )


class SimpleSubstepsSampler(MergeSubstepsSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not len(self.samplers):
            raise ValueError("Missing sampler")

    def step_max_noise_samples(self):
        return 1 + self.samplers[0].self_noise

    def step(self, x):
        ss, ssampler = self.ss, self.samplers[0]
        custom_noise = ssampler.options.get(
            "custom_noise", self.options.get("custom_noise")
        )
        noise_sampler = ss.noise.make_caching_noise_sampler(
            custom_noise, 1, ss.sigma, ss.sigma_next
        )
        ssampler.noise_sampler = noise_sampler
        ss.denoised = ss.model(x, ss.sigma)
        sr = self.simple_substep(x, ssampler)
        return self.merge_steps(sr.x, noise=sr.get_noise())


class NormalMergeSubstepsSampler(MergeSubstepsSampler):
    def step(self, x):
        ss = self.ss
        substeps = self.substeps
        renoise_weight = 1.0 / substeps
        z_avg = torch.zeros_like(x)
        noise = torch.zeros_like(x)
        noise_total = 0.0
        substep = 0
        for ssampler in self.samplers:
            custom_noise = ssampler.options.get(
                "custom_noise", self.options.get("custom_noise")
            )
            noise_sampler = ss.noise.make_caching_noise_sampler(
                custom_noise, ssampler.max_noise_samples(), ss.sigma, ss.sigma_next
            )
            ssampler.noise_sampler = noise_sampler
            for subidx in range(ssampler.substeps):
                print(f"  SUBSTEP {substep + 1}: {ssampler.name}")
                self.ss.denoised = ss.denoised = ss.model(x, ss.sigma)
                sr = self.simple_substep(x, ssampler)
                x = sr.x
                z_avg += renoise_weight * x
                noise_strength = sr.noise_scale
                substep += 1
                if ss.sigma_next == 0 or noise_strength == 0:
                    continue
                noise_curr = sr.get_noise()
                if substep < substeps:
                    x = x + noise_curr
                noise_total += noise_strength.item() * renoise_weight
                noise += noise_curr
        return self.merge_steps(
            x,
            z_avg,
            noise=None
            if not noise_total
            else ss.noise.scale_noise(noise, noise_total * ss.s_noise, normalized=True),
        )


class AverageMergeSubstepsSampler(NormalMergeSubstepsSampler):
    def __init__(self, ss, sitems, *, avgmerge_stretch=0.4, **kwargs):
        super().__init__(ss, sitems, **kwargs)
        self.stretch = avgmerge_stretch

    def step_max_noise_samples(self):
        return sum(
            1 + (2 + sampler.self_noise) * sampler.substeps for sampler in self.samplers
        )

    def step(self, x):
        ss = orig_ss = self.ss
        substeps = self.substeps
        renoise_weight = 1.0 / substeps
        z_avg = torch.zeros_like(x)
        noise = torch.zeros_like(x)
        stretch = (ss.sigma - ss.sigma_next) * self.stretch
        sig_adj = ss.sigma + stretch
        ss = self.ss.clone_edit(sigma=sig_adj)
        orig_x = x
        stretch_strength = stretch * ss.s_noise
        if stretch_strength != 0:
            noise_sampler = ss.noise.make_caching_noise_sampler(
                self.options.get("custom_noise"), 1, orig_ss.sigma, ss.sigma_next
            )
            x = x + (
                noise_sampler(orig_ss.sigma, ss.sigma_next).mul_(stretch * ss.s_noise)
            )
        self.ss.denoised = ss.denoised = ss.model(x, sig_adj)
        noise_total = 0.0
        substep = 0
        for idx, ssampler in enumerate(self.samplers):
            print(
                f"  SUBSTEP {substep + 1} .. {substep + ssampler.substeps}: {ssampler.name}, stretch={stretch}"
            )
            custom_noise = ssampler.options.get(
                "custom_noise", self.options.get("custom_noise")
            )
            noise_sampler = ss.noise.make_caching_noise_sampler(
                custom_noise,
                ssampler.substeps
                + (0 if ss.sigma_next == 0 else ssampler.max_noise_samples()),
                ss.sigma,
                ss.sigma_next,
            )
            ssampler.noise_sampler = noise_sampler
            for sidx in range(ssampler.substeps):
                curr_x = orig_x + noise_sampler(sig_adj, ss.sigma_next).mul_(stretch)
                sr = self.simple_substep(curr_x, ssampler, ss=ss)
                z_avg += renoise_weight * sr.x
                noise_strength = sr.noise_scale
                if ss.sigma_next == 0 or noise_strength == 0:
                    continue
                noise_curr = sr.get_noise()
                noise_total += noise_strength.item() * renoise_weight
                noise += noise_curr
            substep += ssampler.substeps
        return self.merge_steps(
            x,
            z_avg,
            noise=None
            if not noise_total
            else ss.noise.scale_noise(noise, noise_total * ss.s_noise, normalized=True),
            ss=ss,
        )


class SampleMergeSubstepsSampler(AverageMergeSubstepsSampler):
    cache_model = True

    def __init__(self, ss, sitems, *, merge_sampler=None, **kwargs):
        super().__init__(ss, sitems, **kwargs)
        if merge_sampler is None:
            merge_sampler = STEP_SAMPLERS["euler"](step_method="euler")
        else:
            msitem = merge_sampler.items[0]
            merge_sampler = STEP_SAMPLERS[msitem["step_method"]](**msitem)
        self.merge_sampler = merge_sampler
        self.merge_ss = None

    def step(self, x):
        ss = self.ss
        substeps = self.substeps
        renoise_weight = 1.0 / substeps
        z_avg = torch.zeros_like(x)
        curr_x = x
        ss.denoised = None
        stretch = (ss.sigma - ss.sigma_next) * self.stretch
        sig_adj = ss.sigma + stretch
        ss = self.ss.clone_edit(sigma=sig_adj)
        step = 0
        for idx, ssampler in enumerate(self.samplers):
            print(
                f"  SUBSTEP {step + 1} .. {step + ssampler.substeps}: {ssampler.name}, stretch={stretch}"
            )
            custom_noise = ssampler.options.get(
                "custom_noise", self.options.get("custom_noise")
            )
            noise_sampler = ss.noise.make_caching_noise_sampler(
                custom_noise,
                ssampler.max_noise_samples() + ssampler.substeps,
                ss.sigma,
                ss.sigma_next,
            )
            ssampler.noise_sampler = noise_sampler
            for sidx in range(ssampler.substeps):
                if idx + sidx == 0 or not self.cache_model:
                    self.ss.denoised = ss.denoised = ss.model(
                        curr_x,
                        ss.sigma,
                        # + ss.noise_sampler(sig_adj.sigma, ss.sigma_next) * stretch * ss.s_noise,
                        # sig_adj,
                    )
                curr_x = x + noise_sampler(sig_adj, ss.sigma_next).mul_(
                    ssampler.s_noise * stretch
                )
                sr = self.simple_substep(curr_x, ssampler, ss=ss)
                z_avg += renoise_weight * sr.x
                curr_x = sr.noise_x(sr.x)
            step += ssampler.substeps
        return self.merge_steps(curr_x, z_avg)

    def merge_steps(self, x, result):
        ss = self.ss
        ss.dhist.push(ss.denoised)
        ss.denoised = None
        ss.model.reset_cache()
        msampler = self.merge_sampler
        if self.merge_ss is None:
            merge_ss = self.merge_ss = self.ss.clone_edit(
                denoised=result,
                dhist=History(x, 3),
                xhist=History(x, 2),
                s_noise=msampler.s_noise,
                eta=msampler.eta,
            )
        else:
            merge_ss = self.merge_ss
            merge_ss.denoised = result
        merge_ss.update(self.ss.idx)
        final = merge_ss.sigma_next == 0
        noise_sampler = merge_ss.noise.make_caching_noise_sampler(
            msampler.options.get("custom_noise", self.options.get("custom_noise")),
            msampler.max_noise_samples() + int(not final),
            merge_ss.sigma,
            merge_ss.sigma_next,
        )
        msampler.noise_sampler = noise_sampler
        sr = self.simple_substep(x, msampler, ss=merge_ss)
        self.ss.callback(sr.x)
        sr.noise_x()
        merge_ss.dhist.push(result)
        merge_ss.xhist.push(sr.x)
        merge_ss.denoised = None
        ss.xhist.push(sr.x)
        return sr.x


class SampleUncachedMergeSubstepsSampler(SampleMergeSubstepsSampler):
    cache_model = False


class DivideMergeSubstepsSampler(MergeSubstepsSampler):
    def __init__(self, ss, sitems, *, schedule_multiplier=4, **kwargs):
        super().__init__(ss, sitems, **kwargs)
        self.schedule_multiplier = schedule_multiplier

    def make_schedule(self, ss):
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
                steps=self.substeps + 1,
                device=sigmas_slice.device,
                dtype=sigmas_slice.dtype,
            )[0 if not idx else 1 :]
            for idx in range(len(sigmas_slice) - 1)
        )
        # print("CHUNKS", chunks)
        return torch.cat(chunks)

    def step(self, x):
        ss = self.ss
        # print("SUBSIGMAS", subsigmas)
        subss = self.ss.clone_edit(idx=0, sigmas=self.make_schedule(ss))
        subss.main_idx = ss.idx
        subss.main_sigmas = ss.sigmas
        substep = 0
        for ssampler in self.samplers:
            custom_noise = ssampler.options.get(
                "custom_noise", self.options.get("custom_noise")
            )
            noise_sampler = ss.noise.make_caching_noise_sampler(
                custom_noise, ssampler.max_noise_samples(), ss.sigma, ss.sigma_next
            )
            ssampler.noise_sampler = noise_sampler
            for subidx in range(ssampler.substeps):
                print(f"  SUBSTEP {substep + 1}: {ssampler.name}")
                subss.update(substep)
                subss.denoised = subss.model(x, subss.sigma)
                sr = self.simple_substep(x, ssampler, ss=subss)
                x = sr.x
                noise_strength = sr.noise_scale
                subss.dhist.push(subss.denoised)
                subss.denoised = None
                if substep == self.substeps - 1:
                    ss.callback(x)
                if noise_strength != 0 and subss.sigma_next != 0:
                    x = sr.noise_x()
                subss.xhist.push(x)
                substep += 1
        # subss.xhist.push(x)
        # subss.dhist.push(subss.denoised)
        # subss.denoised = None
        return x

    # def step(self, x):
    #     ss = self.ss
    #     # print("SUBSIGMAS", subsigmas)
    #     subss = self.ss.clone_edit(idx=0, sigmas=self.make_schedule(ss))
    #     subss.main_idx = ss.idx
    #     subss.main_sigmas = ss.sigmas
    #     substep = 0
    #     for ssampler in self.samplers:
    #         custom_noise = ssampler.options.get(
    #             "custom_noise", self.options.get("custom_noise")
    #         )
    #         noise_sampler = ss.noise.make_caching_noise_sampler(
    #             custom_noise, ssampler.max_noise_samples(), ss.sigma, ss.sigma_next
    #         )
    #         ssampler.noise_sampler = noise_sampler
    #         for subidx in range(ssampler.substeps):
    #             print(f"  SUBSTEP {substep + 1}: {ssampler.name}")
    #             subss.update(substep)
    #             subss.denoised = subss.model(x, subss.sigma)
    #             sr = self.simple_substep(x, ssampler, ss=subss)
    #             x = sr.x
    #             noise_strength = sr.noise_scale
    #             subss.dhist.push(subss.denoised)
    #             subss.denoised = None
    #             if substep == self.substeps - 1:
    #                 ss.callback(x)
    #             if noise_strength != 0 and subss.sigma_next != 0:
    #                 x = sr.noise_x()
    #             subss.xhist.push(x)
    #             substep += 1
    #     return x


MERGE_SUBSTEPS_CLASSES = {
    "normal": NormalMergeSubstepsSampler,
    "divide": DivideMergeSubstepsSampler,
    "average": AverageMergeSubstepsSampler,
    "sample": SampleMergeSubstepsSampler,
    "sample_uncached": SampleUncachedMergeSubstepsSampler,
    "simple": SimpleSubstepsSampler,
}
