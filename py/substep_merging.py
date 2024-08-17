import operator

import torch
import tqdm

from . import expression as expr
from . import utils

from .filtering import make_filter, FilterRefs
from .restart import Restart
from .step_samplers import STEP_SAMPLERS
from .utils import check_time, fallback


class MergeSubstepsSampler:
    name = "unknown"

    def __init__(self, ss, group):
        samplers = tuple(
            STEP_SAMPLERS[sitem["step_method"]](**sitem) for sitem in group.items
        )
        options = group.options.copy()
        self.time_mode = group.time_mode
        self.time_start = group.time_start
        self.time_end = group.time_end
        self.ss = ss
        self.samplers = samplers
        self.substeps = sum(sampler.substeps for sampler in samplers)
        when_expr = options.pop("when", None)
        self.when = expr.Expression(when_expr) if when_expr else None
        pre_filter = options.pop("pre_filter", None)
        post_filter = options.pop("post_filter", None)
        self.pre_filter = None if pre_filter is None else make_filter(pre_filter)
        self.post_filter = None if post_filter is None else make_filter(post_filter)
        self.preview_mode = options.pop("preview_mode", "denoised")
        self.options = options

    def check_match(self, handlers: None | object, *, ss: None | object = None):
        ss = fallback(ss, self.ss)
        if not check_time(
            self.time_mode,
            self.time_start,
            self.time_end,
            ss.sigma,
            ss.step,
            ss.total_steps,
        ):
            return False
        if self.when is None:
            return True
        if handlers is None:
            raise ValueError("Group has when expression but handlers not passed")
        return operator.truth(self.when.eval(handlers))

    def step_input(self, x, *, ss=None):
        if self.pre_filter is None:
            return x
        ss = fallback(ss, self.ss)
        return self.pre_filter.apply(x, refs=fallback(ss, self.ss).refs)

    def step_output(self, x, *, orig_x=None, ss=None):
        if self.post_filter is None:
            return x
        ss = fallback(ss, self.ss)
        refs = ss.refs if orig_x is None else ss.refs | FilterRefs({"orig_x": orig_x})
        return self.post_filter.apply(x, refs=refs)

    def __call__(self, x):
        orig_x = x
        x = self.step_input(x)
        x = self.step(x)
        return self.step_output(x, orig_x=orig_x)

    def step(self, x):
        raise NotImplementedError

    def substep(self, x, sampler, ss=None):
        sg = sampler(x, fallback(ss, self.ss))
        yield from utils.step_generator(sg, get_next=lambda sr: sr.x)

    def simple_substep(self, x, sampler, ss=None):
        for sr in self.substep(x, sampler, ss=ss):
            if not sr.final:
                sr.noise_x(ss=fallback(ss, self.ss))
        return sr

    def merge_steps(self, x, result=None, *, noise=None, ss=None, denoised=True):
        ss = ss if ss is not None else self.ss
        result = fallback(result, x)
        if noise is not None:
            result = result + noise
        return result

    def step_max_noise_samples(self):
        return sum(
            (1 + sampler.self_noise) * sampler.substeps for sampler in self.samplers
        )

    def reset(self):
        pass

    def callback(self, *, ss=None, mr=None, preview_mode=None):
        ss = fallback(ss, self.ss)
        preview_mode = fallback(preview_mode, self.preview_mode)
        return ss.callback(hi=mr, preview_mode=preview_mode)


class SimpleSubstepsSampler(MergeSubstepsSampler):
    name = "simple"

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
            custom_noise,
            1,
            ss.sigma,
            ss.sigma_next,
            immiscible=fallback(ssampler.immiscible, ss.noise.immiscible),
        )
        ssampler.noise_sampler = noise_sampler
        ss.hist.push(ss.model(x, ss.sigma, ss=ss))
        ss.refs = FilterRefs.from_ss(ss, have_current=True)
        self.callback()
        sr = self.simple_substep(x, ssampler)
        return self.merge_steps(sr.x, noise=sr.get_noise(ss=ss))


class NormalMergeSubstepsSampler(MergeSubstepsSampler):
    name = "normal"

    def step(self, x):
        ss = self.ss
        substeps = self.substeps
        renoise_weight = 1.0 / substeps
        z_avg = torch.zeros_like(x)
        noise = z_avg.clone()
        noise_total = 0.0
        substep = 0
        pbar = tqdm.tqdm(total=self.substeps, initial=1, disable=ss.disable_status)
        ss.hist.push(ss.model(x, ss.sigma, ss=ss))
        ss.refs = FilterRefs.from_ss(ss, have_current=True)
        self.callback()
        for ssampler in self.samplers:
            custom_noise = ssampler.options.get(
                "custom_noise", self.options.get("custom_noise")
            )
            noise_sampler = ss.noise.make_caching_noise_sampler(
                custom_noise,
                ssampler.max_noise_samples(),
                ss.sigma,
                ss.sigma_next,
                immiscible=fallback(ssampler.immiscible, ss.noise.immiscible),
            )
            ssampler.noise_sampler = noise_sampler
            for subidx in range(ssampler.substeps):
                pbar.set_description(f"{ssampler.name}: {substep + 1}/{substeps}")
                sr = self.simple_substep(x, ssampler)
                z_avg += renoise_weight * sr.x
                if sr.noise_scale != 0 and ss.sigma_next != 0:
                    noise_total += renoise_weight * sr.noise_scale
                    noise += renoise_weight * sr.get_noise(ss=ss)
                substep += 1
                ss.substep = substep
                pbar.update(1)

        noise = ss.noise.scale_noise(
            noise,
            noise_total * self.options.get("s_noise", 1.0),
            normalized=True,
        )
        return self.merge_steps(
            x, z_avg, noise=None if noise_total == 0 else noise, denoised=ss.denoised
        )


# class AverageMergeSubstepsSampler(NormalMergeSubstepsSampler):
#     name = "average"

#     def __init__(self, ss, sitems, *, avgmerge_stretch=0.4, **kwargs):
#         super().__init__(ss, sitems, **kwargs)
#         self.stretch = avgmerge_stretch

#     def step_max_noise_samples(self):
#         return sum(
#             1 + (2 + sampler.self_noise) * sampler.substeps for sampler in self.samplers
#         )

#     def step(self, x):
#         ss = orig_ss = self.ss
#         substeps = self.substeps
#         renoise_weight = 1.0 / substeps
#         z_avg = torch.zeros_like(x)
#         noise = torch.zeros_like(x)
#         stretch = (ss.sigma - ss.sigma_next) * self.stretch
#         sig_adj = ss.sigma + stretch
#         ss = self.ss.clone_edit(sigma=sig_adj)
#         orig_x = x
#         stretch_strength = stretch * ss.s_noise
#         if stretch_strength != 0:
#             noise_sampler = ss.noise.make_caching_noise_sampler(
#                 self.options.get("custom_noise"), 1, orig_ss.sigma, ss.sigma_next
#             )
#             x = x + (
#                 noise_sampler(orig_ss.sigma, ss.sigma_next).mul_(stretch * ss.s_noise)
#             )
#         self.ss.denoised = ss.denoised = ss.model(x, sig_adj)
#         noise_total = 0.0
#         substep = 0
#         for idx, ssampler in enumerate(self.samplers):
#             print(
#                 f"  SUBSTEP {substep + 1} .. {substep + ssampler.substeps}: {ssampler.name}, stretch={stretch}"
#             )
#             custom_noise = ssampler.options.get(
#                 "custom_noise", self.options.get("custom_noise")
#             )
#             noise_sampler = ss.noise.make_caching_noise_sampler(
#                 custom_noise,
#                 ssampler.substeps
#                 + (0 if ss.sigma_next == 0 else ssampler.max_noise_samples()),
#                 ss.sigma,
#                 ss.sigma_next,
#             )
#             ssampler.noise_sampler = noise_sampler
#             for sidx in range(ssampler.substeps):
#                 curr_x = orig_x + noise_sampler(sig_adj, ss.sigma_next).mul_(stretch)
#                 sr = self.simple_substep(curr_x, ssampler, ss=ss)
#                 z_avg += renoise_weight * sr.x
#                 noise_strength = sr.noise_scale
#                 if ss.sigma_next == 0 or noise_strength == 0:
#                     continue
#                 if noise_strength != 0 and ss.sigma_next != 0:
#                     noise_curr = sr.get_noise()
#                     noise_total += noise_strength.item() * renoise_weight
#                     noise += noise_curr
#                 substep += 1
#             substep += ssampler.substeps
#         return self.merge_steps(
#             x,
#             z_avg,
#             noise=None
#             if not noise_total
#             else ss.noise.scale_noise(noise, noise_total * ss.s_noise, normalized=True),
#             ss=ss,
#         )


# class SampleMergeSubstepsSampler(AverageMergeSubstepsSampler):
#     name = "sample"
#     cache_model = True

#     def __init__(self, ss, sitems, *, merge_sampler=None, **kwargs):
#         super().__init__(ss, sitems, **kwargs)
#         if merge_sampler is None:
#             merge_sampler = STEP_SAMPLERS["euler"](step_method="euler")
#         else:
#             msitem = merge_sampler.items[0]
#             merge_sampler = STEP_SAMPLERS[msitem["step_method"]](**msitem)
#         self.merge_sampler = merge_sampler
#         self.merge_ss = None

#     def step(self, x):
#         ss = self.ss
#         substeps = self.substeps
#         renoise_weight = 1.0 / substeps
#         z_avg = torch.zeros_like(x)
#         curr_x = x
#         ss.denoised = None
#         stretch = (ss.sigma - ss.sigma_next) * self.stretch
#         sig_adj = ss.sigma + stretch
#         ss = self.ss.clone_edit(sigma=sig_adj)
#         step = 0
#         for idx, ssampler in enumerate(self.samplers):
#             print(
#                 f"  SUBSTEP {step + 1} .. {step + ssampler.substeps}: {ssampler.name}, stretch={stretch}"
#             )
#             custom_noise = ssampler.options.get(
#                 "custom_noise", self.options.get("custom_noise")
#             )
#             noise_sampler = ss.noise.make_caching_noise_sampler(
#                 custom_noise,
#                 ssampler.max_noise_samples() + ssampler.substeps,
#                 ss.sigma,
#                 ss.sigma_next,
#             )
#             ssampler.noise_sampler = noise_sampler
#             for sidx in range(ssampler.substeps):
#                 if idx + sidx == 0 or not self.cache_model:
#                     self.ss.denoised = ss.denoised = ss.model(
#                         curr_x,
#                         ss.sigma,
#                         # + ss.noise_sampler(sig_adj.sigma, ss.sigma_next) * stretch * ss.s_noise,
#                         # sig_adj,
#                     )
#                 curr_x = x + noise_sampler(sig_adj, ss.sigma_next).mul_(
#                     ssampler.s_noise * stretch
#                 )
#                 sr = self.simple_substep(curr_x, ssampler, ss=ss)
#                 z_avg += renoise_weight * sr.x
#                 curr_x = sr.noise_x(sr.x)
#             step += ssampler.substeps
#         return self.merge_steps(curr_x, z_avg)

#     def merge_steps(self, x, result):
#         ss = self.ss
#         ss.dhist.push(ss.denoised)
#         ss.denoised = None
#         ss.model.reset_cache()
#         msampler = self.merge_sampler
#         if self.merge_ss is None:
#             merge_ss = self.merge_ss = self.ss.clone_edit(
#                 denoised=result,
#                 dhist=History(x, 3),
#                 xhist=History(x, 2),
#                 s_noise=msampler.s_noise,
#                 eta=msampler.eta,
#             )
#         else:
#             merge_ss = self.merge_ss
#             merge_ss.denoised = result
#         merge_ss.update(self.ss.idx, step=self.ss.step)
#         final = merge_ss.sigma_next == 0
#         noise_sampler = merge_ss.noise.make_caching_noise_sampler(
#             msampler.options.get("custom_noise", self.options.get("custom_noise")),
#             msampler.max_noise_samples() + int(not final),
#             merge_ss.sigma,
#             merge_ss.sigma_next,
#         )
#         msampler.noise_sampler = noise_sampler
#         sr = self.simple_substep(x, msampler, ss=merge_ss)
#         self.ss.callback(sr.x)
#         sr.noise_x()
#         merge_ss.dhist.push(result)
#         merge_ss.xhist.push(sr.x)
#         merge_ss.denoised = None
#         ss.xhist.push(sr.x)
#         return sr.x

#     def reset(self):
#         if self.merge_ss is None:
#             return
#         self.merge_ss.reset()
#         self.merge_ss.sigmas = self.ss.sigmas
#         self.merge_ss.update(self.ss.idx, step=self.ss.step)


# class SampleUncachedMergeSubstepsSampler(SampleMergeSubstepsSampler):
#     name = "sample_uncached"
#     cache_model = False


class DivideMergeSubstepsSampler(MergeSubstepsSampler):
    name = "divide"

    def __init__(self, ss, group, *, schedule_multiplier=4, **kwargs):
        super().__init__(ss, group, **kwargs)
        self.schedule_multiplier = schedule_multiplier

    def make_schedule(self, ss):
        max_steps = len(self.ss.sigmas) - 1
        sigmas_slice = ss.sigmas[
            ss.idx : min(max_steps + 1, ss.idx + self.schedule_multiplier)
        ]
        unsorted_idx = utils.find_first_unsorted(sigmas_slice)
        if unsorted_idx is not None:
            sigmas_slice = sigmas_slice[:unsorted_idx]
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
        return torch.cat(chunks)

    def step(self, x):
        ss = self.ss
        subss = self.ss.clone_edit(idx=0, sigmas=self.make_schedule(ss))
        subss.main_idx = ss.idx
        subss.main_sigmas = ss.sigmas
        substep = 0
        pbar = tqdm.tqdm(total=self.substeps, initial=0, disable=ss.disable_status)
        for ssampler in self.samplers:
            custom_noise = ssampler.options.get(
                "custom_noise", self.options.get("custom_noise")
            )
            noise_sampler = ss.noise.make_caching_noise_sampler(
                custom_noise,
                ssampler.max_noise_samples(),
                ss.sigma,
                ss.sigma_next,
                immiscible=fallback(ssampler.immiscible, ss.noise.immiscible),
            )
            ssampler.noise_sampler = noise_sampler
            for subidx in range(ssampler.substeps):
                subss.update(substep, substep=substep)
                pbar.set_description(
                    f"substep({ssampler.name}): {subss.sigma.item():.03} -> {subss.sigma_next.item():.03}"
                )
                subss.hist.push(subss.model(x, subss.sigma, ss=subss))
                subss.refs = FilterRefs.from_ss(subss, have_current=True)
                if substep == 0:
                    self.callback(ss=subss)
                sr = self.simple_substep(x, ssampler, ss=subss)
                x = sr.x
                noise_strength = sr.noise_scale
                if noise_strength != 0 and subss.sigma_next != 0:
                    x = sr.noise_x(ss=subss)
                substep += 1
                pbar.update(1)
        pbar.update(0)
        return x


class OvershootMergeSubstepsSampler(MergeSubstepsSampler):
    name = "overshoot"

    def __init__(
        self,
        ss,
        group,
        *,
        overshoot_expand_steps=1,
        restart_custom_noise=None,
        restart=None,
        **kwargs,
    ):
        super().__init__(ss, group, **kwargs)
        self.overshoot_expand_steps = overshoot_expand_steps
        restart = fallback(restart, {})
        self.restart = Restart(
            s_noise=restart.get("s_noise", 1.0),
            custom_noise=restart_custom_noise,
            immiscible=restart.get("immiscible", False),
        )

    def make_schedule(self, ss):
        expand = self.overshoot_expand_steps
        if expand > self.substeps:
            raise ValueError(
                "overshoot_expand_steps > substeps: can't make it to the end of step 1"
            )
        if expand < 2:
            return ss.sigmas, ss.idx
        sigmas_cpu = ss.sigmas.cpu()
        sigmas = torch.cat(
            tuple(
                torch.linspace(f, t, expand + 1)[:-1]
                for f, t in torch.stack((sigmas_cpu[:-1], sigmas_cpu[1:]), dim=1)
            )
            + (sigmas_cpu[-1].unsqueeze(0),)
        )
        return sigmas.to(ss.sigmas), ss.idx * expand

    def step(self, x):
        ss = self.ss
        sigmas, sigidx = self.make_schedule(ss)
        subss = ss.clone_edit(idx=sigidx, sigmas=sigmas)
        subss.hist = subss.hist.clone()
        substep = 0
        pbar = tqdm.tqdm(total=self.substeps, initial=0, disable=ss.disable_status)
        max_idx = len(subss.sigmas) - 2
        last_down = None
        for ssampler in self.samplers:
            custom_noise = ssampler.options.get(
                "custom_noise", self.options.get("custom_noise")
            )
            noise_sampler = ss.noise.make_caching_noise_sampler(
                custom_noise,
                ssampler.max_noise_samples(),
                ss.sigma,
                ss.sigma_next,
                immiscible=fallback(ssampler.immiscible, ss.noise.immiscible),
            )
            ssampler.noise_sampler = noise_sampler
            for subidx in range(ssampler.substeps):
                subss.update(subss.idx + substep, substep=substep)
                pbar.set_description(
                    f"substep({ssampler.name}): {subss.sigma.item():.03} -> {subss.sigma_next.item():.03}"
                )
                subss.hist.push(subss.model(x, subss.sigma, ss=subss))
                subss.refs = FilterRefs.from_ss(subss, have_current=True)
                if substep == 0:
                    ss.hist.push(subss.hcur)
                    self.callback(ss=subss)
                sr = self.simple_substep(x, ssampler, ss=subss)
                x = sr.x
                noise_strength = sr.noise_scale
                if noise_strength != 0 and subss.sigma_next != 0:
                    x = sr.noise_x(ss=subss)
                substep += 1
                pbar.update(1)
                last_down = subss.sigma_next.item()
                if subss.idx + substep >= max_idx:
                    break
            if subss.idx >= max_idx:
                break
        if last_down is not None and last_down < ss.sigma_next:
            restart_ns = self.restart.get_noise_sampler(ss.noise)
            x += ss.noise.scale_noise(
                restart_ns(refs=ss.refs),
                self.restart.get_noise_scale(last_down, ss.sigma_next),
            )
        pbar.update(0)
        return x


MERGE_SUBSTEPS_CLASSES = {
    "default (simple)": SimpleSubstepsSampler,
    "normal": NormalMergeSubstepsSampler,
    "divide": DivideMergeSubstepsSampler,
    "overshoot": OvershootMergeSubstepsSampler,
    # "average": AverageMergeSubstepsSampler,
    # "sample": SampleMergeSubstepsSampler,
    # "sample_uncached": SampleUncachedMergeSubstepsSampler,
    "simple": SimpleSubstepsSampler,
}
