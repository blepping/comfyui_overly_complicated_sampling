import operator

import torch
import tqdm

from . import expression as expr
from . import utils

from .filtering import make_filter, FilterRefs, FILTER_HANDLERS
from .noise import ImmiscibleNoise
from .restart import Restart
from .step_samplers import STEP_SAMPLERS
from .step_samplers.base import StepSamplerContext
from .substep_sampling import StepSamplerChain
from .utils import check_time, fallback


class MergeSubstepsSampler:
    name = "unknown"

    def __init__(self, ss, group):
        samplers = tuple(
            STEP_SAMPLERS[sitem["step_method"]](**sitem) for sitem in group.items
        )
        options = group.options.copy()
        self.group = group
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
        self.require_uncond = any(sampler.require_uncond for sampler in samplers)
        self.cfg_scale_override = options.pop("cfg_scale_override", None)
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

    def substep(self, x, sampler):
        sg = sampler(x)
        yield from utils.step_generator(sg, get_next=lambda sr: sr.x)

    def simple_substep(self, x, sampler):
        for sr in self.substep(x, sampler):
            if not sr.final:
                raise RuntimeError("Unexpected non-final sampler result in substep!")
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

    def call_model(self, x, ss=None, sigma=None, **kwargs):
        ss = fallback(ss, self.ss)
        sigma = fallback(sigma, ss.sigma)
        return ss.call_model(
            x,
            sigma,
            ss=ss,
            cfg_scale_override=self.cfg_scale_override,
            require_uncond=self.require_uncond,
        )


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
        ss.hist.push(self.call_model(x))
        ss.refs = FilterRefs.from_ss(ss, have_current=True)
        self.callback()
        with StepSamplerContext(ssampler, ss) as ssampler:
            sr = self.simple_substep(x, ssampler)
        return self.merge_steps(sr.noise_x(ss=ss))


class SupremeAvgMergeSubstepsSampler(MergeSubstepsSampler):
    name = "supreme_avg"

    def step(self, x):
        ss = self.ss
        substeps = self.substeps
        renoise_weight = 1.0 / substeps
        z_avg = torch.zeros_like(x)
        noise = z_avg.clone()
        noise_total = 0.0
        substep = 0
        pbar = tqdm.tqdm(total=self.substeps, initial=1, disable=ss.disable_status)
        ss.hist.push(self.call_model(x))
        ss.refs = FilterRefs.from_ss(ss, have_current=True)
        self.callback()
        for ssampler_ in self.samplers:
            with StepSamplerContext(ssampler_, ss) as ssampler:
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
#                 + (0 if ss.sigma_next == 0 else ssampler.max_noise_samples),
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
#                 ssampler.max_noise_samples + ssampler.substeps,
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
#             msampler.max_noise_samples + int(not final),
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

    def __init__(self, ss, group, **kwargs):
        super().__init__(ss, group, **kwargs)
        self.schedule_multiplier = self.options.pop("schedule_multiplier", 4)

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
        for ssampler_ in self.samplers:
            with StepSamplerContext(ssampler_, subss) as ssampler:
                for subidx in range(ssampler.substeps):
                    subss.update(substep, substep=substep)
                    pbar.set_description(
                        f"substep({ssampler.name}): {subss.sigma.item():.03} -> {subss.sigma_next.item():.03}"
                    )
                    subss.hist.push(self.call_model(x, ss=subss))
                    subss.refs = FilterRefs.from_ss(subss, have_current=True)
                    if substep == 0:
                        self.callback(ss=subss)
                    sr = self.simple_substep(x, ssampler)
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
        **kwargs,
    ):
        super().__init__(ss, group, **kwargs)
        self.overshoot_expand_steps = self.options.pop("overshoot_expand_steps", 1)
        restart = self.options.pop("restart", {})
        restart_custom_noise = self.options.get("restart_custom_noise")
        if isinstance(restart_custom_noise, str):
            restart_custom_noise = self.options.get(
                f"restart_custom_noise_{restart_custom_noise}"
            )
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
        for ssampler_ in self.samplers:
            with StepSamplerContext(ssampler_, subss) as ssampler:
                for subidx in range(ssampler.substeps):
                    subss.update(subss.idx + substep, substep=substep)
                    pbar.set_description(
                        f"substep({ssampler.name}): {subss.sigma.item():.03} -> {subss.sigma_next.item():.03}"
                    )
                    subss.hist.push(self.call_model(x, ss=subss))
                    subss.refs = FilterRefs.from_ss(subss, have_current=True)
                    if substep == 0:
                        ss.hist.push(subss.hcur)
                        self.callback(ss=subss)
                    sr = self.simple_substep(x, ssampler)
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
                restart_ns(last_down, ss.sigma_next, refs=ss.refs),
                self.restart.get_noise_scale(last_down, ss.sigma_next),
            )
        pbar.update(0)
        return x


class LookaheadMergeSubstepsSampler(MergeSubstepsSampler):
    name = "lookahead"

    def __init__(self, ss, group, **kwargs):
        super().__init__(ss, group, **kwargs)
        lookahead = self.options.pop("lookahead", {}).copy()
        self.lookahead_eta = lookahead.pop("eta", 0.0)
        self.lookahead_s_noise = lookahead.pop("s_noise", 1.0)
        self.lookahead_dt_factor = lookahead.pop("dt_factor", 1.0)
        immiscible = lookahead.get("immiscible", False)
        self.immiscible = (
            ImmiscibleNoise(**immiscible) if immiscible is not False else False
        )

        self.custom_noise = self.options.get("custom_noise")
        if isinstance(self.custom_noise, str):
            self.custom_noise = self.options.get(f"custom_noise_{self.custom_noise}")

    def step(self, x):
        orig_x = x.clone()
        ss = self.ss
        subss = self.ss.clone_edit(idx=ss.idx, sigmas=ss.sigmas)
        substep = 0
        max_idx = len(ss.sigmas) - 1
        eff_substeps = min(max_idx - ss.idx, self.substeps)
        pbar = tqdm.tqdm(total=eff_substeps, initial=0, disable=ss.disable_status)
        for ssampler_ in self.samplers:
            substeps_remain = eff_substeps - substep
            if substeps_remain == 0:
                break
            with StepSamplerContext(ssampler_, subss) as ssampler:
                for subidx in range(min(substeps_remain, ssampler.substeps)):
                    subss.update(ss.idx + substep, substep=substep)
                    pbar.set_description(
                        f"substep({ssampler.name}): {subss.sigma.item():.03} -> {subss.sigma_next.item():.03}"
                    )
                    subss.hist.push(self.call_model(x, ss=subss))
                    subss.refs = FilterRefs.from_ss(subss, have_current=True)
                    if substep == 0:
                        self.callback(ss=subss)
                    sr = self.simple_substep(x, ssampler)
                    x = sr.x
                    noise_strength = sr.noise_scale
                    if noise_strength != 0 and subss.sigma_next != 0:
                        x = sr.noise_x(ss=subss)
                    substep += 1
                    pbar.update(1)
                    if substeps_remain == 1:
                        break
        pbar.update(0)
        sigma_down, sigma_up = ss.get_ancestral_step(
            eta=self.lookahead_eta, sigma=ss.sigma, sigma_next=ss.sigma_next
        )
        if sr.sigma_next == sigma_down:
            return x
        dt = (
            torch.sqrt(1.0 + (ss.sigma - sigma_down) ** 2) * 0.05
            + (ss.sigma - sigma_down) * 0.95
        ) * self.lookahead_dt_factor
        denoised = sr.denoised
        d = (orig_x - denoised) / ss.sigma
        x = orig_x + d * -dt
        if sigma_down == 0 or sigma_up == 0:
            return x
        noise_sampler = ss.noise.make_caching_noise_sampler(
            self.custom_noise,
            1,
            ss.sigma,
            ss.sigma_next,
            immiscible=fallback(self.immiscible, ss.noise.immiscible),
        )
        # FIXME: This sigma, sigma_next is probably wrong.
        x += ss.noise.scale_noise(
            noise_sampler(ss.sigma, ss.sigma_next, refs=ss.refs),
            sigma_up * self.lookahead_s_noise,
        )
        return x


class DynamicMergeSubstepsSampler(MergeSubstepsSampler):
    name = "dynamic"

    def __init__(self, ss, group, **kwargs):
        super().__init__(ss, group, **kwargs)
        dynamic = self.options.get("dynamic")
        if dynamic is None:
            raise ValueError(
                "Dynamic group type requires specifying dynamic block in text parameters"
            )
        if isinstance(dynamic, str):
            dynamic = ({"expression": dynamic},)
        elif not isinstance(dynamic, (tuple, list)):
            raise ValueError(
                "Bad type for dynamic block: must be string or list of objects"
            )
        elif len(dynamic) == 0:
            raise ValueError("Dynamic block as a list cannot be empty")
        dynresult = []
        for idx, item in enumerate(dynamic):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Bad item in dynamic block at index {idx}: must be a dict"
                )
            dyn_when = item.get("when")
            if isinstance(dyn_when, str):
                dyn_when = expr.Expression(dyn_when)
            elif dyn_when is not None:
                raise ValueError(
                    f"Unexpected type for when key in dynamic block at index {idx}, must be string or null/unset"
                )
            dyn_params = item.get("expression")
            if not isinstance(dyn_params, str):
                raise ValueError(
                    f"Missing or incorrectly typed expression key for dynamic block at index {idx}: must be a string"
                )
            dynresult.append((dyn_when, expr.Expression(dyn_params)))
        self.dynamic = tuple(dynresult)

    def step(self, x):
        group_params = None
        handlers = FILTER_HANDLERS.clone(constants=self.ss.refs)
        for idx, (dyn_when, dyn_params) in enumerate(self.dynamic):
            if dyn_when is not None and not bool(dyn_when.eval(handlers)):
                continue
            group_params = dyn_params.eval(handlers)
            if group_params is not None:
                break
        if group_params is None:
            raise RuntimeError(
                "Dynamic group could not find matching group: all expressions failed to return a result"
            )
        if not isinstance(group_params, dict):
            raise TypeError(
                f"Dynamic group expression must evaluate to a dict, got type {type(group_params)}"
            )
        if bool(group_params.get("dynamic_inherit")):
            copy_keys = ("preview_mode",)
            opts = {k: getattr(self, k) for k in copy_keys}
        else:
            opts = {}
        opts |= {
            k: v
            for k, v in self.options.items()
            if k.startswith("custom_noise") or k.startswith("restart_custom_noise")
        }
        opts |= group_params
        # print("\n\nDYN GROUP OPTS", opts)
        merge_method = opts.pop("merge_method", "simple").strip()
        if merge_method == "default":
            merge_method = "simple"
        group_class = MERGE_SUBSTEPS_CLASSES.get(merge_method)
        if group_class is None:
            raise ValueError(f"Unknown merge method {merge_method} in dynamic group")
        group = StepSamplerChain(
            merge_method=merge_method, items=self.group.items, **opts
        )
        sampler = group_class(self.ss, group)
        return sampler.step(x)


MERGE_SUBSTEPS_CLASSES = {
    "default (simple)": SimpleSubstepsSampler,
    "supreme_avg": SupremeAvgMergeSubstepsSampler,
    "divide": DivideMergeSubstepsSampler,
    "overshoot": OvershootMergeSubstepsSampler,
    "simple": SimpleSubstepsSampler,
    "lookahead": LookaheadMergeSubstepsSampler,
    "dynamic": DynamicMergeSubstepsSampler,
}
