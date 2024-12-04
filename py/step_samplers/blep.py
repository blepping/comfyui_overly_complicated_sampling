import typing

import inspect
import torch

import comfy

from .. import filtering
from .. import expression as expr
from ..utils import fallback
from .base import (
    StepSamplerContext,
    SingleStepSampler,
    registry,
)


class DynamicStep(SingleStepSampler):
    name = "dynamic"
    sample_sigma_zero = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dynamic = self.options.get("dynamic")
        if dynamic is None:
            raise ValueError(
                "Dynamic sampler type requires specifying dynamic block in text parameters"
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
        sampler_params = None
        handlers = filtering.FILTER_HANDLERS.clone(constants=self.ss.refs)
        for idx, (dyn_when, dyn_params) in enumerate(self.dynamic):
            if dyn_when is not None and not bool(dyn_when.eval(handlers)):
                continue
            sampler_params = dyn_params.eval(handlers)
            if sampler_params is not None:
                break
        if sampler_params is None:
            raise RuntimeError(
                "Dynamic sampler could not find matching sampler: all expressions failed to return a result"
            )
        if not isinstance(sampler_params, dict):
            raise TypeError(
                f"Dynamic sampler expression must evaluate to a dict, got type {type(sampler_params)}"
            )
        if bool(sampler_params.get("dynamic_inherit")):
            copy_keys = (
                "s_noise",
                "eta",
                "pre_filter",
                "post_filter",
                "immiscible",
            )
            opts = {k: getattr(self, k) for k in copy_keys}
        else:
            opts = {}
        opts["custom_noise"] = self.custom_noise
        opts |= sampler_params
        opts |= {k: v for k, v in self.options.items() if k.startswith("custom_noise_")}
        # print("\n\nDYN OPTS", opts)
        step_method = opts.get("step_method", "default")
        sampler_class = registry.STEP_SAMPLER_SIMPLE_NAMES.get(step_method)
        if sampler_class is None:
            raise ValueError(f"Unknown step method {step_method} in dynamic sampler")
        sampler = sampler_class(**opts)
        with StepSamplerContext(sampler, self.ss) as sampler:
            yield from sampler.step(x)


class AdapterStep(SingleStepSampler):
    name = "adapter"
    model_calls = 2
    immiscible = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.external_sampler = self.options.pop(
            "SAMPLER", comfy.samplers.sampler_object("euler")
        )
        sig = inspect.signature(self.external_sampler.sampler_function)
        self.external_sampler_options = {
            k: v
            for k, v in self.options.pop("external_sampler", {}).items()
            if k in sig.parameters
        }
        self.external_sampler_uses_noise = "noise_sampler" in sig.parameters
        self.ancestralize = self.options.pop("ancestralize", self.ancestralize) is True

    def step(self, x):
        ss = self.ss
        sigmas = ss.sigmas[ss.idx : ss.idx + 2]
        kwargs = {
            "callback": None,
            "disable": True,
            "extra_args": {"seed": ss.noise.seed + ss.noise.seed_offset},
        } | self.external_sampler_options
        if self.external_sampler_uses_noise:
            kwargs["noise_sampler"] = ss.noise.make_caching_noise_sampler(
                self.options.get("custom_noise"),
                1,
                sigmas[-1],
                sigmas[0],
                immiscible=fallback(self.immiscible, ss.noise.immiscible),
            )

        mcc = 1

        def model_wrapper(x_, sigma_, *args, **kwargs):
            nonlocal mcc
            if torch.equal(x_, x) and sigma_ == ss.sigma:
                return ss.hcur.denoised.clone()
            mr = self.call_model(x_, sigma_, *args, call_index=mcc, **kwargs)
            mcc += 1
            return mr.denoised.clone()

        result = self.external_sampler.sampler_function(
            model_wrapper, x.clone(), sigmas, **kwargs
        )
        yield from self.result(result, ss.sigma.new_zeros(1))


class CycleSingleStepSampler(SingleStepSampler):
    default_eta = 0.0

    def __init__(self, *, cycle_pct=0.25, **kwargs):
        super().__init__(**kwargs)
        if cycle_pct < 0:
            raise ValueError("cycle_pct must be positive")
        self.cycle_pct = cycle_pct

    def get_cycle_scales(self, sigma_next):
        keep_scale = 1 - self.cycle_pct
        add_scale = ((sigma_next**2.0 - (keep_scale * sigma_next) ** 2.0) ** 0.5) * (
            0.95 + 0.25 * self.cycle_pct
        )
        # print(f">> keep={keep_scale}, add={add_scale}")
        return keep_scale, add_scale


class EulerCycleStep(CycleSingleStepSampler):
    name = "blep_euler_cycle"
    allow_alt_cfgpp = True
    allow_cfgpp = True

    def step(self, x):
        sigma_next = self.ss.sigma_next
        denoised_pred, d = self.get_split_prediction()
        keep_scale, add_scale = self.get_cycle_scales(sigma_next)
        return (
            yield from self.split_result(
                denoised_pred, d * keep_scale, sigma_up=add_scale, sigma_down=sigma_next
            )
        )


class TrapezoidalCycleStep(CycleSingleStepSampler):
    name = "blep_trapezoidal_cycle"
    model_calls = 1
    allow_alt_cfgpp = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        blend_mode = self.options.get("blend_mode", "lerp").strip()
        self.blend = (
            filtering.BLENDING_MODES[blend_mode] if blend_mode != "lerp" else torch.lerp
        )

    def step(self, x):
        ss = self.ss
        sigma, sigma_next = ss.sigma, ss.sigma_next
        ratio = sigma_next / sigma
        dratio = 1 - (sigma / sigma_next) * 0.5

        # Denoised sample at the next sigma
        mr_next = self.call_model(
            self.blend(ss.denoised, x, ratio),
            ss.sigma_next,
            call_index=1,
        )

        keep_scale, add_scale = self.get_cycle_scales(ss.sigma_next)

        denoised_prime = self.blend(mr_next.denoised, ss.denoised, dratio)
        noise_pred = (x - denoised_prime).mul_(ratio * keep_scale)

        yield from self.result(
            denoised_prime.add_(noise_pred), add_scale, sigma_down=sigma_next
        )


class BASConfig(typing.NamedTuple):
    batch_multiplier: int = 2
    start_step: int = 0
    end_step: int = 3
    s_noise: float = 1.0
    eta: float = 0.0
    eta_retry_increment: float = 0
    denoised_factors: list | tuple | None = None
    denoised_factors_scale: float = 1.0
    denoised_multiplier: float = 1.0
    renoise_mode: str = "restart"
    fromstep_factor: float = 1.0
    tostep_factor: float = 1.0
    tostep_source: str = "dt"


# Batch augmented sampler
class BASStep(SingleStepSampler):
    name = "blep_bas"
    model_calls = -1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bas = BASConfig(**self.options.get("bas", {}))
        if self.bas.renoise_mode not in {"restart", "restart_noneta", "simple"}:
            raise ValueError("Bad BAS renoise mode")
        if self.bas.tostep_source not in {"dt", "sigma", "sigma_next"}:
            raise ValueError("Bad BAS tostep_source")
        blend_mode = self.options.get("blend_mode", "lerp").strip()
        self.blend = (
            filtering.BLENDING_MODES[blend_mode] if blend_mode != "lerp" else torch.lerp
        )

    def step(self, x):
        ss = self.ss
        bas = self.bas
        sigma, sigma_next = ss.sigma, ss.sigma_next
        eta = self.get_dyn_eta()
        sigma_down, sigma_up = self.get_ancestral_step(eta)
        denoised = ss.denoised
        ratio = sigma_down / sigma
        if (
            ss.step >= bas.end_step
            or ss.step < bas.start_step
            or bas.batch_multiplier < 1
        ):
            return (yield from self.result(self.blend(denoised, x, ratio), sigma_up))
        bsigma = sigma * bas.fromstep_factor
        if bas.tostep_source == "sigma":
            bsigma_next = sigma * bas.tostep_factor
        elif bas.tostep_source == "sigma_next":
            bsigma_next = sigma_next * bas.tostep_factor
        elif bas.tostep_source == "dt":
            bsigma_next = bsigma + (sigma_next - bsigma) * bas.tostep_factor
        else:
            raise RuntimeError("Impossible BAS tostep_source")
        if bsigma <= bsigma_next:
            raise ValueError("BAS: Bad configuration, got sigma <= sigma_next")
        bsigma_down, bsigma_up = self.get_ancestral_step(
            bas.eta,
            sigma=bsigma,
            sigma_next=bsigma_next,
            retry_increment=bas.eta_retry_increment,
        )
        bratio = bsigma_down / bsigma
        x_new = self.blend(denoised, x, bratio)
        batch_factor = bas.batch_multiplier
        if bas.denoised_factors is None:
            dn_factors = (bas.denoised_factors_scale / (batch_factor + 1),) * (
                batch_factor + 1
            )
            dn_sum = bas.denoised_factors_scale
        else:
            dn_factors = tuple(bas.denoised_factors)[: batch_factor + 1]
            tooshort = (batch_factor + 1) - len(dn_factors)
            if tooshort > 0:
                dn_factors = dn_factors + (dn_factors[-1],) * tooshort
            if len(dn_factors) != batch_factor + 1:
                raise ValueError("Bad length for bas_denoised_factors")
            dn_sum = sum(dn_factors)
            if bas.denoised_factors_scale != 0:
                dn_factors = tuple(
                    (f / dn_sum) * bas.denoised_factors_scale for f in dn_factors
                )
        # print(
        #     f"\nBAS STEP: step {bsigma} -> {bsigma_next} : down={bsigma_down}, up={bsigma_up}, bratio={bratio}, dn_factors={dn_factors}"
        # )
        if dn_sum == 0:
            raise ValueError("bas_denoised_factors must sum to a non-zero quantity")
        batch_size = x.shape[0]
        expanded_batch = batch_size * batch_factor
        x_expanded = x.new_zeros(expanded_batch, *x.shape[1:])
        renoise_mode = bas.renoise_mode
        if renoise_mode == "restart":
            noise_factor = (bsigma**2 - bsigma_down**2) ** 0.5
        elif renoise_mode == "restart_noneta":
            noise_factor = bsigma_up + (bsigma**2 - bsigma_next**2) ** 0.5
        else:
            noise_factor = bsigma_up + (bsigma - bsigma_next)
        for bidx in range(batch_factor):
            # print(
            #     f"NOISE ITER {bidx} -- {bidx * batch_size} -> {bidx * batch_size + batch_size}"
            # )
            x_expanded[
                bidx * batch_size : bidx * batch_size + batch_size
            ] = yield from self.result(
                x_new,
                noise_factor,
                s_noise=bas.s_noise,
                final=False,
            )
        s_in = x.new_ones(expanded_batch)
        del x_new
        mr_expanded = self.call_model(x_expanded, bsigma, s_in=s_in, call_index=1)
        denoised_expanded = mr_expanded.denoised
        denoised_new = torch.zeros_like(denoised)
        for obidx in range(batch_size):
            for bidx in range(-1, batch_factor):
                # print(f"DN ITER {obidx} <{bidx}> = dn_exp[{bidx * batch_size + obidx}]")
                dn_curr = (
                    denoised[obidx]
                    if bidx == -1
                    else denoised_expanded[bidx * batch_size + obidx]
                )
                denoised_new[obidx] += dn_curr * dn_factors[bidx + 1]
        denoised_new *= bas.denoised_multiplier
        result = self.blend(denoised_new, x, ratio)
        yield from self.result(result, sigma_up)


registry.add(
    BASStep,
    DynamicStep,
    AdapterStep,
    EulerCycleStep,
    TrapezoidalCycleStep,
)
