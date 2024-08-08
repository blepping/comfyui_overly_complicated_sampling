import collections

import torch

from . import expression as expr
from . import expression_handlers

from .external import MODULES as EXT
from .utils import fallback

OD = collections.OrderedDict

EXT_BLEH = EXT.get("bleh")
EXT_SONAR = EXT.get("sonar")

if "bleh" in EXT:
    BLENDING_MODES = EXT_BLEH.latent_utils.BLENDING_MODES
else:
    BLENDING_MODES = {
        "lerp": lambda a, b, t: (1 - t) * a + t * b,
    }

BLENDING_MODES = BLENDING_MODES | {
    "a_only": lambda a, b, t: a * t,
    "b_only": lambda a, b, t: b * t,
}

FILTER = {}


FILTER_HANDLERS = expr.HandlerContext(
    expr.BASIC_HANDLERS | expression_handlers.HANDLERS
)


class FilterRefs:
    def __init__(self, kvs=None):
        self.kvs = fallback(kvs, {})

    def get(self, k, default=None):
        return self.kvs.get(k, default)

    def __getitem__(self, k):
        return self.kvs[k]

    def __setitem__(self, k, v):
        self.kvs[k] = v

    def clone(self):
        return self.__class__(self.kvs.copy())

    def __or__(self, other):
        return self.__class__(self.kvs | other.kvs)

    def __ior__(self, other):
        self.kvs |= other.kvs
        return self

    def __delitem__(self, k):
        del self.kvs[k]

    def __contains__(self, k):
        return k in self.kvs

    def __missing__(self, k):
        return self.kvs.__missing__(k)

    def __len__(self):
        return len(self.kvs)

    def __iter__(self):
        return self.kvs.__iter__()

    @classmethod
    def from_ss(cls, ss, *, have_current=False):
        ms = ss.model.model_sampling
        fr = cls({
            "step": ss.step,
            "substep": ss.substep,
            "dt": ss.dt,
            "sigma_idx": ss.idx,
            "sigma": ss.sigma,
            "sigma_next": ss.sigma_next,
            "sigma_down": ss.sigma_down,
            "sigma_prev": ss.sigma_prev,
            "hist_len": len(ss.hist),
            "sigma_min": ms.sigma_min.item(),
            "sigma_max": ms.sigma_max.item(),
            "step_pct": float(ss.step / ss.total_steps),
            "total_steps": ss.total_steps,
            "sampling_pct": (999 - ms.timestep(ss.sigma).item()) / 999,
        })
        if have_current and len(ss.hist) > 0:
            fr |= cls.from_mr(ss.hcur)
            fr["d"] = ss.d
        if not have_current and len(ss.hist) > 0:
            hist_offs = -1
        elif len(ss.hist) > 1:
            hist_offs = -2
        else:
            hist_offs = None
        if hist_offs is not None:
            hprev = ss.hist[hist_offs]
            fr.kvs |= {f"{k}_prev": v for k, v in cls.from_mr(hprev).kvs.items()}
            fr["d_prev"] = hprev.d
        return fr

    @classmethod
    def from_mr(cls, mr):
        return cls({
            k: getattr(mr, ak)
            for k, ak in (
                ("cond", "denoised_cond"),
                ("denoised", "denoised"),
                ("model_call", "call_idx"),
                ("sigma", "sigma"),
                ("uncond", "denoised_uncond"),
                ("x", "x"),
            )
            if getattr(mr, ak, None) is not None
        })

    @classmethod
    def from_sr(cls, sr):
        return cls({
            k: getattr(sr, ak)
            for k, ak in (
                ("cond", "denoised_cond"),
                ("denoised", "denoised"),
                ("noise", "noise_pred"),
                ("sigma_down", "sigma_down"),
                ("sigma_next", "sigma_next"),
                ("sigma_up", "sigma_up"),
                ("sigma", "sigma"),
                ("step", "step"),
                ("substep", "substep"),
                ("uncond", "denoised_uncond"),
                ("x", "x"),
            )
            if getattr(sr, ak, None) is not None
        })


class Filter:
    name = "unknown"
    uses_ref = False
    default_options = {
        "enabled": True,
        "when": None,
        "input": "default",
        "output": "default",
        "ref": "default",
        "final": "default",
        "blend_mode": "lerp",
        "strength": 1.0,
    }

    def __init__(self, **options):
        self.options = options
        self.set_options(self.default_options)
        if self.when is not None:
            self.when = expr.Expression(self.when)
        for key in ("input", "output", "ref", "final"):
            if not self.uses_ref and key == "ref":
                continue
            val = getattr(self, key)
            val = make_filter(val) if isinstance(val, dict) else expr.Expression(val)
            setattr(self, key, val)
        if self.blend_mode not in BLENDING_MODES:
            raise ValueError("Bad blend mode")

    def set_options(self, defaults):
        for k, v in defaults.items():
            setattr(self, k, self.options.pop(k, v))

    def apply(self, input_latent, default_ref=None, refs=None, **kwargs):
        if not self.check_applies(refs):
            return input_latent
        refs = fallback(refs, FilterRefs()).clone()
        latent = self.get_ref("input", input_latent, self.input, refs=refs)
        refs["input"] = latent
        if not self.uses_ref:
            ref_latent = None
        else:
            ref_latent = (
                self.get_ref("ref", default_ref, self.ref, refs=refs)
                if default_ref is not None
                else None
            )
            refs["ref"] = ref_latent
        output_latent = self.get_ref(
            "output",
            self.filter(latent, ref_latent, refs=refs, **kwargs),
            self.output,
            refs=refs,
        )
        refs["output"] = output_latent
        return self.get_ref(
            "final",
            BLENDING_MODES[self.blend_mode](
                input_latent[: output_latent.shape[0]], output_latent, self.strength
            ),
            self.final,
            refs=refs,
        )

    def filter(self, latent, ref_latent, *, refs, **kwargs):
        raise NotImplementedError

    def check_applies(self, refs=None):
        if not self.enabled:
            return False
        if self.when is None:
            return True
        refs = fallback(refs, FilterRefs())
        matched = self.when.eval(FILTER_HANDLERS.clone(constants=refs, variables={}))
        # if matched:
        #     print("\nMATCH", self.name)
        return matched

    def get_ref(self, name, default_ref, ops, *, refs=None):
        if isinstance(ops, Filter):
            return ops.apply(default_ref, ops, refs=refs)
        drefs = FilterRefs({"default": default_ref})
        refs = drefs if refs is None else refs | drefs
        return ops.eval(FILTER_HANDLERS.clone(constants=refs, variables={}))


class SimpleFilter(Filter):
    name = "simple"

    def filter(self, latent, *args, **kwargs):
        return latent


class BlendFilter(Filter):
    name = "blend"
    default_options = Filter.default_options | {"filter1": None, "filter2": None}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not (isinstance(self.filter1, dict) and isinstance(self.filter2, dict)):
            raise ValueError("Must set filter1 and filter2")
        self.filter1 = make_filter(self.filter1)
        self.filter2 = make_filter(self.filter2)

    def filter(self, latent, ref_latent, *, refs, **kwargs):
        if self.blend_mode == "lerp":
            if self.strength == 0:
                return self.filter1(latent, ref_latent, refs=refs, **kwargs)
            if self.strength == 1:
                return self.filter2(latent, ref_latent, refs=refs, **kwargs)
        return BLENDING_MODES[self.blend_mode](
            self.filter1.apply(latent, ref_latent, refs=refs, **kwargs),
            self.filter2.apply(latent, ref_latent, refs=refs, **kwargs),
            self.strength,
        )


class ListFilter(Filter):
    name = "list"
    default_options = Filter.default_options | {"filters": ()}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.filters, (list, tuple)):
            raise ValueError("filters key must be a sequence")
        self.filters = tuple(make_filter(filt) for filt in self.filters)

    def filter(self, latent, ref_latent, *, refs, **kwargs):
        if not self.filters:
            return latent
        for filt in self.filters:
            latent = filt.apply(latent, ref_latent, refs=refs, **kwargs)
        return latent


class NormalizeFilter(Filter):
    name = "normalize"
    uses_ref = True
    default_options = Filter.default_options | {
        "adjust_target": 0,
        "balance_scale": 1.0,
        "adjust_scale": 1.0,
        "dims": (-2, -1),
    }

    def __init__(
        self,
        start_step=0,
        end_step=9999,
        phase="after",
        adjust_target=0,
        balance_scale=1.0,
        adjust_scale=1.0,
        dims=(-2, -1),
    ):
        self.start_step = start_step
        self.end_step = end_step
        self.phase = phase.lower().strip()  # before, after, all
        if isinstance(adjust_target, str):
            adjust_target = adjust_target.lower().strip()
            if adjust_target not in ("x",):
                raise ValueError("Bad target mean")
        # "x", scalar or array matching mean dims
        self.adjust_target = adjust_target
        # multiplier on adjustment, scalar or array matching mean dims
        self.adjust_scale = adjust_scale
        self.balance_scale = balance_scale
        self.dims = dims

    def __call__(self, ss, sigma, latent, phase, orig_x=None):
        if ss.step < self.start_step or ss.step > self.end_step:
            return latent
        if self.phase != "all" and phase != self.phase:
            return latent
        if self.adjust_target == "x" and orig_x is None:
            raise ValueError("Can only use source x in after phase")
        adjust_scale, balance_scale = (
            torch.tensor(v, dtype=latent.dtype).to(latent)
            if isinstance(v, (list, tuple))
            else v
            for v in (self.adjust_scale, self.balance_scale)
        )
        latent_mean = latent.mean(dim=self.dims, keepdim=True)
        # print("MEAN", latent_mean)
        latent = latent - latent_mean * balance_scale
        if self.adjust_target == "x":
            latent += orig_x.mean(dim=self.dims, keepdim=True) * adjust_scale
        elif isinstance(self.adjust_target, (list, tuple)):
            adjust_target = torch.tensor(self.adjust_target, dtype=latent.dtype).to(
                latent
            )
            latent += adjust_target * adjust_scale
        else:
            latent += self.adjust_target * adjust_scale
        return latent


class NormalizeFilter_:
    def __init__(
        self,
        start_step=0,
        end_step=9999,
        phase="after",
        adjust_target=0,
        balance_scale=1.0,
        adjust_scale=1.0,
        dims=(-2, -1),
    ):
        self.start_step = start_step
        self.end_step = end_step
        self.phase = phase.lower().strip()  # before, after, all
        if isinstance(adjust_target, str):
            adjust_target = adjust_target.lower().strip()
            if adjust_target not in ("x",):
                raise ValueError("Bad target mean")
        # "x", scalar or array matching mean dims
        self.adjust_target = adjust_target
        # multiplier on adjustment, scalar or array matching mean dims
        self.adjust_scale = adjust_scale
        self.balance_scale = balance_scale
        self.dims = dims

    def __call__(self, ss, sigma, latent, phase, orig_x=None):
        if ss.step < self.start_step or ss.step > self.end_step:
            return latent
        if self.phase != "all" and phase != self.phase:
            return latent
        if self.adjust_target == "x" and orig_x is None:
            raise ValueError("Can only use source x in after phase")
        adjust_scale, balance_scale = (
            torch.tensor(v, dtype=latent.dtype).to(latent)
            if isinstance(v, (list, tuple))
            else v
            for v in (self.adjust_scale, self.balance_scale)
        )
        latent_mean = latent.mean(dim=self.dims, keepdim=True)
        # print("MEAN", latent_mean)
        latent = latent - latent_mean * balance_scale
        if self.adjust_target == "x":
            latent += orig_x.mean(dim=self.dims, keepdim=True) * adjust_scale
        elif isinstance(self.adjust_target, (list, tuple)):
            adjust_target = torch.tensor(self.adjust_target, dtype=latent.dtype).to(
                latent
            )
            latent += adjust_target * adjust_scale
        else:
            latent += self.adjust_target * adjust_scale
        return latent


Normalize = NormalizeFilter

if EXT_BLEH:

    class BlehEnhanceFilter(Filter):
        name = "bleh_enhance"
        default_options = Filter.default_options | {
            "enhance_mode": None,
            "enhance_scale": 1.0,
        }

        def filter(self, latent, *args, **kwargs):
            if self.enhance_mode is None or self.enhance_scale == 1:
                return latent
            return EXT_BLEH.latent_utils.enhance_tensor(
                latent, self.enhance_mode, scale=self.enhance_scale, adjust_scale=False
            )

    class BlehOpsFilter(Filter):
        name = "bleh_ops"
        default_options = Filter.default_options | {"ops": ()}

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if isinstance(self.ops, (tuple, list)):
                self.ops = EXT_BLEH.nodes.ops.RuleGroup(
                    tuple(
                        r
                        for rs in self.ops
                        for r in EXT_BLEH.nodes.ops.Rule.from_dict(rs)
                    )
                )
                return
            if not isinstance(self.ops, str):
                raise ValueError("ops key must be a YAML string or list of object")
            self.ops = EXT_BLEH.nodes.ops.RuleGroup.from_yaml(self.ops)

        def filter(self, latent, ref_latent, *args, refs=None, **kwargs):
            if not self.ops:
                return latent
            refs = fallback(refs, {})
            bops = EXT_BLEH.nodes.ops
            state = {
                bops.CondType.TYPE: bops.PatchType.LATENT,
                bops.CondType.PERCENT: 0.0,
                bops.CondType.BLOCK: -1,
                bops.CondType.STAGE: -1,
                bops.CondType.STEP: refs.get("step", 0),
                bops.CondType.STEP_EXACT: refs.get("step", -1),
                "h": latent,
                "hsp": ref_latent,
                "target": "h",
            }
            self.ops.eval(state, toplevel=True)
            return state["h"]

    FILTER |= {
        "bleh_enhance": BlehEnhanceFilter,
        "bleh_ops": BlehOpsFilter,
    }

if EXT_SONAR:

    class SonarPowerFilter(Filter):
        name = "sonar_power_filter"
        default_options = Filter.default_options

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            power_filter = self.options.pop("power_filter", None)
            if power_filter is None:
                self.power_filter = None
                return
            if not isinstance(power_filter, dict):
                raise ValueError("power_filter key must be dict or null")
            self.power_filter = (
                expression_handlers.SonarPowerFilterHandler.make_power_filter(
                    power_filter
                )
            )

        def filter(self, latent, ref_latent, *args, refs=None, **kwargs):
            if not self.power_filter:
                return latent
            filter_rfft = self.power_filter.make_filter(latent.shape).to(
                latent.device, non_blocking=True
            )
            ns = self.power_filter.make_noise_sampler_internal(
                latent,
                lambda *_unused, latent=latent: latent,
                filter_rfft,
                normalized=False,
            )
            return ns(None, None)

    FILTER |= {"sonar_power_filter": SonarPowerFilter}


def make_filter(args):
    if not isinstance(args, dict):
        raise TypeError(f"Bad type for filter: {type(args)}")
    args = args.copy()
    filter_type = args.pop("filter_type", "simple")
    if not isinstance(filter_type, str):
        raise ValueError("Missing or invalid filter_type")
    filter_fun = FILTER.get(filter_type)
    if filter_fun is None:
        raise ValueError(f"Unknown filter_type: {filter_type}")
    return filter_fun(**args)


FILTER |= {
    "simple": SimpleFilter,
    "blend": BlendFilter,
    "list": ListFilter,
    "normalize": NormalizeFilter,
}
