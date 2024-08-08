import os

import torch
import numpy as np

from . import expression as expr
from . import latent

from .external import MODULES as EXT
from .utils import scale_noise, resolve_value

ALLOW_UNSAFE = os.environ.get("COMFYUI_OCS_ALLOW_UNSAFE_EXPRESSIONS") is not None
ALLOW_ALL_UNSAFE = os.environ.get("COMFYUI_OCS_ALLOW_ALL_UNSAFE") is not None

EXT_BLEH = EXT.get("bleh")
EXT_SONAR = EXT.get("sonar")

if "bleh" in EXT:
    BLENDING_MODES = EXT_BLEH.latent_utils.BLENDING_MODES
else:
    BLENDING_MODES = {
        "lerp": lambda a, b, t: (1 - t) * a + t * b,
    }

HANDLERS = {}


class NormHandler(expr.BaseHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numeric("factor", 1.0),
        expr.Arg.numscalar_sequence("dim", (-3, -2, -1)),
    )

    def handle(self, obj, getter):
        tensor, factor, dim = self.safe_get_all(obj, getter)
        return scale_noise(tensor, factor, normalize_dims=dim)

    validate_output = expr.Arg.tensor("output")


class MeanHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numscalar_sequence("dim", (-3, -2, -1)),
    )

    def handle(self, obj, getter):
        tensor, dim = self.safe_get_all(obj, getter)
        return tensor.mean(keepdim=True, dim=dim)


class StdHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numscalar_sequence("dim", (-3, -2, -1)),
    )

    def handle(self, obj, getter):
        tensor, dim = self.safe_get_all(obj, getter)
        return tensor.std(keepdim=True, dim=dim)


class RollHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numeric_scalar("amount", 0.5),
        expr.Arg.numscalar_sequence("dim", (-2,)),
    )

    def handle(self, obj, getter):
        tensor, amount, dim = self.safe_get_all(obj, getter)
        if isinstance(amount, float) and amount < 1.0 and amount > -1.0:
            if len(dim) > 1:
                raise ValueError(
                    "Cannot use percentage based amount with multiple roll dimensions",
                )
            amount = int(tensor.shape[dim[0]] * amount)
        return tensor.roll(amount, dims=dim)


class FlipHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.integer("dim"),
        expr.Arg.boolean("mirror", False),
    )

    def handle(self, obj, getter):
        tensor, dim, mirror = self.safe_get_all(obj, getter)
        if dim < 0:
            dim += tensor.ndim
        if dim < 0 or dim >= tensor.ndim:
            raise ValueError(
                f"Dimension out of range, wanted {dim}, tensor has {tensor.ndim} dimension(s)"
            )
        if not mirror:
            return torch.flip(tensor, (dim,))
        result = tensor.detach().clone()
        pivot = tensor.shape[dim] // 2
        out_slice = (
            np.s_[:] if d != dim else np.s_[pivot:] for d in range(tensor.ndim)
        )
        in_slice = (np.s_[:] if d != dim else np.s_[:pivot] for d in range(tensor.ndim))
        result[*out_slice] = torch.flip(tensor[*in_slice], dims=(dim,))
        return result


class BlendHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor1"),
        expr.Arg.tensor("tensor2"),
        expr.Arg.numeric("scale", 0.5),
        expr.Arg.string("mode", "lerp"),
    )

    def handle(self, obj, getter):
        t1, t2, scale, mode = self.safe_get_all(obj, getter)
        blend_handler = BLENDING_MODES.get(mode)
        if not blend_handler:
            raise KeyError(f"Unknown blend mode {mode!r}")
        return blend_handler(t1, t2, scale)


class ContrastAdaptiveSharpeningHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numeric("scale", 0.5),
    )

    def handle(self, obj, getter):
        t, scale = self.safe_get_all(obj, getter)
        return latent.contrast_adaptive_sharpening(t, scale)


class ScaleHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.one_of(
            "scale",
            (
                expr.ValidateArg.validate_numeric_scalar,
                expr.ValidateArg.validate_numscalar_sequence,
            ),
        ),
        expr.Arg.string("mode", "bicubic"),
        expr.Arg.boolean("absolute_scale", False),
    )

    def handle(self, obj, getter):
        t, scale, mode, abs_scale = self.safe_get_all(obj, getter)
        if isinstance(scale, (list, tuple)):
            if len(scale) != 2:
                raise ValueError(
                    "When passing scale as a tuple, it must be in the form (h, w)"
                )
        else:
            scale = (scale, scale)
        if abs_scale:
            scale = tuple(int(v) for v in scale)
        else:
            scale = (int(t.shape[-2] * scale[0]), int(t.shape[-1] * scale[1]))
        print("SCALE", t.shape[-2:], "->", scale)
        if not all(v > 0 for v in scale):
            raise ValueError(f"Invalid scale: scale values must be > 0, got: {scale!r}")
        return latent.scale_samples(t, scale[1], scale[0], mode=mode)


class NoiseHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.string("type", "gaussian"),
    )

    def handle(self, obj, getter):
        t, typ = self.safe_get_all(obj, getter)
        ctx = getter.ctx
        smin, smax, s, sn = (
            ctx.get_var(k) for k in ("sigma_min", "sigma_max", "sigma", "sigma_next")
        )
        ns = latent.get_noise_sampler(typ, t, smin, smax, normalized=False)
        return ns(s, sn)


class ShapeHandler(expr.BaseHandler):
    input_validators = (expr.Arg.tensor("tensor"),)

    def handle(self, obj, getter):
        t = self.safe_get("tensor", obj, getter)
        return expr.types.ExpTuple((*t.shape,))


class UnsafeTorchTensorMethodHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("__tensor"),
        expr.Arg.string("__method"),
    )

    if ALLOW_ALL_UNSAFE:

        class AlwaysContains:
            def __contains__(self, k):
                return True

        whitelist = AlwaysContains()
    elif ALLOW_UNSAFE:
        whitelist = {
            "abs",
            "absolute",
            "acos",
            "acosh",
            "add",
            "addbmm",
            "addcdiv",
            "addcmul",
            "addmm",
            "addmv",
            "addr",
            "adjoint",
            "all",
            "allclose",
            "amax",
            "amin",
            "aminmax",
            "angle",
            "any",
            "arccos",
            "arccosh",
            "arcsin",
            "arcsinh",
            "arctan",
            "arctan2",
            "arctanh",
            "argmax",
            "argmin",
            "argsort",
            "argwhere",
            "as_strided",
            "asin",
            "asinh",
            "atan",
            "atan2",
            "atanh",
            "baddbmm",
            "bernoulli",
            "bincount",
            "bitwise_and",
            "bitwise_left_shift",
            "bitwise_not",
            "bitwise_or",
            "bitwise_right_shift",
            "bitwise_xor",
            "bmm",
            "broadcast_to",
            "ceil",
            "cholesky",
            "cholesky_inverse",
            "cholesky_solve",
            "chunk",
            "clamp",
            "clip",
            "clone",
            "conj",
            "conj_physical",
            "contiguous",
            "copysign",
            "corrcoef",
            "cos",
            "cosh",
            "count_nonzero",
            "cov",
            "cross",
            "cummax",
            "cummin",
            "cumprod",
            "cumsum",
            "deg2rad",
            "det",
            "detach",
            "diag",
            "diag_embed",
            "diagflat",
            "diagonal",
            "diagonal_scatter",
            "diff",
            "digamma",
            "dim",
            "dist",
            "div",
            "divide",
            "dot",
            "dsplit",
            "eq",
            "equal",
            "erf",
            "erfc",
            "erfinv",
            "exp",
            "expand",
            "expand_as",
            "expm1",
            "fix",
            "flatten",
            "flip",
            "fliplr",
            "flipud",
            "float_power",
            "floor",
            "floor_divide",
            "fmax",
            "fmin",
            "fmod",
            "frac",
            "frexp",
            "gather",
            "gcd",
            "ge",
            "geqrf",
            "ger",
            "greater",
            "greater_equal",
            "gt",
            "hardshrink",
            "heaviside",
            "histc",
            "hsplit",
            "hypot",
            "i0",
            "igamma",
            "igammac",
            "index_add",
            "index_copy",
            "index_fill",
            "index_put",
            "index_reduce",
            "index_select",
            "inner",
            "inverse",
            "isclose",
            "isfinite",
            "isinf",
            "isnan",
            "isneginf",
            "isposinf",
            "kthvalue",
            "lcm()",
            "ldexp",
            "le",
            "lerp",
            "less",
            "less_equal",
            "lgamma",
            "log",
            "log10",
            "log1p",
            "log2",
            "logaddexp",
            "logaddexp2",
            "logcumsumexp",
            "logdet",
            "logical_and",
            "logical_not",
            "logical_or",
            "logical_xor",
            "logit",
            "logsumexp",
            "lt",
            "lu",
            "lu_solve",
            "masked_fill",
            "masked_scatter",
            "masked_select",
            "matmul",
            "matrix_exp",
            "max",
            "maximum",
            "mean",
            "median",
            "min",
            "minimum",
            "mm",
            "mode",
            "moveaxis",
            "movedim",
            "msort",
            "mul",
            "multinomial",
            "multiply",
            "mv",
            "mvlgamma",
            "nan_to_num",
            "nanmean",
            "nanmedian",
            "nanquantile",
            "nansum",
            "narrow",
            "narrow_copy",
            "ne",
            "neg",
            "negative",
            "new_empty",
            "new_full",
            "new_ones",
            "new_zeros",
            "nextafter",
            "nonzero",
            "norm",
            "not_equal",
            "numel",
            "orgqr",
            "ormqr",
            "outer",
            "permute",
            "polygamma",
            "positive",
            "pow",
            "prod",
            "qr",
            "quantile",
            "rad2deg",
            "ravel",
            "reciprocal",
            "remainder",
            "renorm",
            "repeat",
            "repeat_interleave",
            "reshape",
            "reshape_as",
            "resolve_conj",
            "resolve_neg",
            "roll",
            "rot90",
            "round",
            "rsqrt",
            "scatter",
            "scatter_add",
            "scatter_reduce",
            "select",
            "select_scatter",
            "sgn",
            "sigmoid",
            "sign",
            "signbit",
            "sin",
            "sinc",
            "sinh",
            "slice_scatter",
            "slogdet",
            "smm",
            "softmax",
            "sort",
            "sparse_mask",
            "split",
            "sqrt",
            "square",
            "squeeze",
            "sspaddmm",
            "std",
            "stft",
            "sub",
            "subtract",
            "sum",
            "sum_to_size",
            "svd",
            "swapaxes",
            "swapdims",
            "t",
            "take",
            "take_along_dim",
            "tan",
            "tanh",
            "tensor_split",
            "tile",
            "topk",
            "transpose",
            "triangular_solve",
            "tril",
            "triu",
            "true_divide",
            "trunc",
            "unflatten",
            "unfold",
            "unique",
            "unique_consecutive",
            "unsqueeze",
            "var",
            "vdot",
            "view",
            "view_as",
            "vsplit",
            "where",
            "xlogy",
        }
    else:
        whitelist = set()

    def handle(self, obj, getter):
        if "__method" in obj.kwargs or "__tensor" in obj.kwargs:
            raise ValueError(
                "Tensor method call doesn't support passing method or tensor with keyword args"
            )
        tensor = self.safe_get("__tensor", obj, getter=getter)
        method = self.safe_get("__method", obj, getter=getter)
        args = (
            self.safe_get(idx, obj, getter=getter) for idx in range(2, len(obj.args))
        )
        kwargs = {k: self.safe_get(k, obj, getter=getter) for k in obj.kwargs.keys()}
        if method not in self.whitelist:
            raise ValueError(f"Method {method} not whitelisted: cannot call")
        methodfun = getattr(tensor, method, None)
        if methodfun is None:
            raise KeyError(f"Unknown method {method} for Torch tensor")
        return methodfun(*args, **kwargs)


class UnsafeTorchHandler(expr.BaseHandler):
    input_validators = (expr.Arg.string("path"),)

    if not ALLOW_ALL_UNSAFE:

        def handle(self, obj, getter):
            raise ValueError("Unsafe Torch access not allowed")

    else:

        def handle(self, obj, getter):
            path = self.safe_get("path", obj, getter)
            keys = path.split(".")
            if not keys or not all(k for k in keys):
                raise ValueError(f"Bad path {path}")
            return resolve_value(keys, torch)


if EXT_BLEH:

    class BlehEnhanceHandler(expr.BaseHandler):
        input_validators = (
            expr.Arg.tensor("tensor"),
            expr.Arg.string("mode"),
            expr.Arg.numeric_scalar("scale", 1.0),
        )
        output_validator = expr.Arg.tensor("output")

        def handle(self, obj, getter):
            tensor, mode, scale = self.safe_get_all(obj, getter)
            return EXT_BLEH.latent_utils.enhance_tensor(
                tensor, mode, scale=scale, adjust_scale=False
            )

    HANDLERS["t_bleh_enhance"] = BlehEnhanceHandler()

if EXT_SONAR:

    class SonarPowerFilterHandler(expr.BaseHandler):
        input_validators = (
            expr.Arg.tensor("tensor"),
            expr.Arg.present("filter"),
        )
        output_validator = expr.Arg.tensor("output")

        default_power_filter = {
            "mix": 1.0,
            "normalization_factor": 1.0,
            "common_mode": 0.0,
            "channel_correlation": "1,1,1,1,1,1",
        }

        @classmethod
        def make_power_filter(cls, fdict, *, toplevel=True):
            fdict = fdict.copy()
            compose_with = fdict.pop("compose_with", None)
            if compose_with:
                if not isinstance(compose_with, dict):
                    raise TypeError("compose_with must be a dictionary")
                fdict["compose_with"] = cls.make_power_filter(
                    compose_with, toplevel=False
                )
            topargs = {
                k: fdict.pop(k, dv) for k, dv in cls.default_power_filter.items()
            }
            power_filter = EXT_SONAR.powernoise.PowerFilter(**fdict)
            if not toplevel:
                return power_filter
            cc = topargs.get("channel_correlation")
            if cc is not None:
                if not isinstance(cc, (list, tuple)) or not all(
                    isinstance(v, (int, float)) for v in cc
                ):
                    raise TypeError(
                        "Bad channel correlation type: must be comma separated string or numeric sequence"
                    )
                topargs["channel_correlation"] = ",".join(repr(v) for v in cc)
            return EXT_SONAR.powernoise.PowerNoiseItem(
                1, power_filter=power_filter, time_brownian=True, **topargs
            )

        def handle(self, obj, getter):
            tensor, filter_def = self.safe_get_all(obj, getter)
            if not isinstance(filter_def, dict):
                raise TypeError("filter argument must be a dictionary")
            power_filter = self.make_power_filter(filter_def)
            filter_rfft = power_filter.make_filter(tensor.shape).to(
                tensor.device, non_blocking=True
            )
            ns = power_filter.make_noise_sampler_internal(
                tensor,
                lambda *_unused, latent=tensor: latent,
                filter_rfft,
                normalized=False,
            )
            return ns(None, None)

    HANDLERS["t_sonar_power_filter"] = SonarPowerFilterHandler()

TENSOR_OP_HANDLERS = {
    "t_norm": NormHandler(),
    "t_mean": MeanHandler(),
    "t_std": StdHandler(),
    "t_blend": BlendHandler(),
    "t_roll": RollHandler(),
    "t_flip": FlipHandler(),
    "t_contrast_adaptive_sharpening": ContrastAdaptiveSharpeningHandler(),
    "t_scale": ScaleHandler(),
    "t_noise": NoiseHandler(),
    "t_shape": ShapeHandler(),
    "unsafe_tensor_method": UnsafeTorchTensorMethodHandler(),
    "unsafe_torch": UnsafeTorchHandler(),
}

HANDLERS |= TENSOR_OP_HANDLERS
