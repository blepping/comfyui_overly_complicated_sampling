import os

import numpy as np
import PIL.Image as PILImage
import torch

from . import expression as expr
from . import latent, unsafe_expression_whitelists
from .external import MODULES as EXT
from .latent import (
    OCSTAESD,
    DimCorrelationConfig,
    ImageBatch,
    normalize_to_scale,
    randomized_svd,
)
from .quantile_norm import quantile_normalize
from .utils import flip_tensor_range, resolve_value, scale_noise, softplus_soft_clamp

F = torch.nn.functional

ALLOW_UNSAFE = os.environ.get("COMFYUI_OCS_ALLOW_UNSAFE_EXPRESSIONS") is not None
ALLOW_ALL_UNSAFE = os.environ.get("COMFYUI_OCS_ALLOW_ALL_UNSAFE") is not None

EXT_BLEH = EXT_SONAR = None

BLENDING_MODES = {
    "lerp": torch.lerp,
}

HANDLERS = {}


def init_integrations(integrations):
    global EXT_BLEH, EXT_SONAR, BLENDING_MODES, HANDLERS
    EXT_BLEH = integrations.bleh
    EXT_SONAR = integrations.sonar
    if EXT_BLEH is not None:
        BLENDING_MODES = EXT_BLEH.latent_utils.BLENDING_MODES
        HANDLERS["t_bleh_enhance"] = BlehEnhanceHandler()
    if EXT_SONAR is not None:
        HANDLERS["t_sonar_power_filter"] = SonarPowerFilterHandler()
    if integrations.nnlatentupscale is not None:
        HANDLERS["t_scale_nnlatentupscale"] = ScaleNNLatentUpscaleHandler()


EXT.register_init_handler(init_integrations)


class UnaryTensorOpHandler(expr.BaseHandler):
    input_validators = (expr.Arg.tensor("tensor"),)

    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def handle(self, obj, getter):
        (tensor,) = self.safe_get_all(obj, getter)
        return self.handler(tensor)

    validate_output = expr.Arg.tensor("output")


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


class QuantileNormHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numeric("quantile", 0.75),
        expr.Arg.integer("dim", 1),
        expr.Arg.boolean("flatten", True),
        expr.Arg.numeric("norm_factor", 1.0),
        expr.Arg.numeric("norm_power", 0.5),
        expr.Arg.string("mode", "clamp"),
    )

    def handle(self, obj, getter):
        tensor, quantile, dim, flatten, norm_factor, norm_power, mode = (
            self.safe_get_all(obj, getter)
        )
        return quantile_normalize(
            tensor,
            quantile=quantile,
            dim=dim,
            flatten=flatten,
            nq_fac=norm_factor,
            pow_fac=norm_power,
            strategy=mode,
        )


class NormToScaleHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numeric("target_min", 0.0),
        expr.Arg.numeric("target_max", 1.0),
        expr.Arg.numscalar_sequence("dim", (-3, -2, -1)),
    )

    def handle(self, obj, getter):
        tensor, tmin, tmax, dim = self.safe_get_all(obj, getter)
        return normalize_to_scale(tensor, tmin, tmax, dim=dim)


class ClampHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numeric("min", 0.0),
        expr.Arg.numeric("max", 1.0),
    )

    def handle(self, obj, getter):
        tensor, tmin, tmax = self.safe_get_all(obj, getter)
        return torch.clamp(tensor, min=tmin, max=tmax)


class SoftClampHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numeric("min", 0.0),
        expr.Arg.numeric("max", 1.0),
        expr.Arg.numeric_scalar("stiffness", 1.0),
        expr.Arg.boolean("safe", True),
    )

    def handle(self, obj, getter):
        tensor, tmin, tmax, stiffness, safe = self.safe_get_all(obj, getter)
        return softplus_soft_clamp(tensor, tmin, tmax, stiffness=stiffness, safe=safe)


class StackHandler(NormHandler):
    input_validators = (
        expr.Arg.sequence("tensors", item_validator=expr.ValidateArg.validate_tensor),
        expr.Arg.integer("dim", 1),
    )

    def handle(self, obj, getter):
        tensors, dim = self.safe_get_all(obj, getter)
        return torch.stack(tensors, dim)


class CatHandler(StackHandler):
    def handle(self, obj, getter):
        tensors, dim = self.safe_get_all(obj, getter)
        return torch.cat(tensors, dim)


class ReshapeHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numscalar_sequence("shape"),
    )

    def handle(self, obj, getter):
        tensor, shape = self.safe_get_all(obj, getter)
        return torch.reshape(tensor.clone(), shape)


class SplitHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.integer("chunk_size"),
        expr.Arg.integer("dim"),
        expr.Arg.boolean("pad_last", False),
    )

    def handle(self, obj, getter):
        tensor, chunk_size, dim, pad_last = self.safe_get_all(obj, getter)
        chunks = torch.split(tensor, chunk_size, dim=dim)
        if not chunks or not pad_last or chunks[-1].shape == chunks[0].shape:
            return chunks
        dsize = chunks[0].shape[dim]
        replacement_chunk = tensor.new_zeros(chunks[0].shape)
        replacement_chunk[
            tuple(
                slice(None, dsize if d == dim else None) for d in replacement_chunk.ndim
            )
        ] = chunks[-1]
        return (*chunks[:-1], replacement_chunk)


class TrimHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numscalar_sequence("shape"),
        expr.Arg.boolean("flip", False),
    )

    def handle(self, obj, getter):
        tensor, shape, flip = self.safe_get_all(obj, getter)
        tshape = tensor.shape
        if tensor.ndim != len(shape):
            raise ValueError("Shape dimension count does not match tensor")
        slices = tuple(
            slice(None)
            if tsize <= dsize
            else (
                slice(None, min(dsize, tsize))
                if not flip
                else slice(-min(dsize, tsize))
            )
            for dsize, tsize in zip(shape, tshape, strict=True)
        )
        return tensor[slices]


class IndexedCopyHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor_dest"),
        expr.Arg.tensor("tensor_src"),
        expr.Arg.tensor_slice("slice"),
        expr.Arg.boolean("slice_src", default=True),
    )

    def handle(self, obj, getter):
        tensor1, tensor2, tensor_slice, slice_src = self.safe_get_all(obj, getter)
        result = tensor1.clone()
        result[tensor_slice] = tensor2[tensor_slice] if slice_src else tensor2
        return result


class NewLikeHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.nested_sequence(
            "values",
            default=(),
            item_validator=expr.ValidateArg.validate_numeric_scalar,
        ),
    )

    def handle(self, obj, getter):
        tensor, values = self.safe_get_all(obj, getter)
        return torch.tensor(values, dtype=tensor.dtype, device=tensor.device)


class MeanHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numscalar_sequence_or_single("dim", (-3, -2, -1)),
    )

    def handle(self, obj, getter):
        tensor, dim = self.safe_get_all(obj, getter)
        dim = dim if isinstance(dim, tuple) else (dim,)
        return tensor.mean(keepdim=True, dim=dim)


class StdHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numscalar_sequence_or_single("dim", (-3, -2, -1)),
        expr.Arg.numeric("eps", default=1e-07),
    )

    def handle(self, obj, getter):
        tensor, dim, eps = self.safe_get_all(obj, getter)
        dim = dim if isinstance(dim, tuple) else (dim,)
        std = tensor.std(keepdim=True, dim=dim)
        if eps != 0:
            std = std.clamp_min_(eps)
        return std


class RollHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numeric_scalar("amount", 0.5),
        expr.Arg.numscalar_sequence_or_single("dim", -2),
    )

    def handle(self, obj, getter):
        tensor, amount, dim = self.safe_get_all(obj, getter)
        if not isinstance(dim, tuple):
            dim = (dim,)
        if isinstance(amount, float) and amount < 1.0 and amount > -1.0:
            if len(dim) > 1:
                raise ValueError(
                    "Cannot use percentage based amount with multiple roll dimensions",
                )
            amount = int(tensor.shape[dim[0]] * amount)
        amount = (amount,) * len(dim)
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
        out_slice = tuple(
            np.s_[:] if d != dim else np.s_[pivot:] for d in range(tensor.ndim)
        )
        in_slice = tuple(
            np.s_[:] if d != dim else np.s_[:pivot] for d in range(tensor.ndim)
        )
        result[out_slice] = torch.flip(tensor[in_slice], dims=(dim,))
        return result


class CopySignHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.tensor("other"),
    )

    def handle(self, obj, getter):
        return torch.copysign(*self.safe_get_all(obj, getter))


class CloneHandler(NormHandler):
    input_validators = (expr.Arg.tensor("tensor"),)

    def handle(self, obj, getter):
        return self.safe_get("tensor", obj, getter).clone()


class NewFullHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numscalar_sequence("shape"),
        expr.Arg.numeric("value", 0.0),
    )

    def handle(self, obj, getter):
        tensor, shape, value = self.safe_get_all(obj, getter)
        return tensor.new_full(shape, value)


class InvertRangeHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.integer("dim"),
    )

    def handle(self, obj, getter):
        tensor, dim = self.safe_get_all(obj, getter)
        if dim < 0:
            dim += tensor.ndim
        if dim < 0 or dim >= tensor.ndim:
            raise ValueError(
                f"Dimension out of range, wanted {dim}, tensor has {tensor.ndim} dimension(s)"
            )
        return flip_tensor_range(tensor, dim=dim)


class MinHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.integer("dim"),
    )

    def handle(self, obj, getter):
        tensor, dim = self.safe_get_all(obj, getter)
        return tensor.min(dim=dim, keepdim=True).values


class MaxHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.integer("dim"),
    )

    def handle(self, obj, getter):
        tensor, dim = self.safe_get_all(obj, getter)
        return tensor.max(dim=dim, keepdim=True).values


class CumSumHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.integer("dim"),
    )

    def handle(self, obj, getter):
        tensor, dim = self.safe_get_all(obj, getter)
        return tensor.cumsum(dim=dim)


class MinimumHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor1"),
        expr.Arg.tensor("tensor2"),
    )

    def handle(self, obj, getter):
        tensor1, tensor2 = self.safe_get_all(obj, getter)
        return tensor1.minimum(tensor2)


class MaximumHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor1"),
        expr.Arg.tensor("tensor2"),
    )

    def handle(self, obj, getter):
        tensor1, tensor2 = self.safe_get_all(obj, getter)
        return tensor1.maximum(tensor2)


class MoveDimHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.integer("from_dim"),
        expr.Arg.integer("to_dim", -1),
    )

    def handle(self, obj, getter):
        tensor, from_dim, to_dim = self.safe_get_all(obj, getter)
        return tensor.movedim(from_dim, to_dim)


class PermuteHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numscalar_sequence("perms"),
        expr.Arg.boolean("reverse", default=False),
    )

    def handle(self, obj, getter):
        tensor, perms, reverse = self.safe_get_all(obj, getter)
        if not reverse:
            return tensor.permute(tuple(perms))
        inv_perms = torch.nn.utils.rnn.invert_permutation(
            torch.tensor(perms, device="cpu"),
        )
        if inv_perms is None:
            raise RuntimeError("Failed to calculate reverse permutation")
        return tensor.permute(tuple(inv_perms.tolist()))


class FlattenHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.integer("start_dim", 0),
        expr.Arg.integer("end_dim", -1),
    )

    def handle(self, obj, getter):
        tensor, start_dim, end_dim = self.safe_get_all(obj, getter)
        return tensor.flatten(start_dim=start_dim, end_dim=end_dim)


class MatmulHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor1"),
        expr.Arg.tensor("tensor2"),
    )

    def handle(self, obj, getter):
        tensor1, tensor2 = self.safe_get_all(obj, getter)
        return tensor1 @ tensor2


class BlendHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor1"),
        expr.Arg.tensor("tensor2"),
        expr.Arg.numeric("scale", 0.5),
        expr.Arg.one_of(
            "mode",
            (
                expr.ValidateArg.validate_string,
                expr.ValidateArg.validate_dict,
            ),
            default="lerp",
        ),
        expr.Arg.one_of(
            "blend_kwargs",
            (
                expr.ValidateArg.validate_none,
                expr.ValidateArg.validate_dict,
            ),
            default=None,
        ),
    )

    def handle(self, obj, getter):
        t1, t2, scale, mode, blend_kwargs = self.safe_get_all(obj, getter)
        blend_handler = BLENDING_MODES.get(mode)
        if not blend_handler:
            raise KeyError(f"Unknown blend mode {mode!r}")
        return blend_handler(
            t1,
            t2,
            scale,
            **({} if blend_kwargs is None else blend_kwargs),
        )


class ContrastAdaptiveSharpeningHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.numeric("scale", 0.5),
        expr.Arg.boolean("normalize", True),
    )

    def handle(self, obj, getter):
        t, scale, normalize = self.safe_get_all(obj, getter)
        return latent.contrast_adaptive_sharpening(t, scale, normalize=normalize)


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
            ctx.get_var(k, default=0.0)
            for k in ("sigma_min", "sigma_max", "sigma", "sigma_next")
        )
        ns = latent.get_noise_sampler(
            typ,
            t,
            smin,
            smax,
            normalized=False,
        )
        return ns(s, sn)


class ShapeHandler(expr.BaseHandler):
    input_validators = (expr.Arg.tensor("tensor"),)

    def handle(self, obj, getter):
        t = self.safe_get("tensor", obj, getter)
        return expr.types.ExpTuple((*t.shape,))


class NumelHandler(expr.BaseHandler):
    input_validators = (expr.Arg.tensor("tensor"),)

    def handle(self, obj, getter):
        t = self.safe_get("tensor", obj, getter)
        return t.numel()


class QRHandler(expr.BaseHandler):
    input_validators = (expr.Arg.tensor("tensor"),)

    def handle(self, obj, getter):
        tensor = self.safe_get("tensor", obj, getter)
        return tuple(torch.linalg.qr(tensor))


class SVDHandler(expr.BaseHandler):
    input_validators = (expr.Arg.tensor("tensor"),)

    def handle(self, obj, getter):
        tensor = self.safe_get("tensor", obj, getter)
        return tuple(torch.linalg.svd(tensor, full_matrices=False))


class RandomizedSVDHandler(expr.BaseHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.one_of(
            "base",
            (
                expr.ValidateArg.validate_none,
                expr.ValidateArg.validate_tensor,
            ),
            default=None,
        ),
        expr.Arg.integer("n_iter", default=6),
        expr.Arg.one_of(
            "rank",
            (
                expr.ValidateArg.validate_none,
                expr.ValidateArg.validate_integer,
            ),
            default=None,
        ),
    )

    def handle(self, obj, getter):
        tensor, base, n_iter, rank = self.safe_get_all(obj, getter)
        return randomized_svd(tensor, y=base, n_iter=n_iter, rank=rank)


class GaussianBlur2DHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.integer("kernel_size"),
        expr.Arg.numeric_scalar("sigma"),
    )

    def handle(self, obj, getter):
        return latent.gaussian_blur_2d(*self.safe_get_all(obj, getter))


class SNFGuidanceHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("t_tensor"),
        expr.Arg.tensor("s_tensor"),
        expr.Arg.integer("t_kernel_size", 3),
        expr.Arg.numeric_scalar("t_sigma", 1),
        expr.Arg.integer("s_kernel_size", 3),
        expr.Arg.numeric_scalar("s_sigma", 1),
    )

    def handle(self, obj, getter):
        return latent.snf_guidance(*self.safe_get_all(obj, getter))


class CorrelateHandler(NormHandler):
    input_validators = (
        expr.Arg.tensor("tensor1"),
        expr.Arg.one_of(
            "tensor2",
            (
                expr.ValidateArg.validate_none,
                expr.ValidateArg.validate_tensor,
            ),
            default=None,
        ),
        expr.Arg.one_of(
            "kwargs",
            (
                expr.ValidateArg.validate_none,
                expr.ValidateArg.validate_dict,
            ),
            default=None,
        ),
    )

    def handle(self, obj, getter):
        t1, t2, kwargs = self.safe_get_all(obj, getter)
        kwargs = kwargs.copy() if kwargs is not None else {}
        if "leave" not in kwargs:
            kwargs["leave"] = True
        return (
            DimCorrelationConfig.build(**kwargs)
            .get_correlation_order(x=t1, ref=t2)
            .reorder(t1)
            .clone()
        )


class RGBLatentHandler(expr.BaseHandler):
    input_validators = (
        expr.Arg.tensor("reference"),
        expr.Arg.numscalar_sequence("rgb"),
    )

    def handle(self, obj, getter):
        ctx = getter.ctx.constants.ctx
        model = ctx.get("model")
        if model is None:
            raise ValueError("Ohno")
        reference, rgb = self.safe_get_all(obj, getter)
        if len(rgb) != 3 or not all(0 <= v <= 1.0 for v in rgb):
            raise ValueError("Bad RGB parameter")
        img = torch.tensor(
            tuple(v * 2 - 1.0 for v in rgb),
            device=reference.device,
            dtype=reference.dtype,
        ).view(1, 3)
        latent = torch.zeros_like(reference).movedim(1, -1)
        latent += model.latent_format.rgb_to_latent(img)
        return latent.movedim(-1, 1)


class TAESDDecodeHandler(expr.BaseHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.string("mode", "sd15"),
    )
    validate_output = expr.Arg.image("output")

    def handle(self, obj, getter):
        t, mode = self.safe_get_all(obj, getter)
        return OCSTAESD.decode(mode, t)


class TAESDEncodeHandler(expr.BaseHandler):
    input_validators = (
        expr.Arg.image("image"),
        expr.Arg.tensor("reference_latent"),
        expr.Arg.string("mode", "sd15"),
    )
    validate_output = expr.Arg.tensor("output")

    def handle(self, obj, getter):
        imgbatch, ref, mode = self.safe_get_all(obj, getter)
        return OCSTAESD.encode(mode, imgbatch, ref)


class ImgShapeHandler(expr.BaseHandler):
    input_validators = (expr.Arg.tensor("image"),)

    def handle(self, obj, getter):
        imgbatch = self.safe_get("image", obj, getter)
        if len(imgbatch) == 0:
            raise ValueError("Can't get shape of empty image batch")
        isz = imgbatch[0].size
        return expr.types.ExpTuple((isz[1], isz[0]))


class ImgPILResizeHandler(expr.BaseHandler):
    input_validators = (
        expr.Arg.image("image"),
        expr.Arg.one_of(
            "size",
            (
                expr.ValidateArg.validate_numeric_scalar,
                expr.ValidateArg.validate_numscalar_sequence,
            ),
        ),
        expr.Arg.string("resample_mode", "bicubic"),
        expr.Arg.boolean("absolute_scale", False),
    )
    validate_output = expr.Arg.image("output")

    def handle(self, obj, getter):
        imgbatch, size, resample_mode, abs_scale = self.safe_get_all(obj, getter)
        if not isinstance(size, tuple):
            size = (size, size)
        if len(size) != 2 or not all(n > 0 for n in size):
            raise ValueError(
                "Image resize size parameter must be a positive non-zero number or tuple of positive non-zero height, width"
            )
        try:
            resample_mode = PILImage.Resampling[resample_mode.upper()]
        except KeyError:
            raise ValueError("Bad resample mode")
        if len(imgbatch) == 0:
            return imgbatch
        if abs_scale:
            size = tuple(int(v) for v in size)
        else:
            imgsize = imgbatch[0].size
            size = (int(imgsize[1] * size[0]), int(imgsize[0] * size[1]))
        new_size = (size[1], size[0])
        return ImageBatch(img.resize(new_size, resample_mode) for img in imgbatch)


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
        whitelist = unsafe_expression_whitelists.TORCH_FUNCTION_WHITELIST
    else:
        whitelist = frozenset()

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
            fdict["compose_with"] = cls.make_power_filter(compose_with, toplevel=False)
        topargs = {k: fdict.pop(k, dv) for k, dv in cls.default_power_filter.items()}
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


class ScaleNNLatentUpscaleHandler(expr.BaseHandler):
    input_validators = (
        expr.Arg.tensor("tensor"),
        expr.Arg.string("mode", "sd1"),
        expr.Arg.numeric_scalar("scale", 2.0),
    )
    output_validator = expr.Arg.tensor("output")

    def handle(self, obj, getter):
        tensor, mode, scale = self.safe_get_all(obj, getter)
        if mode not in {"sd1", "sdxl"}:
            raise ValueError(
                "Bad mode for t_scale_nnlatentupscale: must be either sd15 or sdxl"
            )
        return latent.scale_nnlatentupscale(mode, tensor, scale)


class ForkRngHandler(expr.BaseHandler):
    input_validators = (
        expr.Arg.present("expression"),
        expr.Arg.one_of(
            "seed",
            (
                expr.ValidateArg.validate_none,
                expr.ValidateArg.validate_integer,
            ),
            default=None,
        ),
        expr.Arg.boolean("enabled", default=True),
    )

    def handle(self, obj, getter):
        enabled = bool(self.safe_get("enabled", obj, getter))
        seed = self.safe_get("seed", obj, getter)
        with torch.random.fork_rng(enabled=enabled):
            if enabled and seed is not None:
                torch.manual_seed(seed)
            return self.safe_get("expression", obj, getter)


SKIP_TORCH_OPS = frozenset(
    (
        "fork_rng",
        "t_cat",
        "t_stack",
    )
)

TORCH_OP_HANDLERS = {
    "fork_rng": ForkRngHandler(),
    "t_blend": BlendHandler(),
    "t_cat": CatHandler(),
    "t_clamp": ClampHandler(),
    "t_soft_clamp": SoftClampHandler(),
    "t_clone": CloneHandler(),
    "t_contrast_adaptive_sharpening": ContrastAdaptiveSharpeningHandler(),
    "t_copysign": CopySignHandler(),
    "t_correlate": CorrelateHandler(),
    "t_cumsum": CumSumHandler(),
    "t_flatten": FlattenHandler(),
    "t_flip": FlipHandler(),
    "t_gaussianblur2d": GaussianBlur2DHandler(),
    "t_indexed_copy": IndexedCopyHandler(),
    "t_invert_range": InvertRangeHandler(),
    "t_matmul": MatmulHandler(),
    "t_max": MaxHandler(),
    "t_maximum": MaximumHandler(),
    "t_mean": MeanHandler(),
    "t_min": MinHandler(),
    "t_minimum": MinimumHandler(),
    "t_movedim": MoveDimHandler(),
    "t_new_like": NewLikeHandler(),
    "t_newfull": NewFullHandler(),
    "t_noise": NoiseHandler(),
    "t_norm": NormHandler(),
    "t_normalize_to_scale": NormToScaleHandler(),
    "t_normtoscale": NormToScaleHandler(),
    "t_numel": NumelHandler(),
    "t_quantilenorm": QuantileNormHandler(),
    "t_reshape": ReshapeHandler(),
    "t_rgb_latent": RGBLatentHandler(),
    "t_roll": RollHandler(),
    "t_scale": ScaleHandler(),
    "t_shape": ShapeHandler(),
    "t_permute": PermuteHandler(),
    "t_qr": QRHandler(),
    "t_svd": SVDHandler(),
    "t_randomized_svd": RandomizedSVDHandler(),
    "t_snf_guidance": SNFGuidanceHandler(),
    "t_split": SplitHandler(),
    "t_stack": StackHandler(),
    "t_std": StdHandler(),
    "t_taesd_decode": TAESDDecodeHandler(),
    "t_trim": TrimHandler(),
    "unsafe_tensor_method": UnsafeTorchTensorMethodHandler(),
    "unsafe_torch": UnsafeTorchHandler(),
}

TORCH_UOP_HANDLERS = {
    f"t_{k}": UnaryTensorOpHandler(handler=h)
    for k, h in (
        ("abs", torch.abs),
        ("acos", torch.acos),
        ("acosh", torch.acosh),
        ("atan", torch.atan),
        ("atanh", torch.atanh),
        ("ceil", torch.ceil),
        ("clone", torch.clone),
        ("cos", torch.cos),
        ("erf", torch.erf),
        ("erfinv", torch.erfinv),
        ("exp", torch.exp),
        ("expm1", torch.expm1),
        ("floor", torch.floor),
        ("frac", torch.frac),
        ("gelu", F.gelu),
        ("log", torch.log),
        ("log1p", torch.log1p),
        ("reciprocal", torch.reciprocal),
        ("relu", F.relu),
        ("remainder", torch.remainder),
        ("rsqrt", torch.rsqrt),
        ("sigmoid", torch.sigmoid),
        ("sign", torch.sign),
        ("sin", torch.sin),
        ("tan", torch.tan),
        ("tanh", torch.tanh),
        ("trunc", torch.trunc),
        ("diag_embed", torch.diag_embed),
    )
}

TORCH_OP_HANDLERS |= TORCH_UOP_HANDLERS  # ty: ignore[unsupported-operator]

TORCH_OP_HANDLERS |= {
    f"Tensor::{k.removeprefix('t_')}": v
    for k, v in TORCH_OP_HANDLERS.items()
    if k not in SKIP_TORCH_OPS
}

IMAGE_OP_HANDLERS = {
    "img_taesd_encode": TAESDEncodeHandler(),
    "img_shape": ImgShapeHandler(),
    "img_pil_resize": ImgPILResizeHandler(),
}

HANDLERS |= TORCH_OP_HANDLERS
HANDLERS |= IMAGE_OP_HANDLERS
