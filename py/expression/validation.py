import contextlib
import functools

from ..latent import ImageBatch
from .util import torch
from .types import Empty


class Arg:
    __slots__ = ("name", "default", "validator")

    def __init__(self, name, default=Empty, *, validator=None):
        self.name = name
        self.default = default
        self.validator = validator

    def __call__(self, _key, value, *args, **kwargs):
        return self.validate(value, *args, **kwargs)

    def validate(self, value):
        if value is Empty:
            if self.default is Empty:
                raise ValueError(f"Missing value for argument {self.name}")
            return self.default
        try:
            return self.validator(self.name, value) if self.validator else value
        except ValidateError as exc:
            raise ValidateError(f"Failed to validate argument {self.name}: {exc}")

    @classmethod
    def tensor(cls, name):
        return cls(name, validator=ValidateArg.validate_tensor)

    @classmethod
    def image(cls, name):
        return cls(name, validator=ValidateArg.validate_image)

    @classmethod
    def numeric(cls, name, default=Empty):
        return cls(name, default=default, validator=ValidateArg.validate_numeric)

    @classmethod
    def numeric_scalar(cls, name, default=Empty):
        return cls(name, default=default, validator=ValidateArg.validate_numeric_scalar)

    @classmethod
    def integer(cls, name, default=Empty):
        return cls(name, default=default, validator=ValidateArg.validate_integer)

    @classmethod
    def numscalar_sequence(cls, name, default=Empty):
        return cls(
            name, default=default, validator=ValidateArg.validate_numscalar_sequence
        )

    @classmethod
    def tensor_slice(cls, name, default=Empty):
        return cls(name, default=default, validator=ValidateArg.validate_tensor_slice)

    @classmethod
    def sequence(cls, name, default=Empty, *, item_validator=None):
        return cls(
            name,
            default=default,
            validator=functools.partial(
                ValidateArg.validate_sequence, item_validator=item_validator
            ),
        )

    @classmethod
    def nested_sequence(cls, name, default=Empty, *, item_validator=None):
        return cls(
            name,
            default=default,
            validator=functools.partial(
                ValidateArg.validate_nested_sequence, item_validator=item_validator
            ),
        )

    @classmethod
    def string(cls, name, default=Empty):
        return cls(name, default=default, validator=ValidateArg.validate_string)

    @classmethod
    def boolean(cls, name, default=Empty):
        return cls(name, default=default, validator=ValidateArg.validate_boolean)

    @classmethod
    def present(cls, name):
        return cls(name, validator=ValidateArg.validate_passthrough)

    @classmethod
    def one_of(cls, name, validators, *, default=Empty):
        def validate(idx, val):
            for validator in validators:
                try:
                    return validator(idx, val)
                except ValidateError:
                    continue
            raise ValidateError(
                f"Failed to validate argument at {idx} of type {type(val)}"
            )

        return cls(name, default=default, validator=validate)


class ValidateError(Exception):
    pass


class ValidateArg:
    __slots__ = ("valfuns", "groupfun", "kwargs", "kwargslist")

    def __init__(self, name, *args, kwargslist=(), group=all, **kwargs):
        if not isinstance(name, (list, tuple)):
            name = (name,)
            args = ((args,),)
            kwargslist = kwargs
            kwargs = {}
        self.valfuns = (getattr(self, f"validate_{n}", None) for n in name)
        if not all(self.valfuns):
            raise ValueError("Unknown validator")
        self.groupfun = group
        self.kwargs = kwargs
        self.kwargslist = kwargslist if kwargslist is not None else {}

    def __call__(self, *args, **kwargs):
        kalen = len(self.kwargslist)
        return self.groupfun(
            vf(
                *args,
                **(self.kwargslist if idx < kalen else {}),
                **self.kwargs,
            )
            for idx, vf in enumerate(self.valfuns)
        )

    @staticmethod
    def validate_numeric(idx, val):
        if not isinstance(val, (int, float, torch.Tensor)):
            raise ValidateError(
                f"Expected numeric or tensor argument at {idx}, got {type(val)}"
            )
        return val

    @classmethod
    def validate_numeric_scalar(cls, idx, val):
        if not isinstance(val, (int, float)):
            raise ValidateError(f"Expected numeric argument at {idx}, got {type(val)}")
        return val

    @classmethod
    def validate_tensor_slice_item(cls, idx, val):
        with contextlib.suppress(ValidateError):
            ok = (
                val in {Ellipsis, None}
                or isinstance(val, (int, slice))
                or cls.validate_sequence(
                    idx, val, item_validator=ValidateArg.validate_integer
                )
            )
            if ok:
                return val
        raise ValidateError(
            f"Expected none, int, slice, tuple of int or ellipsis argument at {idx}, got {type(val)}"
        )

    @classmethod
    def validate_integer(cls, idx, val):
        if not isinstance(val, int):
            raise ValidateError(f"Expected integer argument at {idx}, got {type(val)}")
        return val

    @staticmethod
    def validate_tensor(idx, val):
        if not isinstance(val, torch.Tensor):
            raise ValidateError(f"Expected tensor argument at {idx}, got {type(val)}")
        return val

    @staticmethod
    def validate_image(idx, val):
        if not isinstance(val, ImageBatch):
            raise ValidateError(
                f"Expected PIL Image argument at {idx}, got {type(val)}"
            )
        return val

    @staticmethod
    def validate_sequence(idx, val, *, item_validator=None):
        if not isinstance(val, (list, tuple)):
            raise ValidateError(f"Expected sequence argument at {idx}, got {type(val)}")
        if item_validator is None:
            return val
        try:
            return tuple(item_validator(iidx, v) for iidx, v in enumerate(val))
        except ValidateError as exc:
            raise ValidateError(
                f"Item validation failed for sequence argument at {idx}: {exc}"
            )

    @classmethod
    def validate_nested_sequence(cls, idx, val, *, item_validator=None, depth=0):
        if not isinstance(val, (list, tuple)):
            raise ValidateError(
                f"Expected nested sequence argument at {idx}, depth {depth} but got {type(val)}"
            )
        try:
            return tuple(
                cls.validate_nested_sequence(
                    idx, v, item_validator=item_validator, depth=depth + 1
                )
                if isinstance(v, (list, tuple))
                else (item_validator(iidx, v) if item_validator is not None else v)
                for iidx, v in enumerate(val)
            )
        except ValidateError as exc:
            raise ValidateError(
                f"Item validation failed for nested sequence argument at {idx}, depth {depth}: {exc}"
            )

    @classmethod
    def validate_numscalar_sequence(cls, idx, val):
        return cls.validate_sequence(
            idx, val, item_validator=cls.validate_numeric_scalar
        )

    @classmethod
    def validate_tensor_slice(cls, idx, val):
        return cls.validate_sequence(
            idx, val, item_validator=cls.validate_tensor_slice_item
        )

    @classmethod
    def validate_string(cls, idx, val):
        if not isinstance(val, str):
            raise ValidateError(f"Expected string argument at {idx}, got {type(val)}")
        return val

    @classmethod
    def validate_boolean(cls, idx, val):
        if val is not True and val is not False:
            raise ValidateError(f"Expected boolean argument at {idx}, got {type(val)}")
        return val

    @classmethod
    def validate_none(cls, idx, val):
        if val is not None:
            raise ValidateError(f"Expected none argument at {idx}, got {type(val)}")
        return val

    @classmethod
    def validate_passthrough(cls, idx, val):
        return val
