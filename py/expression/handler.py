import operator

from .validation import ValidateArg, Arg, ValidateError
from .types import Empty, ExpDict
from .util import torch


class HandlerError(Exception):
    pass


class BaseHandler:
    input_validators = ()

    def __init__(self):
        self.input_validators_by_key = {
            v.name: (idx, v) for idx, v in enumerate(self.input_validators)
        }

    def __call__(self, obj, *, getter):
        try:
            val = self.handle(obj, getter)
            return self.validate_output(obj, val)
        except Exception as exc:
            raise HandlerError(f'Error evaluating "{obj.name}":\n  {exc!r}') from exc

    def safe_get(self, key, obj, getter=None, *, default=Empty):
        str_key = isinstance(key, str)
        if str_key:
            argidx, validator = self.input_validators_by_key[key]
        else:
            argidx, validator = (
                key,
                (
                    self.input_validators[key]
                    if key < len(self.input_validators)
                    else None
                ),
            )
        default = (
            default
            if default is not Empty or validator is None
            else getattr(validator, "default", Empty)
        )
        if argidx < len(obj.args):
            eff_key = argidx
            str_eff_key = False
        elif str_key:
            eff_key = key
            str_eff_key = True
        else:
            raise ValidateError(
                f"Error validating input argument {key}, out of range for actual function arguments"
            )
        if getter is None:
            if str_eff_key:
                val = obj.kwargs.get(eff_key)
            else:
                val = default if eff_key > len(obj.args) else obj.args[eff_key]
        else:
            val = getter(eff_key, default=default)
        if validator is None:
            return val
        try:
            return validator(key, val)
        except ValidateError as exc:
            raise ValidateError(
                f"Error validating input argument {key}, type {type(val)}: {exc!r}"
            ) from None

    def safe_get_multi(self, keys, obj, getter=None, *, default=Empty):
        return (self.safe_get(k, obj, getter, default=default) for k in keys)

    def safe_get_all(self, obj, getter=None, *, default=Empty):
        return self.safe_get_multi(
            (v.name for v in self.input_validators), obj, getter, default=default
        )

    def handle(self, obj, getter):
        raise NotImplementedError

    def validate_output(self, obj, value):
        return value


class BinopLogicHandler(BaseHandler):
    input_validators = (
        Arg.present("lhs"),
        Arg.present("rhs"),
    )

    def validate_output(self, obj, value):
        return operator.truth(value)


class OrHandler(BinopLogicHandler):
    def handle(self, obj, getter):
        return operator.truth(
            self.safe_get("lhs", obj, getter=getter)
        ) or operator.truth(self.safe_get("rhs", obj, getter=getter))


class AndHandler(BinopLogicHandler):
    def handle(self, obj, getter):
        return operator.truth(
            self.safe_get("lhs", obj, getter=getter)
        ) and operator.truth(self.safe_get("rhs", obj, getter=getter))


class AllHandler(BinopLogicHandler):
    input_validators = ()

    def handle(self, obj, getter):
        return all(
            operator.truth(self.safe_get(idx, obj, getter=getter))
            for idx in range(len(obj.args))
        ) and all(
            operator.truth(self.safe_get(key, obj, getter=getter)) for key in obj.kwargs
        )


class AnyHandler(BinopLogicHandler):
    def handle(self, obj, getter):
        return any(
            operator.truth(self.safe_get(idx, obj, getter=getter))
            for idx in range(len(obj.args))
        ) or any(
            operator.truth(self.safe_get(key, obj, getter=getter)) for key in obj.kwargs
        )


class EqHandler(BinopLogicHandler):
    def handle(self, obj, getter):
        a1, a2 = self.safe_get_all(obj, getter)
        if isinstance(a1, torch.Tensor) and isinstance(a2, torch.Tensor):
            return torch.equal(a1, a2)
        return a1 == a2


class NeqHandler(BinopLogicHandler):
    def handle(self, *args, **kwargs):
        return not super().handle(*args, **kwargs)


class NotHandler(BinopLogicHandler):
    input_validators = (Arg.present("value"),)

    def handle(self, obj, getter):
        return not operator.truth(self.safe_get("value", obj, getter=getter))


class IfHandler(BaseHandler):
    input_validators = (
        Arg.present("condition"),
        Arg.present("then"),
        Arg.present("else"),
    )

    def handle(self, obj, getter):
        if operator.truth(self.safe_get("condition", obj, getter=getter)):
            return self.safe_get("then", obj, getter=getter)
        return self.safe_get("else", obj, getter=getter)


class BetweenHandler(BaseHandler):  # Inclusive
    input_validators = (
        Arg.numeric("value"),
        Arg.numeric("from", 0.0),
        Arg.numeric("to"),
    )

    def handle(self, obj, getter):
        value, low, high = self.safe_get_all(obj, getter)
        return low <= value <= high


class SimpleMathHandler(BaseHandler):
    input_validators = (Arg.numeric("lhs"), Arg.numeric("rhs"))

    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def validate_output(self, obj, value):
        return ValidateArg.validate_numeric(-1, value)

    def handle(self, obj, getter):
        args = (
            self.safe_get(idx, obj, getter=getter)
            for idx in range(len(self.input_validators))
        )
        return self.handler(*args)


class MinusHandler(SimpleMathHandler):
    input_validators = (Arg.numeric("lhs"), Arg.numeric("rhs", default=Empty))

    __init__ = BaseHandler.__init__

    def handle(self, obj, getter):
        lhs, rhs = self.safe_get_all(obj, getter)
        if rhs is Empty:
            return operator.neg(lhs)
        return operator.sub(lhs, rhs)


class RelComparisonHandler(SimpleMathHandler):
    def validate_output(self, obj, value):
        return operator.truth(value)


class UnarySimpleMathHandler(SimpleMathHandler):
    input_validators = (Arg.numeric("lhs"),)


class IsSetHandler(BaseHandler):
    input_validators = (Arg.string("name"),)

    def handle(self, obj, getter):
        key = self.safe_get(0, obj, getter=getter)
        return key in getter.handlers

    def validate_output(self, obj, value):
        return operator.truth(value)


class GetHandler(BaseHandler):
    input_validators = (
        Arg.string("name"),
        Arg.present("fallback"),
    )

    def handle(self, obj, getter):
        key = self.safe_get("name", obj, getter=getter)
        h = getter.handlers.get(key)
        if h is None:
            return self.safe_get("fallback", obj, getter=getter)
        return h(getter.handlers, *getter.args, **getter.kwargs)


class S_Handler(BaseHandler):
    input_validators = (
        Arg.integer("start", None),
        Arg.integer("end", None),
        Arg.integer("step", None),
    )

    def handle(self, obj, getter):
        return slice(*self.safe_get_all(obj, getter=getter))


class IndexHandler(BaseHandler):
    input_validators = (
        Arg.present("index"),
        Arg.one_of(
            "value", (ValidateArg.validate_sequence, ValidateArg.validate_tensor)
        ),
    )

    def handle(self, obj, getter):
        idx, value = self.safe_get_all(obj, getter=getter)
        return value[idx]


class MinHandler(BaseHandler):
    input_validators = (Arg.numscalar_sequence("values"),)

    def handle(self, obj, getter):
        return min(*self.safe_get("values", obj, getter))

    def validate_output(self, obj, value):
        return ValidateArg.validate_numeric(-1, value)


class MaxHandler(MinHandler):
    def handle(self, obj, getter):
        return max(*self.safe_get("values", obj, getter))


class UnsafeCallHandler(BaseHandler):
    input_validators = (Arg.present("__callable"),)

    def handle(self, obj, getter):
        if "__callable" in obj.kwargs:
            raise ValueError(
                "unsafe_call does not support passing the callable via keyword arg"
            )
        fun = self.safe_get("__callable", obj, getter)
        if not callable(fun):
            raise ValueError("Cannot call supplied value: not a callable")
        args = (self.safe_get(idx, obj, getter) for idx in range(1, len(obj.args)))
        kwargs = {k: self.safe_get(k, obj, getter) for k in obj.kwargs}
        return fun(*args, **kwargs)


class DictHandler(BaseHandler):
    def handle(self, obj, getter):
        if len(obj.args):
            raise ValueError("Non-KV items passed to dict constructor")
        return ExpDict({
            k: self.safe_get(k, obj, getter=getter) for k in obj.kwargs.keys()
        })


LOGIC_HANDLERS = {
    "||": OrHandler(),
    "&&": AndHandler(),
    "==": EqHandler(),
    "!=": NeqHandler(),
    "not": NotHandler(),
    "if": IfHandler(),
    "all": AllHandler(),
    "any": AnyHandler(),
}
for k, alias in (
    ("||", "or"),
    ("&&", "and"),
    ("==", "eq"),
    ("!=", "neq"),
):
    LOGIC_HANDLERS[alias] = LOGIC_HANDLERS[k]


MATH_HANDLERS = {
    "+": SimpleMathHandler(operator.add),
    "-": MinusHandler(),
    "*": SimpleMathHandler(operator.mul),
    "/": SimpleMathHandler(operator.truediv),
    "//": SimpleMathHandler(operator.floordiv),
    "**": SimpleMathHandler(operator.pow),
    "mod": SimpleMathHandler(operator.mod),
    "neg": UnarySimpleMathHandler(operator.neg),
    "between": BetweenHandler(),
    "<": RelComparisonHandler(operator.lt),
    "<=": RelComparisonHandler(operator.le),
    ">": RelComparisonHandler(operator.gt),
    ">=": RelComparisonHandler(operator.ge),
    "min": MinHandler(),
    "max": MaxHandler(),
}
for k, alias in (
    ("+", "add"),
    ("-", "sub"),
    ("*", "mul"),
    ("/", "div"),
    ("//", "idiv"),
    ("**", "pow"),
):
    MATH_HANDLERS[alias] = MATH_HANDLERS[k]

MISC_HANDLERS = {
    "is_set": IsSetHandler(),
    "get": GetHandler(),
    "index": IndexHandler(),
    "s_": S_Handler(),
    "unsafe_call": UnsafeCallHandler(),
    "dict": DictHandler,
}

BASIC_HANDLERS = LOGIC_HANDLERS | MATH_HANDLERS | MISC_HANDLERS
