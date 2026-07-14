class Empty:
    def __bool__(self):
        return False

class ExpReturn(Exception):
    pass

class ExpBase:
    def __bool__(self):
        return True

    def pretty_string(self, *, depth=0):
        return repr(self)

    def eval(self, *args, **kwargs):
        return self

    def clone(self, *, mapper=None):
        return self if not mapper else mapper(self)


class ExpOp(str, ExpBase):
    __slots__ = ()

    def eval(self, handlers, *args, **kwargs):
        value = handlers.get_var(self)
        if value is Empty:
            raise KeyError(f"No handler for op/var {self}")
        return value


class ExpBinOp(ExpOp):
    __slots__ = ()


class ExpSym(str, ExpBase):
    __slots__ = ()

    def __repr__(self):
        return f"'{self}"


class ExpTuple(tuple, ExpBase):
    __slots__ = ()

    def clone(self):
        return self.__class__(v.clone() if isinstance(ExpBase) else v for v in self)

    def get_eval(self, k, handlers, *args, default=None, **kwargs):
        val = super().__getitem__(k)
        if isinstance(val, ExpBase):
            return val.eval(handlers, *args, **kwargs)
        return val

    def pretty_string(self, depth=0):
        vals = (
            repr(v) if not isinstance(v, ExpBase) else v.pretty_string(depth=depth + 1)
            for v in self
        )
        pad = " " * (depth + 1) * 2
        nlpad = f",\n{pad}"
        return f"(\n{pad}{nlpad.join(vals)}\n{pad[:-2]})"

    def eval(self, handlers, *args, **kwargs):
        return tuple(
            v.eval(handlers, *args, **kwargs) if isinstance(v, ExpBase) else v
            for v in self
        )


class ExpKV(ExpBase):
    __slots__ = ("k", "v")

    def __init__(self, k, v):
        self.k = k
        self.v = v


class ExpDict(dict, ExpBase):
    __slots__ = ()

    def clone(self):
        return self.__class__(v.clone() if isinstance(ExpBase) else v for v in self)


    def get_eval(self, k, handlers, *args, default=Empty, **kwargs):
        val = super().get(k, default)
        if isinstance(val, ExpBase):
            return val.eval(handlers, *args, **kwargs)
        return val

    def pretty_string(self, depth=0):
        vals = (
            f"{k}: {v!r}"
            if not isinstance(v, ExpBase)
            else f"{k}: {v.pretty_string(depth=depth + 1)}"
            for k, v in self.items()
        )
        pad = " " * (depth + 1) * 2
        nlpad = f",\n{pad}"
        return f"{{\n{pad}{nlpad.join(vals)}\n{pad[:-2]}}}"

    def eval(self, handlers, *args, **kwargs):
        return {
            k: v.eval(handlers, *args, **kwargs) if isinstance(v, ExpBase) else v
            for k, v in self.items()
        }

    # Can't remember if there was a compelling reason ExpDict can't be mutable but
    # it breaks deep copy stuff.
    #
    # def pop(self, *args, **kwargs):
    #     raise NotImplementedError
    # popitem = pop
    # update = pop
    # clear = pop
    # __delitem__ = pop
    # __setitem__ = pop
    # __ior__ = pop


class ExpStatements(ExpBase):
    def __init__(self, statements):
        if not isinstance(statements, ExpTuple) or not len(statements):
            raise ValueError("Must have at least one statement")
        self.statements = statements

    def eval(self, handlers, *args, **kwargs):
        result = Empty
        for stmt in self.statements:
            result = (
                stmt.eval(handlers, *args, **kwargs)
                if isinstance(stmt, ExpBase)
                else stmt
            )
        return result

    def __repr__(self):
        return f"@{self.statements}"


class ExprGetter:
    def __init__(self, obj, ctx, args, kwargs, *, prepend_args=()):
        self.obj = obj
        self.ctx = ctx
        self.args = args
        self.prepend_args = prepend_args
        self.kwargs = kwargs

    def __call__(self, k, *, default=Empty):
        obj = self.obj
        if isinstance(k, str):
            result = obj.kwargs.get_eval(
                k, self.ctx, *self.args, default=default, **self.kwargs
            )
        elif isinstance(k, int):
            pa = self.prepend_args
            pa_len = len(pa)
            result = (
                pa[k]
                if k < pa_len
                else obj.args.get_eval(k, self.ctx, *self.args, **self.kwargs)
            )
        if result is Empty:
            raise KeyError(f"Unknown key {k!r}")
        return result


class ExpMethodAp(ExpBase):
    __slots__ = ("object_expression", "funap")

    def __init__(self, object_expression, funap):
        super().__init__()
        self.object_expression = object_expression
        self.funap = funap

    def eval(self, handlers, *args, **kwargs):
        object_value = self.object_expression.eval(handlers, *args, **kwargs)
        type_name = type(object_value).__name__
        handler_key = f"{type_name}::{self.funap.name}"
        handler = handlers.get_handler(handler_key)
        if handler is Empty:
            raise KeyError(f"No handler for method call op: {handler_key!r}")
        return handler(
            self,
            getter=ExprGetter(
                self.funap, handlers, args, kwargs, prepend_args=(object_value,)
            ),
            **kwargs,
        )

    def clone(self):
        return self.__class__(
            object_expression=self.object_expression.clone(),
            funap=self.funap.clone(),
        )
    __copy__ = clone

    def __getattr__(self, k):
        if k == "name":
            return f"method::{self.funap.name}"
        if k == "args":
            return self.funap.args
        if k == "kwargs":
            return self.funap.kwargs
        # This doesn't play well with deep copy.
        # if hasattr(self.funap, k):
        #     return getattr(self.funap, k)
        raise AttributeError(f"Can't get attribute {k}")

    def __repr__(self):
        return f"<METHAP:{self.object_expression}::{self.funap}>"


class ExpFunAp(ExpBase):
    __slots__ = ("name", "args", "kwargs")

    def __init__(self, name, args=None, kwargs=None):
        self.name = name
        self.args = args if args is not None else ExpTuple()
        self.kwargs = kwargs if kwargs is not None else ExpDict()

    def eval(self, handlers, *args, **kwargs):
        handler = handlers.get_handler(self.name)
        if handler is Empty:
            raise KeyError(f"No handler for op: {self.name!r}")
        return handler(self, getter=ExprGetter(self, handlers, args, kwargs), **kwargs)

    def clone(self):
        return self.__class__(self.name, self.args.clone(), self.kwargs.clone())

    def pretty_string(self, depth=0):
        pad = " " * (depth + 1) * 2
        kwargs_str = f", {self.kwargs.pretty_string(depth + 1)}" if self.kwargs else ""
        return f"<FUNAP {self.name}\n{pad}{self.args.pretty_string(depth + 1)}{kwargs_str}\n{pad[:-2]}>"

    def __repr__(self):
        kwargs_str = f", {self.kwargs}" if self.kwargs else ""
        return f"<FUNAP:{self.name}{self.args}{kwargs_str}>"


class ExpBoundFunAp(ExpFunAp):
    __slots__ = ("fun",)

    def __init__(self, name, fun, args, kwargs):
        super().__init__(name, args, kwargs)
        self.fun = fun

    def eval(self, handlers, *args, **kwargs):
        def get_evaled(k, default=None):
            return (
                self.kwargs.get_eval(k, handlers, *args, default=default, **kwargs)
                if isinstance(k, str)
                else self.args.get_eval(k, handlers, *args, **kwargs)
            )

        return self.fun(self.name, self.args, *args, getter=get_evaled, **kwargs)


__all__ = (
    "ExpBase",
    "ExpOp",
    "ExpBinOp",
    "ExpSym",
    "ExpTuple",
    "ExpKV",
    "ExpDict",
    "ExpFunAp",
    "ExpMethodAp",
    "ExpBoundFunAp",
)
