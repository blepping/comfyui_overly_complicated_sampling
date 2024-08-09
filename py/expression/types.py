class Empty:
    def __bool__(self):
        return False


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

    def pop(self, *args, **kwargs):
        raise NotImplementedError

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

    popitem = pop
    update = pop
    clear = pop
    __delitem__ = pop
    __setitem__ = pop
    __ior__ = pop


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
    def __init__(self, obj, ctx, *args, **kwargs):
        self.obj = obj
        self.ctx = ctx
        self.args = args
        self.kwargs = kwargs

    def __call__(self, k, *, default=Empty):
        obj = self.obj
        result = (
            obj.kwargs.get_eval(k, self.ctx, *self.args, default=default, **self.kwargs)
            if isinstance(k, str)
            else obj.args.get_eval(k, self.ctx, *self.args, **self.kwargs)
        )
        if result is Empty:
            raise KeyError(f"Unknown key {k!r}")
        return result


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
        return handler(
            self, getter=ExprGetter(self, handlers, *args, **kwargs), **kwargs
        )

    def clone(self):
        return self.__class__(self.name, self.args.clone(), self.kwargs.clone())

    def pretty_string(self, depth=0):
        pad = " " * (depth + 1) * 2
        kwargs_str = f", {self.kwargs.pretty_string(depth + 1)}" if self.kwargs else ""
        return f"<FUNAP {self.name}\n{pad}{self.args.pretty_string(depth + 1)}{kwargs_str}\n{pad[:-2]}>"

    def __repr__(self):
        return (
            f"<FUNAP:{self.name}{self.args}{f", {self.kwargs}" if self.kwargs else ''}>"
        )


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
    "ExpBoundFunAp",
)
