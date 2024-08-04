import re
import operator

from .parser import Parser, ParserSpec, ParseError
from .types import (
    Empty,
    ExpBase,
    ExpOp,
    ExpBinOp,
    ExpSym,
    ExpStatements,
    ExpFunAp,
    ExpTuple,
    ExpDict,
    ExpKV,
)


class Expression:
    EXPR_RE = re.compile(
        r"""
    \s*
    (
          \d+             # Possibly negative numeric literal
            (?: \. \d* )?    # Floating point
            (?: e [+-] \d+)? # Scientific notation
        | (?: \*\* | // )    # Doubled operators
        | [<>]=?             # Relative comparison
        | [!=]=              # Equality
        | (?: \|\| | && )    # Logic
        | [-+*/|!(),]        # Operators
        | :>                 # Key value binop
        | ;
        | \[ | ]
        | \.\.\.
        | '[\w.]+            # Symbol
        | `?[a-z][\w.]*`?    # Function/variable names
    )
    \s*
    """,
        re.I | re.S | re.X | re.A,
    )

    def __init__(self, toks):
        if isinstance(toks, str):
            toks = tuple(self.tokenize(toks))
        self.expr = Parser(ExprParserSpec(), iter(toks)).go()

    def __repr__(self):
        return f"<Expr{self.expr!r}>"

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def eval(self, handlers, *args, **kwargs):
        print("\nEVAL", self.expr)
        if not isinstance(self.expr, ExpBase):
            return self.expr
        return self.expr.eval(handlers, *args, **kwargs)

    def __len__(self):
        return len(self.expr)

    def pretty_string(self, depth=0):
        sval = (
            repr(self.expr)
            if not isinstance(self.expr, ExpBase)
            else self.expr.pretty_string(depth=depth + 1)
        )
        pad = " " * (depth + 1) * 2
        return f"<Expr:\n{pad}{sval}\n{pad[:-2]}>"

    FIXUP = {"true": True, "false": False, "...": Ellipsis, "none": None}

    @classmethod
    def fixup_token(cls, t):
        if t == "":
            return t
        t = t.lower()
        val = cls.FIXUP.get(t, Empty)
        if val is not Empty:
            return val
        if t[0] == "`":
            return ExpBinOp(t.strip("`"))
        if t[0] == "'":
            return ExpSym(t[1:])
        if (len(t) > 1 and t[0] == "-" and t[1].isdigit()) or t[0].isdigit():
            return float(t) if "." in t else int(t)
        return ExpOp(t)

    @classmethod
    def tokenize(cls, s):
        yield from (cls.fixup_token(m.group(1)) for m in cls.EXPR_RE.finditer(s))


STATIC_OP_HANDLERS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "**": operator.pow,
    "%": operator.mod,
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "idiv": operator.floordiv,
    "pow": operator.pow,
    "mod": operator.mod,
}


def make_funap(op, args=(), kwargs=None):
    if kwargs is None:
        kwargs = ExpDict()
    argc = len(args)
    if argc > 2 or len(kwargs) or not all(isinstance(v, (int, float)) for v in args):
        return ExpFunAp(op, args, kwargs)
    if argc == 1 and op in "-+":
        return -args[0] if op == "-" else args[0]
    h = STATIC_OP_HANDLERS.get(op)
    if h is None:
        return ExpFunAp(op, args, kwargs)
    return h(*args)


class ExprParserSpec(ParserSpec):
    def __init__(self):
        super().__init__()
        self.populate()

    @staticmethod
    def split_funap_args(toks):
        if not isinstance(toks, (list, tuple)):
            return ExpTuple((toks,)), ExpDict()
        return ExpTuple(t for t in toks if not isinstance(t, ExpKV)), ExpDict({
            str(t.k): t.v for t in toks if isinstance(t, ExpKV)
        })

    @staticmethod
    def null_constant(p, token, bp):
        return token

    @staticmethod
    def null_paren(p, token, bp):
        result = p.parse_until(bp) if p.token != ")" else ExpTuple()
        if p.token == ",":
            p.advance()
        p.expect(")")
        return result

    @staticmethod
    def null_prefixop(p, token, bp):
        val = p.parse_until(bp)
        return make_funap(token, ExpTuple((val,)))

    @classmethod
    def left_binop(cls, p, token, left, bp):
        return make_funap(token, *cls.split_funap_args((left, p.parse_until(bp))))

    @staticmethod
    def left_kv(p, token, left, bp):
        if not isinstance(left, (ExpOp, ExpSym)):
            raise ParseError(f"{left!r} is not a valid key")
        return ExpKV(left, p.parse_until(bp))

    @classmethod
    def left_funcall(cls, p, token, left, bp):
        if not isinstance(left, ExpOp):
            raise ParseError(f"{left!r} is not a valid function/variable name")
        args = []
        while p.lexer and p.token != ")":
            args.append(p.parse_until(1))
            if p.token == ",":
                p.advance()
        p.expect(")")
        return make_funap(left, *cls.split_funap_args(args))

    @staticmethod
    def left_comma(p, token, left, bp):
        if p.token == ")":
            return left if isinstance(left, ExpTuple) else ExpTuple((left,))
        r = p.parse_until(bp)
        return ExpTuple((*left, r) if isinstance(left, ExpTuple) else (left, r))

    @staticmethod
    def left_semicolon(p, token, left, bp):
        if p.token == ")" or p.token is None:
            return (
                left
                if isinstance(left, ExpStatements)
                else ExpStatements(ExpTuple((left,)))
            )
        r = p.parse_until(bp)
        return ExpStatements(
            ExpTuple(*left.statements, r)
            if isinstance(left, ExpStatements)
            else ExpTuple((left, r))
        )

    @staticmethod
    def left_index(p, token, left, bp):
        idx = p.parse_until(0)
        p.expect("]")
        return make_funap("index", ExpTuple((idx, left)))

    @staticmethod
    def get_type(token):
        if isinstance(token, (int, float)):
            return "number"
        if isinstance(token, ExpSym):
            return "sym"
        if isinstance(token, ExpBinOp):
            return "binop"
        if isinstance(token, ExpOp) and token[0].isalpha():
            return "op"
        return token

    def populate(self):
        self.add_left(31, self.left_funcall, ("(",))
        self.add_left(31, self.left_index, ("[",))
        self.add_leftright(29, self.left_binop, ("**",))
        self.add_null(27, self.null_prefixop, ("+", "-", "!"))
        self.add_left(25, self.left_binop, ("*", "/"))
        self.add_left(23, self.left_binop, ("+", "-"))
        self.add_left(22, self.left_binop, ("binop",))
        self.add_left(19, self.left_binop, ("<", ">", "<=", ">="))
        self.add_left(19, self.left_binop, ("==", "!="))
        self.add_left(9, self.left_binop, ("&&",))
        self.add_left(7, self.left_binop, ("||",))
        self.add_left(6, self.left_kv, (":>",))
        self.add_left(5, self.left_semicolon, (";",))
        self.add_left(1, self.left_comma, (",",))
        self.add_null(0, self.null_paren, ("(",))
        self.add_null(
            -1, self.null_constant, ("number", "op", "sym", Ellipsis, True, False, None)
        )
        self.add_null(-1, ParserSpec.null_error, (")", "]"))
