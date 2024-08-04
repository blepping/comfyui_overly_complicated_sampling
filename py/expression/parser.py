class ParseError(Exception):
    pass


# Pratt parsing referenced from https://github.com/andychu/pratt-parsing-demo
class ParserSpec:
    @staticmethod
    def null_error(p, token, bp):
        raise ParseError(f"{token!r} cannot be used in prefix position")

    @staticmethod
    def left_error(p, token, bp):
        raise ParseError(f"{token!r} cannot be used in infix position")

    class LeftInfo:
        def __init__(self, led=None, lbp=0, rbp=0):
            self.led, self.lbp, self.rbp = led or ParserSpec.left_error, lbp, rbp

    class NullInfo:
        def __init__(self, nud=None, bp=0):
            self.nud, self.bp = nud or ParserSpec.null_error, bp

    def __init__(self):
        self.null_lookup = {}
        self.left_lookup = {}

    def add_null(self, bp, nud, tokens):
        for token in tokens:
            self.null_lookup[token] = self.NullInfo(nud, bp)
            if token not in self.left_lookup:
                self.left_lookup[token] = self.LeftInfo()

    def add_led(self, lbp, rbp, led, tokens):
        for token in tokens:
            self.left_lookup[token] = self.LeftInfo(led, lbp, rbp)
            if token not in self.null_lookup:
                self.null_lookup[token] = self.NullInfo(self.null_error)

    def add_left(self, bp, led, tokens):
        return self.add_led(bp, bp, led, tokens)

    def add_leftright(self, bp, led, tokens):
        return self.add_led(bp, bp - 1, led, tokens)

    def lookup(self, token, is_left):
        result = (self.left_lookup if is_left else self.null_lookup).get(token)
        if result is None:
            raise ParseError(f"Unexpected token {token!r}")
        return result

    @staticmethod
    def get_type(token):
        if isinstance(token, (int, float)):
            return "number"
        if isinstance(token, str) and token.isidentifier():
            return "op"
        return token


class Parser:
    def __init__(self, spec, lexer):
        self.spec = spec
        self.lexer = lexer
        self.token = None
        self.token_type = None
        self.pos = -1

    def advance(self):
        if self.lexer is None:
            self.token_type = self.token = None
            return None
        try:
            self.token = next(self.lexer)
            self.token_type = self.spec.get_type(self.token)
            self.pos += 1
        except StopIteration:
            self.token = self.token_type = self.lexer = None
        return self.token

    def expect(self, val):
        if val is not None and (self.lexer is None or self.token != val):
            raise ParseError(f"expected {val!r}, got {self.token!r}")
        return self.advance()

    def parse_until(self, rbp):
        if self.lexer is None:
            raise ParseError("unexpected end of input")
        spec = self.spec
        token, token_type = self.token, self.token_type
        self.advance()
        ni = spec.lookup(token_type, False)
        node = ni.nud(self, token, ni.bp)
        while self.lexer:
            token, token_type = self.token, self.token_type
            li = spec.lookup(token_type, True)
            if rbp >= li.lbp:
                break
            self.advance()
            node = li.led(self, token, node, li.rbp)
        return node

    def go(self):
        self.advance()
        try:
            result = self.parse_until(0)
        except ParseError as exc:
            raise ParseError(
                f"pos {self.pos} at token {self.token!r}: parse error: {exc}"
            ) from None
        if self.lexer:
            raise ParseError(f"pos {self.pos}: unexpected end of input")
        return result
