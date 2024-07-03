from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr


class Expression:
    def __init__(self, text):
        self.txt
        self.expr = parse_expr(text)
        self.npf = self.func()

        # first derivatives.
        self.grad = {
            sym.__str__(): self.expr.diff(sym) for sym in list(self.expr.free_symbols)
        }

    def func(self):
        return lambdify(list(self.expr.free_symbols), self.expr, "numpy")


def math_text_to_lambda(text):
    """
    Returns a function represented by the given mathematical text.
    """
    expr = parse_expr(text)
    func = lambdify(list(expr.free_symbols), expr, "numpy")
    return func
