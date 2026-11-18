import warnings

from HARK.distributions import Bernoulli, LogNormal, LogNormalMeanOne
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr
import yaml


class ControlToken:
    """
    Represents a parsed Control variable.
    """

    def __init__(self, args):
        pass


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


def tuple_constructor_from_class(cls):
    def constructor(loader, node):
        value = loader.construct_mapping(node)
        return (cls, value)

    return constructor


def renamed_tag_constructor(old, cls):
    """
    A constructor for the tag !old, named for the class cls before it was
    renamed: it reads as the tag for cls, with a DeprecationWarning.
    """

    def constructor(loader, node):
        warnings.warn(
            f"The tag !{old} is now !{cls.__name__}. The old tag still works and "
            "is not scheduled for removal.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = loader.construct_mapping(node)
        return (cls, value)

    return constructor


def math_text_to_lambda(text):
    """
    Returns a function represented by the given mathematical text.
    """
    expr = parse_expr(text)
    func = lambdify(list(expr.free_symbols), expr, "numpy")
    return func


def harklang_loader():
    """
    A PyYAML loader that supports tags for HARKLang,
    such as random variables and model tags.
    """
    loader = yaml.SafeLoader
    yaml.SafeLoader.add_constructor(
        "!Bernoulli", tuple_constructor_from_class(Bernoulli)
    )
    yaml.SafeLoader.add_constructor(
        "!LogNormalMeanOne", tuple_constructor_from_class(LogNormalMeanOne)
    )
    yaml.SafeLoader.add_constructor(
        "!LogNormal", tuple_constructor_from_class(LogNormal)
    )
    # The tags these classes had before: they still work, with a DeprecationWarning
    yaml.SafeLoader.add_constructor(
        "!MeanOneLogNormal",
        renamed_tag_constructor("MeanOneLogNormal", LogNormalMeanOne),
    )
    yaml.SafeLoader.add_constructor(
        "!Lognormal", renamed_tag_constructor("Lognormal", LogNormal)
    )
    yaml.SafeLoader.add_constructor(
        "!Control", tuple_constructor_from_class(ControlToken)
    )

    return loader
