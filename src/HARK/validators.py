"""
Decorators which can be used for validating arguments passed into decorated functions
"""
from functools import wraps
from inspect import signature


def non_empty(*parameter_names):
    """
    Enforces arguments to parameters passed in have len > 0
    """

    def _decorator(f):
        sig = signature(f)
        # TODO - add validation that parameter names are in signature

        @wraps(f)
        def _inner(*args, **kwargs):
            bindings = sig.bind(*args, **kwargs)
            for parameter_name in parameter_names:
                if not len(bindings.arguments[parameter_name]):
                    raise TypeError(
                        "Expected non-empty argument for parameter {}".format(
                            parameter_name
                        )
                    )
            return f(*args, **kwargs)

        return _inner

    return _decorator
