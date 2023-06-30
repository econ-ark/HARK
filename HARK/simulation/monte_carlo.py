from inspect import signature
from typing import Callable, Mapping


class Control:
    """
    Should go in HARK.model
    """

    def __init__(self, args):
        pass

def sim_one_period(
        dynamics : Mapping[str, Callable],
        pre : Mapping,
        dr : Mapping[str, Callable]
):
    vals = pre.copy()

    for varn in dynamics:
        # Using the fact that Python dictionaries are ordered

        feq = dynamics[varn]

        if isinstance(feq, Control):
            vals[varn] = dr[varn](*[
                vals[var]
                for var 
                in signature(dr[varn]).parameters]) # TODO: test for signature match with Control
        else:
            vals[varn] = feq(*[vals[var] for var in signature(feq).parameters])

    return vals