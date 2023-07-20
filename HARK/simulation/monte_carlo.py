"""
Functions to support Monte Carlo simulation of models.
"""

from distribution import Distribution
from inspect import signature
from typing import Any, Callable, Mapping, Union


class Control:
    """
    Should go in HARK.model
    """

    def __init__(self, args):
        pass

def draw_shocks(shocks: Mapping[str, Distribution], N: int):
    """

    Parameters
    ------------
    shocks Mapping[str, Distribution]
        A dictionary-like mapping from shock names to distributions from which to draw

    N: int
        Number of draws from each shock
    """
    return {
        shock : shocks[shock].draw(N)
        for shock in shocks
    }


def sim_one_period(
        dynamics : Mapping[str, Union[Callable, Control]],
        pre : Mapping[str, Any],
        dr : Mapping[str, Callable]
):
    """

    Parameters
    ------------

    dynamics: Mapping[str, Callable]
        Maps variable names to functions from variables to values.
        Can include Controls
        ## TODO: Make collection of equations into a named type


    pre : Mapping[str, Any]
        Bound values for all variables that must be known before beginning the period's dynamics.


    dr : Mapping[str, Callable]
        Decision rules for all the Control variables in the dynamics.
    """
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