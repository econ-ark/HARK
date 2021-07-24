from HARK import NullFunc, MetricObject
from copy import deepcopy
from types import SimpleNamespace as ns

__all__ = [
    "Stage"
]


class Stage(MetricObject):
    """
    Partial template for solution of single stage of a problem.

    Parameters
    ----------
    stge_kind : dict
        Dictionary with info about this stage
        One built-in entry keeps track of the nature of the stage:
            {'iter_status':'terminal'}: Terminal (last period of existence)
            {'iter_status':'iterator'}: Solution during iteration
            {'iter_status':'finished'}: Stopping requirements are satisfied
                If stopping requirements are satisfied, {'tolerance':tolerance}
                should exist recording what convergence tolerance was satisfied
        Other uses include keeping track of the nature of the next stage
    """
# TODO: Move to core.py (whenever agreement reached on universals of a stage)

    def __init__(
            self,
            stge_kind={}
    ):
        self.stge_kind = stge_kind
