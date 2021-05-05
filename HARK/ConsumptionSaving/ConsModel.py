from HARK import NullFunc, MetricObject
from copy import deepcopy
from types import SimpleNamespace as ns

__all__ = [
    "ConsumerSolutionGeneric"
]


class ConsumerSolutionGeneric(MetricObject):
    """
    Partial template for solution of single stage of any consumption/
    saving problem. Specific solutions inherit from this and specialize
    to particular problems.  But any solution must include a consumption function.

    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.

    Parameters
    ----------
    cFunc : function
        The consumption function for this period/stage, defined over state(s) `m`

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

#    distance_criteria = ["cFunc"]
    distance_criteria = ["vPfunc"]

    def __init__(
            self,
            cFunc=None,
            stge_kind={}
    ):
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.stge_kind = stge_kind
