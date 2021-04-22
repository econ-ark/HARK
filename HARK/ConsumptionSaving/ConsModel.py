from HARK import NullFunc, MetricObject
from copy import deepcopy

__all__ = [
    "ConsumerSolutionGeneric",
    "TrnsPars"
]

# This is needed in order to copy dicts that contain objects constructed
# using __builtins__ because __builtins__ cannot be pickled


class TrnsPars():
    def __init__(self):
        self.about = {
            'TrnsPars': 'Parameters for transition from current to next stage'
        }

    # built-in deepcopy includes builtins which causes recursion problems
    # modify to exclude builtins
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if not k == '__builtins__':
                setattr(result, k, deepcopy(v, memo))
        return result


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
