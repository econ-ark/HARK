"""
Models in the abstract.
"""
from typing import Any, Callable, Mapping, Optional, Sequence, Union

class Control():
    """
    A class used to indicate that a variable is a control variable.

    Parameters
    -----------
    policy_args : Sequence[str]
        A sequence of variable names, which refer to the values that are the arguments
        to decision rules for this control variable.

    """

    def __init__(
            self,
            policy_args : Sequence[str]
            ):
        self.policy_args = policy_args

class Model:
    """
    An economic model.
    This object should contain the information about an environment's dynamics.

    Parameters
    ----------
    equations : Mapping(str, Union[Callable, Control])
        A mapping from model variable names (as strings) to transition equations governing
        these variables.
    parameters : Optional[Mapping(str, Any)]
        A mapping from parameters names (strings) to parameter values.
    options : Mapping(str, Any)
        A mapping from options (str) to option values.
    """

    def assign_parameters(self, **kwds):
        """
        Assign an arbitrary number of attributes to this agent.

        Parameters
        ----------
        **kwds : keyword arguments
            Any number of keyword arguments of the form key=value.  Each value
            will be assigned to the attribute named in self.

        Returns
        -------
        none
        """
        self.parameters.update(kwds)
        for key in kwds:
            setattr(self, key, kwds[key])

    def get_parameter(self, name: str):
        """
        Returns a parameter of this model

        Parameters
        ----------
        name : string
            The name of the parameter to get

        Returns
        -------
        value :
            The value of the parameter
        """
        return self.parameters[name]

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.parameters == other.parameters) and (self.equations == other.equations)

        return NotImplemented

    def __init__(
            self,
            equations : Mapping[str, Union[Callable, Control]] = {},
            parameters : Optional[Mapping[str, Any]] = None,
            options : Mapping[str, Any] = {}
            ):

        self.equations = equations
        self.options = options
        if not hasattr(self, "parameters"):
            self.parameters = {}
            if parameters is not None:
                self.assign_parameters(**parameters)


    def __str__(self):
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__

        s = f"<{module}.{qualname} object at {hex(id(self))}.\n"
        s += "Parameters:"

        for p in self.parameters:
            s += f"\n{p}: {self.parameters[p]}"

        s += ">"
        return s

    def __repr__(self):
        return self.__str__()
