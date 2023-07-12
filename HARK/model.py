"""
Models in the abstract.
"""

class Control():
    """
    A class used to indicate that a variable is a control variable.
    """

    def __init__(self, policy_args):
        self.policy_args = policy_args

class Model:
    """
    A class with special handling of parameters assignment.
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

    def get_parameter(self, name):
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
            return self.parameters == other.parameters

        return NotImplemented

    def __init__(self, equations = {}, parameters = None, options = {}):

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
