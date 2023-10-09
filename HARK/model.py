"""
Tools for crafting models.
"""

from HARK.distribution import Distribution


class Aggregate:
    """
    Used to designate a shock as an aggregate shock.
    If so designated, draws from the shock will be scalar rather
    than array valued.
    """
    def __init__(self, dist: Distribution):
        self.dist = dist



class Control:
    """
    Used to designate a variabel that is a control variable.

    Parameters
    ----------
    args : list of str
        The labels of the variables that are in the information set of this control.
    """

    def __init__(self, args):
        pass
