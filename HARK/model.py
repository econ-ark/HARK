"""
Tools for crafting models.
"""

from dataclasses import dataclass, field
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


@dataclass
class DBlock:
    """
    Represents a 'block' of model behavior.
    It prioritizes a representation of the dynamics of the block.
    Control variables are designated by the appropriate dynamic rule.

    Parameters
    ----------
    ...
    """

    name: str = ""
    description: str = ""
    shocks: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    dynamics: dict = field(default_factory=dict)
    reward: dict = field(default_factory=dict)
